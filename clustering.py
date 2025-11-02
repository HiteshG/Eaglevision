def get_video_frames_generator(source_path, stride=1):
    cap = cv2.VideoCapture(source_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            yield frame
        frame_idx += 1
    cap.release()

def filter_crops_by_size(crops: List[np.ndarray], min_area: int = 500) -> List[np.ndarray]:
    """
    Filter out crops that are too small (likely false detections or partial players).
    """
    filtered_crops = []
    for crop in crops:
        h, w = crop.shape[:2]
        if h * w >= min_area:
            filtered_crops.append(crop)
    return filtered_crops

def crop_image(frame, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    return frame[y1:y2, x1:x2]

def shrink_boxes(xyxy: np.ndarray, scale: float) -> np.ndarray:
    """
    Shrinks bounding boxes by a given scale factor while keeping centers fixed.
    """
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * scale, (y2 - y1) * scale

    new_x1, new_y1 = cx - w / 2, cy - h / 2
    new_x2, new_y2 = cx + w / 2, cy + h / 2

    return np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)


def plot_images_grid(images, grid_size=(10, 10), size=(10, 10)):
    rows, cols = grid_size
    total = rows * cols
    images = images[:total]  # limit to grid size

    fig, axes = plt.subplots(rows, cols, figsize=size)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)  # convert BGR→RGB
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.show()


frame_generator = get_video_frames_generator(CONFIG['SOURCE_VIDEO_PATH'], stride=STRIDE)
crops = []

for frame in tqdm(frame_generator, desc="collecting crops"):
  results = model(frame, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD ,verbose=False)[0]

  # Extract boxes and scores for the given class
  boxes = []
  scores = []

  for box in results.boxes:
      if int(box.cls[0]) == PLAYER_ID:
          xyxy = box.xyxy[0].tolist()
          boxes.append(xyxy)

  # Shrink bounding boxes
  xyxy_arr = np.array(boxes)
  shrunk_boxes = shrink_boxes(xyxy_arr, scale=SHRINK_SCALE)

  # Crop
  players_crops = [crop_image(frame, xyxy) for xyxy in shrunk_boxes]
  crops += players_crops

# crops = filter_crops_by_size(crops, min_area=MIN_CROP_AREA)

print(f"Collected {len(crops)} player crops.")
plot_images_grid(images=crops, grid_size=(10, 10), size=(12, 12))


from typing import Generator, Iterable, List, TypeVar

import numpy as np
from PIL import Image
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    Optimized for NVIDIA T4 GPU with better memory utilization.
    """
    def __init__(self, device: str = 'cuda', batch_size: int = 256):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images (256 for better T4 utilization).
       """
        self.device = device
        self.batch_size = batch_size
        self.use_amp = device == 'cuda'

        # Load model and optimize for inference
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.features_model.eval()

        # Use half precision for T4 GPU
        if self.use_amp:
            self.features_model = self.features_model.half()

        # Use simpler compile mode for T4 or disable if still issues
        # if hasattr(torch, 'compile'):
        #     self.features_model = torch.compile(self.features_model, mode='reduce-overhead')

        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [Image.fromarray(crop[..., ::-1]) for crop in crops]
        batches = list(create_batches(crops, self.batch_size))
        data = []

        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(
                    images=batch, return_tensors="pt")

                # Move to device with non-blocking transfer
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

                # Convert to half precision for T4 optimization
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.features_model(**inputs)
                        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                else:
                    outputs = self.features_model(**inputs)
                    embeddings = torch.mean(outputs.last_hidden_state, dim=1)

                # Convert back to float32 for sklearn compatibility
                embeddings = embeddings.float().cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)

# Training the classifier
team_classifier = TeamClassifier()
team_classifier.fit(crops)
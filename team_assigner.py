"""
Team Assignment Module.
Assigns players to teams using color-based or embedding-based clustering.
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from typing import Dict, List, Tuple, Generator, Iterable, TypeVar
from PIL import Image
import torch
from tqdm import tqdm

try:
    import umap
    from transformers import AutoProcessor, SiglipVisionModel
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("Warning: transformers or umap-learn not installed. Embedding-based team assignment unavailable.")

from config import TeamAssignerConfig

V = TypeVar("V")


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """Generate batches from a sequence with a specified batch size."""
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


def shrink_boxes(xyxy: np.ndarray, scale: float) -> np.ndarray:
    """Shrinks bounding boxes by a given scale factor while keeping centers fixed."""
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    
    new_x1, new_y1 = cx - w / 2, cy - h / 2
    new_x2, new_y2 = cx + w / 2, cy + h / 2
    
    return np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)


def crop_image(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Crop image using bounding box coordinates."""
    x1, y1, x2, y2 = map(int, xyxy)
    # Clip to frame boundaries
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


def resolve_goalkeepers_team_id(
    players_xy: np.ndarray,
    players_team_id: np.ndarray,
    goalkeepers_xy: np.ndarray
) -> np.ndarray:
    """
    Assign team IDs to goalkeepers using distance-weighted voting.
    
    Args:
        players_xy: (N, 2) array of player positions
        players_team_id: (N,) array of player team IDs
        goalkeepers_xy: (M, 2) array of goalkeeper positions
        
    Returns:
        (M,) array of goalkeeper team IDs
    """
    if len(players_xy) == 0 or len(goalkeepers_xy) == 0:
        return np.zeros(len(goalkeepers_xy), dtype=int)
    
    # Vectorized distance computation
    # Shape: (num_goalkeepers, num_players)
    distances = np.linalg.norm(
        goalkeepers_xy[:, np.newaxis, :] - players_xy[np.newaxis, :, :],
        axis=2
    )
    
    goalkeeper_teams = []
    nearest_k = min(5, len(players_xy))
    
    for gk_idx in range(len(goalkeepers_xy)):
        gk_distances = distances[gk_idx]
        
        # Get k nearest players
        nearest_indices = np.argpartition(gk_distances, nearest_k-1)[:nearest_k]
        nearest_distances = gk_distances[nearest_indices]
        nearest_teams = players_team_id[nearest_indices]
        
        # Weighted voting: closer players have more influence
        weights = 1 / (nearest_distances + 1e-6)  # Avoid division by zero
        
        # Sum weights per team
        unique_teams = np.unique(nearest_teams)
        team_weights = {}
        for team in unique_teams:
            team_mask = nearest_teams == team
            team_weights[team] = np.sum(weights[team_mask])
        
        # Select team with highest weight
        majority_team = max(team_weights, key=team_weights.get)
        goalkeeper_teams.append(int(majority_team))
    
    return np.array(goalkeeper_teams, dtype=int)


class ColorBasedTeamAssigner:
    """Color-based team assignment using KMeans clustering."""
    
    def __init__(self, config: TeamAssignerConfig):
        self.config = config
        self.color_ranges = config.color_ranges
        
    def assign_teams(
        self,
        frames: List[np.ndarray],
        detections_per_frame: List[Dict]
    ) -> Dict[int, int]:
        """Assign teams to all detected players using color clustering."""
        print("Assigning teams based on jersey colors...")
        
        # First pass: collect color frequencies for each player
        player_color_counts = {}
        
        for frame, detections in zip(frames, detections_per_frame):
            if "Player" not in detections:
                continue
            
            players = detections["Player"]
            all_bboxes = [item["bbox"] for item in players.values()]
            
            for player_id, detection in players.items():
                bbox = detection["bbox"]
                x1, y1, x2, y2 = bbox
                
                # Calculate overlap with other detections
                overlap_ratio = self._calculate_max_overlap_ratio(
                    bbox, all_bboxes
                )
                
                # Skip heavily overlapped detections
                if overlap_ratio > self.config.overlap_threshold:
                    continue
                
                # Extract player crop
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Detect dominant colors
                color_counts = self._detect_colors(crop)
                
                # Accumulate color counts for this player
                if player_id not in player_color_counts:
                    player_color_counts[player_id] = {}
                
                for color, count in color_counts:
                    if color not in player_color_counts[player_id]:
                        player_color_counts[player_id][color] = 0
                    player_color_counts[player_id][color] += count * (1 - overlap_ratio)
        
        # Determine dominant color for each player
        player_dominant_colors = {}
        for player_id, color_counts in player_color_counts.items():
            if color_counts:
                dominant_color = max(color_counts, key=color_counts.get)
                player_dominant_colors[player_id] = dominant_color
        
        # Find the two most common colors
        all_colors = list(player_dominant_colors.values())
        if len(all_colors) < 2:
            print("Warning: Not enough players detected for team assignment")
            return {pid: 0 for pid in player_dominant_colors.keys()}
        
        color_frequency = Counter(all_colors)
        most_common_colors = color_frequency.most_common(2)
        team_colors = [color for color, _ in most_common_colors]
        
        # Create color to team mapping
        color_to_team = {color: i for i, color in enumerate(team_colors)}
        
        # Second pass: assign teams
        team_mapping = {}
        for player_id, dominant_color in player_dominant_colors.items():
            if dominant_color in color_to_team:
                team_mapping[player_id] = color_to_team[dominant_color]
            else:
                # Outlier: assign based on secondary colors
                color_counts = player_color_counts[player_id]
                team_color_counts = [
                    (color, count) 
                    for color, count in color_counts.items() 
                    if color in color_to_team
                ]
                
                if team_color_counts:
                    best_color = max(team_color_counts, key=lambda x: x[1])[0]
                    team_mapping[player_id] = color_to_team[best_color]
                else:
                    team_mapping[player_id] = 0
        
        print(f"Assigned {len(team_mapping)} players to teams")
        print(f"Team 0: {sum(1 for t in team_mapping.values() if t == 0)} players")
        print(f"Team 1: {sum(1 for t in team_mapping.values() if t == 1)} players")
        
        return team_mapping
    
    def _calculate_max_overlap_ratio(
        self,
        bbox: List[int],
        all_bboxes: List[List[int]]
    ) -> float:
        """Calculate maximum overlap ratio with other bounding boxes."""
        x1, y1, x2, y2 = bbox
        curr_area = (x2 - x1) * (y2 - y1)
        
        if curr_area == 0:
            return 0.0
        
        max_overlap = 0.0
        
        for other_bbox in all_bboxes:
            if other_bbox == bbox:
                continue
            
            x1_, y1_, x2_, y2_ = other_bbox
            
            x_overlap = max(0, min(x2, x2_) - max(x1, x1_))
            y_overlap = max(0, min(y2, y2_) - max(y1, y1_))
            overlap_area = x_overlap * y_overlap
            
            overlap_ratio = overlap_area / curr_area
            max_overlap = max(max_overlap, overlap_ratio)
        
        return max_overlap
    
    def _detect_colors(self, image: np.ndarray) -> List[Tuple[str, int]]:
        """Detect colors in an image crop using HSV color ranges."""
        if image.size == 0:
            return []
        
        # Convert to RGB for KMeans
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Segment player from background
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=3, init="k-means++")
        kmeans.fit(rgb_image.reshape(-1, 3))
        labels = kmeans.labels_.reshape(image.shape[:2])
        
        # Determine background cluster
        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        background_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 if background_cluster == 0 else 0
        
        # Create player mask
        player_mask = (labels == player_cluster).astype(np.uint8) * 255
        
        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=player_mask)
        
        # Count pixels in each color range
        color_counts = {}
        
        for color_name, (lower, upper) in self.color_ranges.items():
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)
            
            color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            color_mask = cv2.bitwise_and(color_mask, player_mask)
            
            count = cv2.countNonZero(color_mask)
            if count > 0:
                color_counts[color_name] = count
        
        # Combine red and red2
        if "red2" in color_counts:
            color_counts["red"] = color_counts.get("red", 0) + color_counts.pop("red2")
        
        return sorted(color_counts.items(), key=lambda x: x[1], reverse=True)


class EmbeddingBasedTeamAssigner:
    """Embedding-based team assignment using SigLip + UMAP + KMeans."""
    
    def __init__(self, config: TeamAssignerConfig):
        if not EMBEDDING_AVAILABLE:
            raise ImportError(
                "Embedding-based team assignment requires transformers and umap-learn. "
                "Install with: pip install transformers umap-learn pillow"
            )
        
        self.config = config
        self.device = config.device
        self.batch_size = config.embedding_batch_size
        self.use_amp = config.device == 'cuda'
        
        print(f"Loading embedding model on device: {config.device}")
        
        # Load SigLip model
        self.features_model = SiglipVisionModel.from_pretrained(
            config.embedding_model
        ).to(config.device)
        self.features_model.eval()
        
        # Use half precision for GPU
        if self.use_amp:
            self.features_model = self.features_model.half()
        
        self.processor = AutoProcessor.from_pretrained(config.embedding_model)
        self.reducer = umap.UMAP(n_components=3, random_state=42)
        self.cluster_model = KMeans(n_clusters=config.n_clusters, random_state=42, init="k-means++")
        
    def assign_teams(
        self,
        frames: List[np.ndarray],
        detections_per_frame: List[Dict]
    ) -> Dict[int, int]:
        """Assign teams using embedding-based clustering."""
        print("Assigning teams using SigLip embeddings...")
        
        # Collect player crops
        crops_per_player = {}
        
        for frame_idx, (frame, detections) in enumerate(zip(frames, detections_per_frame)):
            # Skip frames based on stride
            if frame_idx % self.config.stride != 0:
                continue
            
            if "Player" not in detections:
                continue
            
            players = detections["Player"]
            
            for player_id, detection in players.items():
                bbox = detection["bbox"]
                x1, y1, x2, y2 = bbox
                
                # Shrink bbox to focus on jersey
                bbox_array = np.array([[x1, y1, x2, y2]])
                shrunk_bbox = shrink_boxes(bbox_array, self.config.shrink_scale)[0]
                
                # Crop image
                crop = crop_image(frame, shrunk_bbox)
                
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue
                
                if player_id not in crops_per_player:
                    crops_per_player[player_id] = []
                
                crops_per_player[player_id].append(crop)
        
        # Filter players with insufficient crops
        crops_per_player = {
            pid: crops for pid, crops in crops_per_player.items()
            if len(crops) >= 3
        }
        
        if len(crops_per_player) < 2:
            print("Warning: Not enough player crops for embedding-based assignment")
            return {pid: 0 for pid in crops_per_player.keys()}
        
        print(f"Collected crops for {len(crops_per_player)} players")
        
        # Extract embeddings for all crops
        all_crops = []
        player_ids = []
        for player_id, crops in crops_per_player.items():
            all_crops.extend(crops)
            player_ids.extend([player_id] * len(crops))
        
        print(f"Extracting embeddings for {len(all_crops)} crops...")
        embeddings = self._extract_features(all_crops)
        
        # Average embeddings per player
        player_embeddings = {}
        for player_id in crops_per_player.keys():
            mask = np.array(player_ids) == player_id
            player_embeddings[player_id] = embeddings[mask].mean(axis=0)
        
        # Stack embeddings
        player_ids_list = list(player_embeddings.keys())
        embeddings_array = np.stack([player_embeddings[pid] for pid in player_ids_list])
        
        # UMAP + KMeans
        print("Clustering with UMAP + KMeans...")
        projections = self.reducer.fit_transform(embeddings_array)
        labels = self.cluster_model.fit_predict(projections)
        
        # Create team mapping
        team_mapping = {
            player_ids_list[i]: int(labels[i])
            for i in range(len(player_ids_list))
        }
        
        print(f"Assigned {len(team_mapping)} players to teams")
        print(f"Team 0: {sum(1 for t in team_mapping.values() if t == 0)} players")
        print(f"Team 1: {sum(1 for t in team_mapping.values() if t == 1)} players")
        
        return team_mapping
    
    def _extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract features using SigLip model."""
        # Convert to PIL images
        crops_pil = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in crops]
        batches = list(create_batches(crops_pil, self.batch_size))
        data = []
        
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(images=batch, return_tensors="pt")
                
                # Move to device
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                
                # Extract embeddings
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.features_model(**inputs)
                        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                else:
                    outputs = self.features_model(**inputs)
                    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                
                # Convert to float32 for sklearn
                embeddings = embeddings.float().cpu().numpy()
                data.append(embeddings)
        
        return np.concatenate(data)


def create_team_assigner(config: TeamAssignerConfig):
    """Factory function to create appropriate team assigner."""
    if config.team_method == "color":
        return ColorBasedTeamAssigner(config)
    elif config.team_method == "embedding":
        return EmbeddingBasedTeamAssigner(config)
    else:
        raise ValueError(f"Unknown team method: {config.team_method}. Use 'color' or 'embedding'")
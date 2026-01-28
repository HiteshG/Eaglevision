"""
GTA-Link integration for Football Tracker.
Provides ReID-based tracklet refinement using split and merge operations.
"""
import os
import sys
import tempfile
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Add gta-link to path for imports
GTA_LINK_PATH = os.path.join(os.path.dirname(__file__), "gta-link")
if os.path.exists(GTA_LINK_PATH):
    sys.path.insert(0, GTA_LINK_PATH)
    sys.path.insert(0, os.path.join(GTA_LINK_PATH, "reid"))


@dataclass
class GTALinkConfig:
    """Configuration for GTA-Link post-processing."""
    model_path: str = "gta-link/reid_checkpoints/sports_model.pth.tar-60"
    # Split parameters (DBSCAN)
    eps: float = 0.6
    min_samples: int = 10
    max_k: int = 3
    min_len: int = 100
    # Merge parameters
    merge_dist_thres: float = 0.4
    spatial_factor: float = 1.0
    # Processing options
    use_split: bool = True
    use_connect: bool = True
    batch_size: int = 64  # For feature extraction


def convert_to_mot_format(
    detections_per_frame: List[Dict],
    output_path: str,
    classes: List[str] = None
) -> Dict[int, int]:
    """
    Convert Football Tracker detections to MOT format.

    MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
    Note: MOT uses 1-indexed frames and xywh bbox format.

    Args:
        detections_per_frame: List of detection dicts per frame
        output_path: Path to save MOT .txt file
        classes: Classes to include (default: ["Player", "Goalkeeper"])

    Returns:
        Dict mapping MOT track_id to original track_id
    """
    if classes is None:
        classes = ["Player", "Goalkeeper"]

    lines = []
    # Track ID mapping: we need unique IDs across all classes
    # Original format has separate ID spaces per class
    original_ids = {}  # (class_name, original_id) -> mot_id
    mot_id_counter = 1

    for frame_idx, detections in enumerate(detections_per_frame):
        frame_num = frame_idx + 1  # MOT uses 1-indexed frames

        for class_name in classes:
            if class_name not in detections:
                continue

            for track_id, detection in detections[class_name].items():
                # Get or assign MOT ID
                key = (class_name, track_id)
                if key not in original_ids:
                    original_ids[key] = mot_id_counter
                    mot_id_counter += 1

                mot_id = original_ids[key]

                # Convert xyxy to xywh
                bbox = detection["bbox"]
                x1, y1, x2, y2 = bbox
                x, y = x1, y1
                w, h = x2 - x1, y2 - y1

                conf = detection.get("confidence", 1.0)

                # MOT format line
                line = f"{frame_num},{mot_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                lines.append(line)

    # Sort by frame number
    lines.sort(key=lambda l: int(l.split(",")[0]))

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(lines)

    # Return reverse mapping: mot_id -> (class_name, original_id)
    mot_to_original = {v: k for k, v in original_ids.items()}
    return mot_to_original


def convert_from_mot_format(
    mot_path: str,
    original_detections: List[Dict],
    team_mapping: Dict[int, int],
    mot_to_original: Dict[int, Tuple[str, int]],
    classes: List[str] = None
) -> Tuple[List[Dict], Dict[int, int]]:
    """
    Convert refined MOT results back to Football Tracker format.

    Args:
        mot_path: Path to refined MOT .txt file
        original_detections: Original detections (for Ball and structure reference)
        team_mapping: Original team assignments
        mot_to_original: Mapping from MOT IDs to (class_name, original_id)
        classes: Classes that were processed

    Returns:
        Tuple of (refined_detections, updated_team_mapping)
    """
    if classes is None:
        classes = ["Player", "Goalkeeper"]

    # Load refined MOT results
    mot_data = np.genfromtxt(mot_path, dtype=float, delimiter=",")
    if mot_data.ndim == 1:
        mot_data = mot_data.reshape(1, -1)

    # Group by frame
    frame_to_detections = defaultdict(list)
    for row in mot_data:
        frame_num = int(row[0])
        track_id = int(row[1])
        x, y, w, h = row[2:6]
        conf = row[6]
        frame_to_detections[frame_num].append({
            "track_id": track_id,
            "bbox": [x, y, x + w, y + h],  # Convert back to xyxy
            "confidence": conf
        })

    # Build new detections and team mapping
    refined_detections = []
    new_team_mapping = {}

    # Map refined IDs to teams: if a refined ID was split from an original,
    # propagate the team assignment
    refined_to_team = {}

    # First pass: build refined ID to team mapping based on IoU matching
    # For each refined detection, find the best matching original detection
    for frame_idx, original_frame in enumerate(original_detections):
        frame_num = frame_idx + 1

        if frame_num not in frame_to_detections:
            continue

        refined_dets = frame_to_detections[frame_num]

        # Collect all original bboxes in this frame
        original_bboxes = []
        original_info = []
        for class_name in classes:
            if class_name not in original_frame:
                continue
            for orig_id, det in original_frame[class_name].items():
                original_bboxes.append(det["bbox"])
                original_info.append((class_name, orig_id))

        if not original_bboxes:
            continue

        original_bboxes = np.array(original_bboxes)

        # Match refined to original by IoU
        for ref_det in refined_dets:
            ref_bbox = np.array(ref_det["bbox"])
            ious = _compute_iou(ref_bbox, original_bboxes)
            best_idx = np.argmax(ious)

            if ious[best_idx] > 0.5:  # Good match
                class_name, orig_id = original_info[best_idx]
                # Check if original had a team assignment
                if orig_id in team_mapping:
                    refined_to_team[ref_det["track_id"]] = team_mapping[orig_id]

    # Second pass: build refined detections
    for frame_idx, original_frame in enumerate(original_detections):
        frame_num = frame_idx + 1

        new_frame = {
            "Player": {},
            "Goalkeeper": {},
            "Ball": original_frame.get("Ball", {}),  # Preserve Ball unchanged
        }

        if frame_num in frame_to_detections:
            refined_dets = frame_to_detections[frame_num]

            # Determine class for each refined detection based on original mapping
            for ref_det in refined_dets:
                track_id = ref_det["track_id"]

                # Try to determine class from mot_to_original
                # Note: after refinement, IDs may have changed
                class_name = "Player"  # Default

                # Check if this track_id exists in original mapping
                if track_id in mot_to_original:
                    class_name, _ = mot_to_original[track_id]

                # Calculate bottom_center
                x1, y1, x2, y2 = ref_det["bbox"]
                bottom_center = ((x1 + x2) / 2, y2)

                new_frame[class_name][track_id] = {
                    "bbox": ref_det["bbox"],
                    "confidence": ref_det["confidence"],
                    "bottom_center": bottom_center
                }

                # Update team mapping
                if track_id in refined_to_team:
                    new_team_mapping[track_id] = refined_to_team[track_id]

        refined_detections.append(new_frame)

    return refined_detections, new_team_mapping


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between a box and array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = box_area + boxes_area - intersection

    return intersection / (union + 1e-6)


def save_frames_for_gta_link(frames: List[np.ndarray], output_dir: str) -> str:
    """
    Save frames in MOT dataset format for GTA-Link processing.

    Args:
        frames: List of BGR frames
        output_dir: Base output directory

    Returns:
        Path to frames directory (output_dir/video/img1/)
    """
    import cv2

    frames_dir = os.path.join(output_dir, "video", "img1")
    os.makedirs(frames_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        # MOT format uses 1-indexed, 6-digit zero-padded filenames
        filename = f"{i + 1:06d}.jpg"
        filepath = os.path.join(frames_dir, filename)
        cv2.imwrite(filepath, frame)

    return frames_dir


def run_gta_link_phase1(
    frames: List[np.ndarray],
    mot_path: str,
    config: GTALinkConfig
) -> Dict:
    """
    Run GTA-Link Phase 1: Extract ReID features from tracking results.

    Args:
        frames: List of BGR video frames
        mot_path: Path to MOT format tracking results
        config: GTA-Link configuration

    Returns:
        Dict of {track_id: Tracklet} with features
    """
    from Tracklet import Tracklet
    from torchreid.utils import FeatureExtractor

    # Setup transforms
    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load feature extractor
    model_path = config.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), model_path)

    extractor = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path=model_path,
        device=device
    )

    # Load MOT results
    track_res = np.genfromtxt(mot_path, dtype=float, delimiter=",")
    if track_res.ndim == 1:
        track_res = track_res.reshape(1, -1)

    last_frame = int(track_res[-1, 0])
    seq_tracks = {}

    print(f"  Extracting ReID features for {last_frame} frames...")

    for frame_id in range(1, last_frame + 1):
        if frame_id % 100 == 0:
            print(f"    Frame {frame_id}/{last_frame}")

        # Get frame (0-indexed)
        frame_idx = frame_id - 1
        if frame_idx >= len(frames):
            continue

        frame = frames[frame_idx]
        img = Image.fromarray(frame[:, :, ::-1])  # BGR to RGB

        # Get detections for this frame
        inds = track_res[:, 0] == frame_id
        frame_res = track_res[inds]

        if len(frame_res) == 0:
            continue

        input_batch = None
        tid2idx = {}

        for idx, row in enumerate(frame_res):
            frame_num, track_id = int(row[0]), int(row[1])
            l, t, w, h = row[2:6]
            score = row[6]

            bbox = [l, t, w, h]

            # Update tracklet
            if track_id not in seq_tracks:
                seq_tracks[track_id] = Tracklet(track_id, frame_num, score, bbox)
            else:
                seq_tracks[track_id].append_det(frame_num, score, bbox)

            tid2idx[track_id] = idx

            # Crop and transform
            l, t = max(0, int(l)), max(0, int(t))
            r, b = min(img.width, int(l + w)), min(img.height, int(t + h))

            if r <= l or b <= t:
                continue

            crop = img.crop((l, t, r, b)).convert("RGB")
            crop_tensor = val_transforms(crop).unsqueeze(0)

            if input_batch is None:
                input_batch = crop_tensor
            else:
                input_batch = torch.cat([input_batch, crop_tensor], dim=0)

        # Extract features in batches
        if input_batch is not None:
            # Process in batches if needed
            all_feats = []
            for i in range(0, len(input_batch), config.batch_size):
                batch = input_batch[i:i + config.batch_size]
                if device == "cuda":
                    batch = batch.to(device)
                with torch.no_grad():
                    feats = extractor(batch)
                all_feats.append(feats.cpu().numpy())

            feats = np.concatenate(all_feats, axis=0)

            # Update tracklets with features
            for tid, idx in tid2idx.items():
                if idx < len(feats):
                    feat = feats[idx]
                    feat = feat / (np.linalg.norm(feat) + 1e-6)
                    seq_tracks[tid].append_feat(feat)

    print(f"  Extracted features for {len(seq_tracks)} tracklets")
    return seq_tracks


def run_gta_link_phase2(
    tracklets: Dict,
    config: GTALinkConfig
) -> Dict:
    """
    Run GTA-Link Phase 2: Split and merge tracklets.

    Args:
        tracklets: Dict of {track_id: Tracklet} from Phase 1
        config: GTA-Link configuration

    Returns:
        Dict of refined {track_id: Tracklet}
    """
    from refine_tracklets import (
        split_tracklets,
        merge_tracklets,
        get_distance_matrix,
        get_spatial_constraints
    )

    refined = tracklets

    # Step 1: Split tracklets with multiple identities
    if config.use_split:
        print(f"  Splitting tracklets (eps={config.eps}, min_samples={config.min_samples})...")
        num_before = len(refined)
        refined = split_tracklets(
            refined,
            eps=config.eps,
            max_k=config.max_k,
            min_samples=config.min_samples,
            len_thres=config.min_len
        )
        print(f"    {num_before} -> {len(refined)} tracklets")

    # Step 2: Merge fragmented tracklets
    if config.use_connect:
        print(f"  Merging tracklets (threshold={config.merge_dist_thres})...")
        num_before = len(refined)

        # Get spatial constraints
        max_x_range, max_y_range = get_spatial_constraints(refined, config.spatial_factor)

        # Compute distance matrix
        dist_matrix = get_distance_matrix(refined)

        # Merge
        seq2dist = {}  # Required by merge_tracklets but not used
        refined = merge_tracklets(
            refined,
            seq2dist,
            dist_matrix,
            seq_name="video",
            max_x_range=max_x_range,
            max_y_range=max_y_range,
            merge_dist_thres=config.merge_dist_thres
        )
        print(f"    {num_before} -> {len(refined)} tracklets")

    return refined


def save_tracklets_to_mot(tracklets: Dict, output_path: str):
    """
    Save refined tracklets back to MOT format.

    Args:
        tracklets: Dict of {track_id: Tracklet}
        output_path: Path to save MOT .txt file
    """
    results = []

    for i, tid in enumerate(sorted(tracklets.keys())):
        track = tracklets[tid]
        new_tid = i + 1  # Re-index from 1

        for instance_idx, frame_id in enumerate(track.times):
            bbox = track.bboxes[instance_idx]
            score = track.scores[instance_idx] if instance_idx < len(track.scores) else 1.0

            results.append([
                frame_id, new_tid,
                bbox[0], bbox[1], bbox[2], bbox[3],
                score, -1, -1, -1
            ])

    # Sort by frame
    results.sort(key=lambda x: (x[0], x[1]))

    # Write
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for line in results:
            f.write(f"{line[0]},{line[1]},{line[2]:.2f},{line[3]:.2f},{line[4]:.2f},{line[5]:.2f},{line[6]:.4f},{line[7]},{line[8]},{line[9]}\n")


def run_gta_link_refinement(
    detections_per_frame: List[Dict],
    frames: List[np.ndarray],
    team_mapping: Dict[int, int],
    config: GTALinkConfig = None
) -> Tuple[List[Dict], Dict[int, int]]:
    """
    Main entry point: Run GTA-Link refinement on tracking results.

    Args:
        detections_per_frame: Football Tracker detection format
        frames: List of BGR video frames
        team_mapping: Current team assignments (can be empty {})
        config: GTA-Link configuration (uses defaults if None)

    Returns:
        Tuple of (refined_detections, updated_team_mapping)
    """
    if config is None:
        config = GTALinkConfig()

    # Check if model exists
    model_path = config.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), model_path)

    if not os.path.exists(model_path):
        print(f"  WARNING: GTA-Link model not found at {model_path}")
        print("  Skipping GTA-Link refinement.")
        return detections_per_frame, team_mapping

    print("\n" + "-" * 40)
    print("GTA-Link Tracklet Refinement")
    print("-" * 40)

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="gta_link_")

    try:
        # Step 1: Convert to MOT format
        print("Phase 0: Converting to MOT format...")
        mot_input_path = os.path.join(temp_dir, "tracking", "video.txt")
        mot_to_original = convert_to_mot_format(
            detections_per_frame,
            mot_input_path,
            classes=["Player", "Goalkeeper"]
        )
        print(f"  Converted {len(mot_to_original)} unique track IDs")

        # Step 2: Phase 1 - Feature extraction
        print("Phase 1: Extracting ReID features...")
        tracklets = run_gta_link_phase1(frames, mot_input_path, config)

        # Step 3: Phase 2 - Refinement
        print("Phase 2: Refining tracklets...")
        refined_tracklets = run_gta_link_phase2(tracklets, config)

        # Step 4: Save refined results
        mot_output_path = os.path.join(temp_dir, "refined", "video.txt")
        save_tracklets_to_mot(refined_tracklets, mot_output_path)

        # Step 5: Convert back to Football Tracker format
        print("Phase 3: Converting back to tracker format...")
        refined_detections, new_team_mapping = convert_from_mot_format(
            mot_output_path,
            detections_per_frame,
            team_mapping,
            mot_to_original,
            classes=["Player", "Goalkeeper"]
        )

        print("-" * 40)
        print(f"GTA-Link complete: {len(mot_to_original)} -> {len(refined_tracklets)} tracklets")
        print("-" * 40 + "\n")

        return refined_detections, new_team_mapping

    except Exception as e:
        print(f"  ERROR in GTA-Link: {e}")
        print("  Returning original detections.")
        import traceback
        traceback.print_exc()
        return detections_per_frame, team_mapping

    finally:
        # Cleanup temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

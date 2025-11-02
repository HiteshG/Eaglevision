"""
Object Tracker Module.
Wraps BoTSORT for tracking detected objects across frames.
Designed to be easily swappable with other tracking algorithms.
"""
import numpy as np
from boxmot import BotSort
from pathlib import Path
from typing import Dict, List, Tuple
from config import TrackerConfig


class ObjectTracker:
    """
    Wrapper for BoTSORT tracker.
    Can be easily replaced with ByteTrack, StrongSORT, or custom trackers.
    """
    
    def __init__(self, config: TrackerConfig):
        """
        Initialize tracker with configuration.
        
        Args:
            config: TrackerConfig object with tracker settings
        """
        self.config = config
        print(f"Loading tracker: {config.tracker_type} on device: {config.device}")
        
        # Convert device string to format expected by boxmot
        if config.device == "cuda":
            device = 0
        elif config.device == "mps":
            device = "cpu"  # BoTSORT doesn't support MPS
        else:
            device = config.device
            
        # Initialize BoTSORT
        self.tracker = BotSort(
            reid_weights=Path(config.reid_weights),
            device=device,
            half=False
        )
        
    def update(
        self,
        detections: np.ndarray,
        frame: np.ndarray
    ) -> List[np.ndarray]:
        """
        Update tracker with new detections.
        
        Args:
            detections: (N, 6) array of [x1, y1, x2, y2, conf, class]
            frame: Current frame in BGR format (H, W, 3)
            
        Returns:
            List of tracked objects, each as:
            [x1, y1, x2, y2, track_id, conf, class_idx, detection_idx]
        """
        try:
            tracks = self.tracker.update(detections, frame)
            return tracks
        except Exception as e:
            print(f"Tracker error: {e}")
            return np.array([])
    
    def organize_tracks(
        self,
        tracks: np.ndarray,
        class_names: Dict[int, str],
        confidence_threshold: float,
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Dict[int, Dict]]:
        """
        Organize tracked objects by class and ID.
        
        Args:
            tracks: Array of tracked objects
            class_names: Mapping from class index to class name
            confidence_threshold: Minimum confidence threshold
            frame_shape: (height, width) of the frame
            
        Returns:
            Dictionary organized by class name and track ID
        """
        height, width = frame_shape
        result = {
            "Player": {},
            "Goalkeeper": {}
        }
        
        if len(tracks) == 0:
            return result
        
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, class_idx, _ = track
            
            # Convert to proper types
            class_idx = int(class_idx)
            track_id = int(track_id)
            conf = float(conf)
            
            # Get class name
            class_name = class_names.get(class_idx, None)
            
            # Skip if not a class we track
            if class_name not in result:
                continue
            
            # Skip low confidence
            if conf < confidence_threshold:
                continue
            
            # Clip coordinates
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            
            # Calculate bottom center
            bottom_center = [int((x1 + x2) / 2), y2]
            
            result[class_name][track_id] = {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "bottom_center": bottom_center
            }
        
        return result
    
    def reset(self):
        """Reset tracker state."""
        self.tracker = BotSort(
            reid_weights=Path(self.config.reid_weights),
            device=self.config.device if self.config.device != "mps" else "cpu",
            half=False
        )
    
    def __repr__(self):
        return f"ObjectTracker(type={self.config.tracker_type})"


class SimpleTracker:
    """
    Simple IoU-based tracker as fallback.
    Can be used if BoTSORT is not available.
    """
    
    def __init__(self):
        self.next_id = 0
        self.tracks = {}
        self.iou_threshold = 0.3
        
    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Simple IoU-based tracking."""
        if len(detections) == 0:
            return np.array([])
        
        boxes = detections[:, :4]
        confidences = detections[:, 4]
        classes = detections[:, 5]
        
        results = []
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
            # Find best matching track
            best_iou = 0
            best_id = None
            
            for track_id, track_box in self.tracks.items():
                iou = self._calculate_iou(box, track_box)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_id = track_id
            
            # Assign ID
            if best_id is None:
                track_id = self.next_id
                self.next_id += 1
            else:
                track_id = best_id
            
            self.tracks[track_id] = box
            
            # Format: [x1, y1, x2, y2, id, conf, class, det_idx]
            results.append([
                box[0], box[1], box[2], box[3],
                track_id, conf, cls, i
            ])
        
        return np.array(results)
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def reset(self):
        """Reset tracker."""
        self.next_id = 0
        self.tracks = {}

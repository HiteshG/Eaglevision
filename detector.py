"""
Object Detector Module.
Wraps YOLO for detecting players, goalkeepers, and ball.
Designed to be easily swappable with other detection models.
"""
import torch
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
from .config import DetectorConfig


class ObjectDetector:
    """
    Wrapper for YOLO object detector.
    Can be easily replaced with other detection models.
    """
    
    # Class mapping from YOLO model
    CLASS_NAMES = {
        0: "Player",
        1: "Goalkeeper", 
        2: "Ball",
        3: "Referee",
        4: "Staff"
    }
    
    # Classes we care about
    TRACK_CLASSES = ["Player", "Goalkeeper"]
    
    def __init__(self, config: DetectorConfig):
        """
        Initialize detector with configuration.
        
        Args:
            config: DetectorConfig object with model settings
        """
        self.config = config
        print(f"Loading detector on device: {config.device}")
        
        # Load YOLO model
        self.model = YOLO(config.model_path)
        if config.device != "cpu":
            self.model.to(config.device)
        
        self.device = config.device
        
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame in BGR format (H, W, 3)
            
        Returns:
            Tuple of (boxes, confidences, class_labels)
            - boxes: (N, 4) array of [x1, y1, x2, y2]
            - confidences: (N,) array of confidence scores
            - class_labels: (N,) array of class indices
        """
        with torch.no_grad():
            # Use low confidence threshold for robust tracking
            results = self.model(
                frame, 
                verbose=False,
                conf=self.config.low_confidence_threshold
            )
        
        # Extract detections
        boxes = results[0].boxes
        coords = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_labels = boxes.cls.cpu().numpy().astype(int)
        
        return coords, confidences, class_labels
    
    def filter_detections(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_labels: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Dict[int, Dict]]:
        """
        Filter and organize detections by class.
        
        Args:
            boxes: (N, 4) array of bounding boxes
            confidences: (N,) array of confidence scores
            class_labels: (N,) array of class indices
            frame_shape: (height, width) of the frame
            
        Returns:
            Dictionary organized by class name and detection ID
        """
        height, width = frame_shape
        result = {
            "Player": {},
            "Goalkeeper": {},
            "Ball": {}
        }
        
        # Process each detection
        ball_idx = 0
        for i in range(len(boxes)):
            class_idx = int(class_labels[i])
            class_name = self.CLASS_NAMES.get(class_idx, None)
            
            # Skip classes we don't care about
            if class_name not in result:
                continue
            
            # Skip low confidence detections
            conf = float(confidences[i])
            if conf < self.config.confidence_threshold:
                continue
            
            # Get bounding box
            x1, y1, x2, y2 = boxes[i]
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            
            # Calculate bottom center point (for ground contact)
            bottom_center = [int((x1 + x2) / 2), y2]
            
            detection = {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "bottom_center": bottom_center
            }
            
            # Use temporary ID for ball (will be handled separately)
            if class_name == "Ball":
                result[class_name][ball_idx] = detection
                ball_idx += 1
            else:
                # For players/goalkeepers, use array index as temp ID
                # (will be replaced by tracker ID)
                result[class_name][i] = detection
        
        return result
    
    def get_detection_array(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_labels: np.ndarray
    ) -> np.ndarray:
        """
        Format detections for tracker input.
        
        Args:
            boxes: (N, 4) array of bounding boxes
            confidences: (N,) array of confidence scores  
            class_labels: (N,) array of class indices
            
        Returns:
            (N, 6) array of [x1, y1, x2, y2, conf, class]
        """
        return np.hstack((
            boxes,
            confidences.reshape(-1, 1),
            class_labels.reshape(-1, 1)
        ))
    
    def __repr__(self):
        return f"ObjectDetector(model={self.config.model_path}, device={self.device})"

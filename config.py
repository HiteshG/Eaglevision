"""
Configuration management for Football Tracker.
Centralized configuration makes it easy to modify parameters.
"""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class DetectorConfig:
    """Configuration for object detector."""
    model_path: str = "yolov8n.pt"  # Can be changed to custom weights
    confidence_threshold: float = 0.35
    low_confidence_threshold: float = 0.15  # For robust tracking
    device: Optional[str] = None  # Auto-detect if None
    
    def __post_init__(self):
        if self.device is None:
            self.device = self._get_device()
    
    @staticmethod
    def _get_device() -> str:
        """Auto-detect available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


@dataclass
class TrackerConfig:
    """Configuration for object tracker."""
    tracker_type: str = "botsort"  # Can be changed to: bytetrack, strongsort, etc.
    reid_weights: str = "osnet_x0_25_msmt17.pt"
    device: Optional[str] = None
    
    def __post_init__(self):
        if self.device is None:
            # BoTSORT doesn't support MPS, use CPU fallback
            device = DetectorConfig._get_device()
            self.device = "cpu" if device == "mps" else device


@dataclass
class TeamAssignerConfig:
    """Configuration for team assignment."""
    n_clusters: int = 2  # Number of teams
    overlap_threshold: float = 0.35  # Ignore detections with high overlap
    color_ranges: dict = None  # HSV color ranges for detection
    
    def __post_init__(self):
        if self.color_ranges is None:
            self.color_ranges = {
                "red": [(0, 100, 100), (10, 255, 255)],
                "red2": [(160, 100, 100), (179, 255, 255)],
                "orange": [(11, 100, 100), (25, 255, 255)],
                "yellow": [(26, 100, 100), (35, 255, 255)],
                "green": [(36, 100, 100), (85, 255, 255)],
                "cyan": [(86, 100, 100), (95, 255, 255)],
                "blue": [(96, 100, 100), (125, 255, 255)],
                "purple": [(126, 100, 100), (145, 255, 255)],
                "magenta": [(146, 100, 100), (159, 255, 255)],
                "white": [(0, 0, 200), (180, 30, 255)],
                "gray": [(0, 0, 50), (180, 30, 200)],
                "black": [(0, 0, 0), (180, 255, 50)],
            }


@dataclass
class ProcessorConfig:
    """Configuration for data processing."""
    interpolate: bool = True
    smooth: bool = False
    temporal_threshold_seconds: float = 1.1  # For ID merging
    spatial_threshold_per_frame: float = 10.0  # Distance threshold


@dataclass
class VisualizerConfig:
    """Configuration for visualization."""
    show_ids: bool = True
    show_bboxes: bool = True
    show_ball: bool = True
    team_colors: dict = None
    goalkeeper_color: tuple = (0, 255, 0)  # Green in BGR
    ball_color: tuple = (0, 255, 0)  # Green in BGR
    
    def __post_init__(self):
        if self.team_colors is None:
            self.team_colors = {
                0: (0, 0, 255),    # Red in BGR
                1: (255, 0, 0),    # Blue in BGR
            }


@dataclass
class MainConfig:
    """Main configuration combining all components."""
    detector: DetectorConfig = None
    tracker: TrackerConfig = None
    team_assigner: TeamAssignerConfig = None
    processor: ProcessorConfig = None
    visualizer: VisualizerConfig = None
    
    # Video processing
    fps: int = 24
    output_dir: str = "output"
    
    def __post_init__(self):
        if self.detector is None:
            self.detector = DetectorConfig()
        if self.tracker is None:
            self.tracker = TrackerConfig()
        if self.team_assigner is None:
            self.team_assigner = TeamAssignerConfig()
        if self.processor is None:
            self.processor = ProcessorConfig()
        if self.visualizer is None:
            self.visualizer = VisualizerConfig()

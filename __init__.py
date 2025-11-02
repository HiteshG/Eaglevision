"""
Football Tracker - Modular Detection, Tracking, and Team Assignment System

A modular system for detecting, tracking, and assigning teams to football players
in broadcast video footage.

Components:
- ObjectDetector: YOLO-based detection
- ObjectTracker: BoTSORT-based tracking
- TeamAssigner: Color-based team clustering
- DataProcessor: Data processing and interpolation
- Visualizer: Drawing and visualization
"""

__version__ = "1.0.0"
__author__ = "Football Tracker Team"

from .config import (
    MainConfig,
    DetectorConfig,
    TrackerConfig,
    TeamAssignerConfig,
    ProcessorConfig,
    VisualizerConfig
)

from .detector import ObjectDetector
from .tracker import ObjectTracker, SimpleTracker
from .team_assigner import TeamAssigner
from .processor import DataProcessor
from .visualizer import Visualizer

from .utils import (
    read_video,
    write_video,
    save_tracking_data,
    load_tracking_data,
    create_output_directory,
    print_summary
)

__all__ = [
    # Config
    "MainConfig",
    "DetectorConfig",
    "TrackerConfig",
    "TeamAssignerConfig",
    "ProcessorConfig",
    "VisualizerConfig",
    
    # Core modules
    "ObjectDetector",
    "ObjectTracker",
    "SimpleTracker",
    "TeamAssigner",
    "DataProcessor",
    "Visualizer",
    
    # Utils
    "read_video",
    "write_video",
    "save_tracking_data",
    "load_tracking_data",
    "create_output_directory",
    "print_summary",
]

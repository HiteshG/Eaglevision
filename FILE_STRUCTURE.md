# Football Player Tracker - Complete File Structure

## Overview

This document provides a complete overview of all files in the Football Player Tracker project, their purposes, and how they interact.

---

## Core Package Files

### `__init__.py`
**Purpose**: Package initialization and exports  
**Contents**: Imports all public APIs, version info  
**Usage**: Enables `from football_tracker import FootballTracker`

### `config.py` (394 lines)
**Purpose**: Configuration management using dataclasses  
**Key Classes**:
- `DetectorConfig`: YOLO detection settings
- `TrackerConfig`: BoTSORT tracking settings
- `TeamAssignerConfig`: Team clustering settings
- `ProcessorConfig`: Data processing settings
- `VisualizerConfig`: Visualization settings
- `MainConfig`: Combined configuration

**Why Dataclasses**: Type safety, default values, easy modification

### `detector.py` (166 lines)
**Purpose**: Object detection wrapper for YOLO  
**Key Classes**: 
- `ObjectDetector`: Main detector class
**Key Methods**:
- `detect(frame)`: Run YOLO detection
- `filter_detections(...)`: Organize by class
- `get_detection_array(...)`: Format for tracker

**Modularity**: Easy to replace with Faster R-CNN, SSD, etc.

### `tracker.py` (183 lines)
**Purpose**: Object tracking wrapper  
**Key Classes**:
- `ObjectTracker`: BoTSORT wrapper
- `SimpleTracker`: IoU-based fallback
**Key Methods**:
- `update(detections, frame)`: Update tracks
- `organize_tracks(...)`: Format output
- `reset()`: Reset tracker state

**Modularity**: Easy to replace with ByteTrack, DeepSORT, etc.

### `team_assigner.py` (241 lines)
**Purpose**: Color-based team clustering  
**Key Classes**:
- `TeamAssigner`: Main team assignment
- `DeepLearningTeamAssigner`: Placeholder for ML approach
**Algorithm**:
1. Segment players from background (KMeans)
2. Extract HSV color distribution
3. Find 2 most common colors → teams
4. Assign with outlier handling

**Modularity**: Easy to replace with CNN-based classifier

### `processor.py` (275 lines)
**Purpose**: Data processing and ID merging  
**Key Class**: `DataProcessor`  
**Key Methods**:
- `process(...)`: Main processing pipeline
- `_create_dataframe(...)`: Structure data
- `_interpolate_column(...)`: Fill gaps
- `_merge_fragmented_ids(...)`: Fix ID switching

**Processing Steps**:
1. Create DataFrame from detections
2. Interpolate missing positions
3. Merge fragmented IDs
4. Smooth trajectories (optional)

### `visualizer.py` (224 lines)
**Purpose**: Drawing and video creation  
**Key Class**: `Visualizer`  
**Key Methods**:
- `draw_frame(...)`: Annotate single frame
- `draw_from_dataframe(...)`: Draw from processed data
- `create_annotated_video(...)`: Generate output video

**Drawing Elements**:
- Player ellipses (color-coded by team)
- Player IDs
- Bounding boxes (optional)
- Ball markers (triangle)
- Info panel (optional)

### `utils.py` (220 lines)
**Purpose**: I/O operations and helpers  
**Key Functions**:
- `read_video(...)`: Load video with FPS sampling
- `write_video(...)`: Save annotated video
- `save_tracking_data(...)`: Export JSON files
- `load_tracking_data(...)`: Load results
- `print_summary(...)`: Display statistics

### `main.py` (245 lines)
**Purpose**: Main orchestration script  
**Key Classes**:
- `FootballTracker`: Pipeline coordinator
**Pipeline Steps**:
1. Read video
2. Detect and track objects
3. Assign teams
4. Process data
5. Create visualizations
6. Save results

**Usage**: 
- CLI: `python -m football_tracker.main --video input.mp4`
- API: `tracker.process_video("input.mp4")`

---

## Documentation Files

### `README.md` (450 lines)
**Purpose**: Main project documentation  
**Sections**:
- Features and overview
- Installation instructions
- Usage examples
- Configuration guide
- Modular design explanation
- Command-line arguments
- Google Colab usage
- Performance tips
- Troubleshooting

### `TECHNICAL_REPORT.md` (730 lines)
**Purpose**: Detailed technical documentation  
**Sections**:
1. System Overview
2. Included Components (detailed)
3. Excluded Components (from Eagle)
4. System Architecture
5. Performance Characteristics
6. Output Format
7. Limitations and Future Work
8. Comparison with Eagle
9. Conclusion
10. References

**Key Insights**:
- What was kept from Eagle
- What was removed and why
- Performance metrics
- Design decisions

### `QUICKSTART.md` (180 lines)
**Purpose**: Quick start guide  
**Sections**:
- Installation steps
- Basic usage examples
- Google Colab quick script
- Configuration options
- Troubleshooting
- Performance tips

### `examples.py` (500 lines)
**Purpose**: Example usage scripts  
**Examples**:
1. Basic usage
2. Custom configuration
3. Load and visualize results
4. Batch processing
5. Custom detector
6. Extract statistics
7. Filter players
8. Create highlights
9. Team comparison
10. Export to CSV

---

## Configuration Files

### `requirements.txt`
**Purpose**: Python dependencies  
**Key Packages**:
- torch, torchvision: Deep learning
- ultralytics: YOLO detection
- boxmot: Object tracking
- opencv-python: Computer vision
- pandas: Data processing
- scikit-learn: Clustering

### `setup.py`
**Purpose**: Package installation script  
**Usage**: `pip install -e .`  
**Enables**: 
- `football-tracker` command
- Package distribution

### `.gitignore`
**Purpose**: Git ignore patterns  
**Ignores**:
- Python cache files
- Virtual environments
- IDE files
- Model weights
- Output videos
- OS files

### `LICENSE`
**Purpose**: MIT License  
**Permissions**: Free to use, modify, distribute

---

## Additional Files

### `colab_notebook.py`
**Purpose**: Google Colab notebook cells  
**Usage**: Copy-paste into Colab  
**Features**:
- Installation cell
- Upload cell
- Processing cell
- Preview cell
- Download cell
- Statistics cell

---

## File Dependencies

```
main.py
├── config.py (all configs)
├── detector.py → ultralytics
├── tracker.py → boxmot
├── team_assigner.py → sklearn, cv2
├── processor.py → pandas, numpy
├── visualizer.py → cv2
└── utils.py → cv2, json

Each module imports:
- config.py: Standard library only
- detector.py: torch, ultralytics, numpy
- tracker.py: boxmot, numpy
- team_assigner.py: cv2, sklearn, numpy
- processor.py: pandas, numpy
- visualizer.py: cv2, numpy, pandas
- utils.py: cv2, json, pandas, numpy
```

---

## Data Flow

```
Input: video.mp4
    ↓
main.py (FootballTracker)
    ├── utils.read_video() → frames[]
    ├── detector.detect() → boxes, confs, classes
    ├── tracker.update() → tracked_objects
    ├── team_assigner.assign_teams() → team_mapping
    ├── processor.process() → DataFrame
    └── visualizer.create_annotated_video() → annotated.mp4

Output Files:
├── annotated.mp4 (visualizer)
├── raw_data.json (utils)
├── processed_data.json (utils)
└── metadata.json (utils)
```

---

## Modularity Map

Each component can be independently replaced:

```python
# Swap detector
tracker.detector = CustomYOLOv9()
tracker.detector = FasterRCNN()

# Swap tracker
tracker.tracker = ByteTrack()
tracker.tracker = DeepSORT()

# Swap team assigner
tracker.team_assigner = CNNTeamClassifier()
tracker.team_assigner = JerseyNumberOCR()
```

---

## Output Format

### Directory Structure
```
output/
└── video_name/
    ├── annotated.mp4         # Visualized video
    ├── raw_data.json         # DataFrame format
    ├── processed_data.json   # Frame-by-frame format
    └── metadata.json         # Video + team info
```

### File Sizes (typical for 10-second 720p clip)
- `annotated.mp4`: 5-10 MB
- `raw_data.json`: 100-500 KB
- `processed_data.json`: 200-800 KB
- `metadata.json`: 1-5 KB

---

## Code Statistics

### Lines of Code
```
Core package:       ~1,600 lines
Documentation:      ~1,400 lines
Examples:           ~500 lines
Total:              ~3,500 lines
```

### File Count
```
Python files:       10 (.py)
Documentation:      4 (.md)
Configuration:      4 (txt, py, gitignore, license)
Total:              18 files
```

---

## Testing Checklist

When modifying the code, test:

1. ✅ Detector works on sample frame
2. ✅ Tracker assigns IDs correctly
3. ✅ Team assignment produces 2 teams
4. ✅ Processor creates DataFrame
5. ✅ Visualizer draws annotations
6. ✅ Main pipeline completes
7. ✅ Output files are created
8. ✅ CLI arguments work
9. ✅ Python API works
10. ✅ Can load and re-visualize

---

## Common Modification Points

### Change Detection Model
**File**: `detector.py`  
**Method**: `__init__` → change `model_path`

### Change Tracking Algorithm
**File**: `tracker.py`  
**Class**: Replace `BotSort` with alternative

### Change Team Colors
**File**: `config.py`  
**Class**: `VisualizerConfig.team_colors`

### Change Processing FPS
**File**: `config.py`  
**Class**: `MainConfig.fps`

### Add New Visualization
**File**: `visualizer.py`  
**Method**: Add new `draw_*` method

---

## Version Control

**Current Version**: 1.0.0  
**Python Compatibility**: 3.8+  
**PyTorch**: 2.0+  
**Last Updated**: 2024

---

## Support

For questions about specific files:
- Core modules → Technical Report
- Usage → README or Quick Start
- Advanced features → examples.py
- Installation → requirements.txt + README

---

**This document is auto-generated and maintained as the project evolves.**
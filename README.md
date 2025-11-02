# Football Player Tracker

A modular system for detecting, tracking, and assigning teams to football players in broadcast video footage.

## Features

✅ **Object Detection** - YOLO-based detection of players, goalkeepers, and ball  
✅ **Multi-Object Tracking** - BoTSORT tracking with ID persistence  
✅ **Team Assignment** - Color-based clustering for team identification  
✅ **Data Processing** - Interpolation and smoothing of trajectories  
✅ **Visualization** - Annotated videos with player IDs and team colors  
✅ **Modular Design** - Easy to swap components (detector, tracker, team assigner)

## What's Included vs Eagle

### ✅ Included Components

1. **Object Detection** (YOLO)
   - Player detection
   - Goalkeeper detection
   - Ball detection
   - Confidence thresholding
   - Bounding box extraction

2. **Object Tracking** (BoTSORT)
   - ID assignment and persistence
   - Re-identification across frames
   - Trajectory tracking
   - ID fragmentation merging

3. **Team Assignment** (Color-based)
   - HSV color space analysis
   - KMeans clustering for player segmentation
   - Dominant color extraction
   - Two-team classification
   - Outlier handling

4. **Data Processing**
   - Missing value interpolation
   - Trajectory smoothing
   - ID merging for fragmented tracks
   - Team consistency checking

5. **Visualization**
   - Bounding boxes
   - Player IDs
   - Team colors
   - Ball markers
   - Annotated video output

### ❌ Excluded Components (from Eagle)

1. **Camera Calibration**
   - Homography matrix computation
   - Pitch keypoint detection (HRNet)
   - Camera position estimation
   - Optical flow keypoint tracking
   - Geometry-based keypoint synthesis

2. **Coordinate Transformation**
   - Image → Pitch coordinate mapping
   - UEFA pitch coordinate system (105×68)
   - Visible pitch boundary calculation
   - Out-of-bounds detection

3. **Advanced Visualizations**
   - Minimap generation
   - Voronoi diagrams
   - Pass trajectory plots
   - Player trajectory heatmaps

## Installation

### Local Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd football_tracker

# Install dependencies
pip install -r requirements.txt

# Download YOLO weights (if not using default)
# YOLOv8n will be auto-downloaded by ultralytics
```

### Google Colab Installation

```python
# In a Colab notebook cell:
!git clone <your-repo-url>
%cd football_tracker
!pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
python -m football_tracker.main --video path/to/video.mp4
```

### With Custom Options

```bash
python -m football_tracker.main \
    --video input.mp4 \
    --output-dir results \
    --fps 24 \
    --detector-conf 0.4 \
    --show-bboxes
```

### Python API Usage

```python
from football_tracker import FootballTracker, MainConfig

# Create custom configuration
config = MainConfig()
config.fps = 24
config.detector.confidence_threshold = 0.35

# Initialize tracker
tracker = FootballTracker(config)

# Process video
output_dir = tracker.process_video("input.mp4")
print(f"Results saved to: {output_dir}")
```

## Project Structure

```
football_tracker/
├── __init__.py           # Package initialization
├── config.py             # Configuration management
├── detector.py           # YOLO object detection
├── tracker.py            # BoTSORT tracking
├── team_assigner.py      # Team clustering
├── processor.py          # Data processing
├── visualizer.py         # Visualization
├── utils.py              # I/O utilities
├── main.py               # Main script
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Configuration

### Detector Configuration

```python
from football_tracker import DetectorConfig

detector_config = DetectorConfig(
    model_path="yolov8n.pt",           # YOLO model
    confidence_threshold=0.35,          # Minimum confidence
    low_confidence_threshold=0.15,      # For tracking
    device="cuda"                       # cuda/cpu/mps
)
```

### Tracker Configuration

```python
from football_tracker import TrackerConfig

tracker_config = TrackerConfig(
    tracker_type="botsort",             # Tracker algorithm
    reid_weights="osnet_x0_25_msmt17.pt",
    device="cuda"
)
```

### Team Assigner Configuration

```python
from football_tracker import TeamAssignerConfig

team_config = TeamAssignerConfig(
    n_clusters=2,                       # Number of teams
    overlap_threshold=0.35              # Ignore overlaps
)
```

## Modular Design

### Swapping the Detector

```python
class CustomDetector:
    """Your custom detector implementation."""
    
    def detect(self, frame):
        # Your detection logic
        return boxes, confidences, class_labels
    
    def get_detection_array(self, boxes, confidences, class_labels):
        # Format for tracker
        return detection_array

# Use custom detector
tracker = FootballTracker()
tracker.detector = CustomDetector()
```

### Swapping the Tracker

```python
from football_tracker import SimpleTracker

# Use simple IoU tracker instead of BoTSORT
tracker = FootballTracker()
tracker.tracker = SimpleTracker()
```

### Swapping Team Assignment

```python
class MLTeamAssigner:
    """Deep learning-based team assignment."""
    
    def assign_teams(self, frames, detections_per_frame):
        # Your ML model logic
        return team_mapping

# Use ML-based team assignment
tracker = FootballTracker()
tracker.team_assigner = MLTeamAssigner()
```

## Output Files

After processing, you'll get:

1. **annotated.mp4** - Video with tracking visualization
2. **raw_data.json** - Raw tracking data (frame-by-frame)
3. **processed_data.json** - Processed and interpolated data
4. **metadata.json** - Video metadata and team assignments

### Output Format

```json
{
  "frame": 0,
  "time": "00:00",
  "detections": [
    {
      "id": 5,
      "type": "Player",
      "team": 0,
      "x": 450.5,
      "y": 620.3
    },
    {
      "id": "Ball",
      "type": "Ball",
      "x": 300.2,
      "y": 400.1
    }
  ]
}
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video` | str | Required | Path to input video |
| `--output-dir` | str | Auto | Output directory |
| `--fps` | int | 24 | Target FPS |
| `--detector-conf` | float | 0.35 | Confidence threshold |
| `--model` | str | yolov8n.pt | YOLO model path |
| `--no-gpu` | flag | False | Force CPU usage |
| `--show-bboxes` | flag | False | Show bounding boxes |

## Google Colab Example

```python
# Install and import
!git clone <your-repo-url>
%cd football_tracker
!pip install -q -r requirements.txt

from football_tracker import FootballTracker, MainConfig

# Upload video
from google.colab import files
uploaded = files.upload()
video_path = list(uploaded.keys())[0]

# Process
config = MainConfig()
config.fps = 24
tracker = FootballTracker(config)
output_dir = tracker.process_video(video_path)

# Download results
!zip -r results.zip {output_dir}
files.download(f"{output_dir}/results.zip")
```

## Performance Tips

1. **Lower FPS** - Process at 12-15 FPS for faster results
2. **Smaller Model** - Use YOLOv8n instead of YOLOv8x
3. **GPU** - Use CUDA for 10-20x speedup
4. **Confidence** - Increase threshold to reduce false positives

## Troubleshooting

### Issue: Low detection accuracy
**Solution:** Increase `detector_conf` or use a larger YOLO model

### Issue: ID fragmentation
**Solution:** Adjust `temporal_threshold_seconds` and `spatial_threshold_per_frame` in ProcessorConfig

### Issue: Wrong team assignments
**Solution:** Team colors might be similar. Consider implementing deep learning-based assignment

### Issue: GPU out of memory
**Solution:** Process at lower FPS or use a smaller model

## Future Enhancements

Potential additions to make the system even better:

- [ ] Deep learning-based team assignment (ResNet/EfficientNet)
- [ ] Player re-identification across occlusions
- [ ] Jersey number recognition (OCR)
- [ ] Action recognition (pass, shoot, tackle)
- [ ] Tactical analysis (formation detection)
- [ ] Real-time processing support
- [ ] Multi-camera fusion

## Citation

If you use this code in your research, please cite:

```bibtex
@software{football_tracker_2024,
  title={Football Player Tracker: Modular Detection and Tracking System},
  author={Your Name},
  year={2024},
  url={your-repo-url}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- YOLO by Ultralytics
- BoTSORT by NirAharon
- Inspired by Eagle project for football analysis

## Contact

For questions, issues, or contributions:
- GitHub Issues: [your-repo-url/issues]
- Email: your-email@example.com

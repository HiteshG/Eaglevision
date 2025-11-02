# Football Player Tracker - Quick Start Guide

## Installation

### Step 1: Clone or Download

```bash
# If using git
git clone <your-repo-url>
cd football_tracker

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- Ultralytics YOLO (object detection)
- BoxMOT (object tracking)
- OpenCV (computer vision)
- Pandas, NumPy, scikit-learn (data processing)

## Basic Usage

### Command Line

Process a video with default settings:

```bash
python -m football_tracker.main --video input.mp4
```

With custom options:

```bash
python -m football_tracker.main \
    --video input.mp4 \
    --fps 24 \
    --detector-conf 0.4 \
    --output-dir my_results \
    --show-bboxes
```

### Python API

```python
from football_tracker import FootballTracker

# Initialize tracker
tracker = FootballTracker()

# Process video
output_dir = tracker.process_video("input.mp4")

print(f"Results saved to: {output_dir}")
```

## Google Colab Usage

1. Upload the `colab_notebook.py` to Google Colab
2. Run cells in order:
   - Installation
   - Upload video
   - Process video
   - Download results

Or copy this quick script:

```python
# Install
!git clone <your-repo-url>
%cd football_tracker
!pip install -q -r requirements.txt

# Import
from football_tracker import FootballTracker
from google.colab import files

# Upload video
uploaded = files.upload()
video_path = list(uploaded.keys())[0]

# Process
tracker = FootballTracker()
output_dir = tracker.process_video(video_path)

# Download
!zip -r results.zip {output_dir}
files.download("results.zip")
```

## Output Files

After processing, you'll get:

```
output/video_name/
├── annotated.mp4         # Video with tracking overlay
├── raw_data.json         # Raw tracking data
├── processed_data.json   # Formatted tracking data
└── metadata.json         # Video metadata and teams
```

## Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | Required | Input video path |
| `--fps` | 24 | Processing FPS |
| `--detector-conf` | 0.35 | Detection confidence |
| `--output-dir` | Auto | Output directory |
| `--model` | yolov8n.pt | YOLO model |
| `--no-gpu` | False | Force CPU |
| `--show-bboxes` | False | Show boxes |

### Python Configuration

```python
from football_tracker import MainConfig

config = MainConfig()
config.fps = 24
config.detector.confidence_threshold = 0.35
config.detector.model_path = "yolov8n.pt"
config.visualizer.show_bboxes = True
config.visualizer.show_ids = True

tracker = FootballTracker(config)
```

## Troubleshooting

### Issue: ImportError
**Solution**: Run `pip install -r requirements.txt`

### Issue: CUDA out of memory
**Solution**: 
- Use smaller model: `--model yolov8n.pt`
- Lower FPS: `--fps 12`
- Use CPU: `--no-gpu`

### Issue: Poor detection accuracy
**Solution**:
- Increase confidence: `--detector-conf 0.5`
- Use larger model: `--model yolov8m.pt` or `yolov8l.pt`

### Issue: Wrong team colors
**Solution**: Team assignment is color-based. If jerseys are similar, results may be incorrect. Consider implementing deep learning-based assignment.

### Issue: ID switching
**Solution**: Adjust temporal and spatial thresholds in ProcessorConfig

## Next Steps

1. Read `README.md` for detailed documentation
2. Check `TECHNICAL_REPORT.md` for architecture details
3. See `examples.py` for advanced usage examples
4. Modify configurations for your specific use case

## Support

- GitHub Issues: [your-repo-url/issues]
- Documentation: `README.md` and `TECHNICAL_REPORT.md`
- Examples: `examples.py`

## Performance Tips

1. **Speed**: Use YOLOv8n and lower FPS (12-15)
2. **Accuracy**: Use YOLOv8l and higher confidence (0.4-0.5)
3. **Balance**: Use YOLOv8m at 24 FPS with 0.35 confidence
4. **GPU**: Always use GPU if available (10-20x faster)

## Minimum Requirements

- Python 3.8+
- 4GB RAM (8GB recommended)
- GPU with 2GB VRAM (optional but recommended)
- OpenCV-compatible video formats (mp4, avi, mov)

## Recommended Setup

- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.8+
- Video resolution: 720p-1080p
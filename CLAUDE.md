# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Football Player Tracker** is a modular computer vision system for detecting, tracking, and analyzing football players in broadcast video footage. The system uses YOLO for detection, BoTSORT for tracking, and color-based clustering for team assignment.

## Common Commands

### Running the System

```bash
# Basic usage - process a video with default settings
python main.py --video path/to/video.mp4

# Custom configuration
python main.py \
    --video input.mp4 \
    --fps 24 \
    --detector-conf 0.35 \
    --model yolov8n.pt \
    --output-dir results \
    --show-bboxes

# Force CPU usage (no GPU)
python main.py --video input.mp4 --no-gpu
```

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Check for import errors
python -c "import main; import detector; import tracker"

# Test on a short clip (process at lower FPS for speed)
python main.py --video short_clip.mp4 --fps 12
```

## System Architecture

### High-Level Pipeline

The system follows a sequential 5-step pipeline:

```
1. Video Reading → 2. Detection & Tracking → 3. Team Assignment → 4. Data Processing → 5. Visualization
```

**Data Flow:**
```
Input: video.mp4
  ↓
utils.read_video() → frames[] (sampled at target FPS)
  ↓
FOR EACH FRAME:
  detector.detect() → boxes, confidences, class_labels
  ↓
  tracker.update() → tracked_objects with persistent IDs
  ↓
  Accumulate: detections_per_frame[]
  ↓
team_assigner.assign_teams() → team_mapping {player_id → team_id}
  ↓
processor.process() → DataFrame with interpolated trajectories
  ↓
visualizer.create_annotated_video() → annotated.mp4
  ↓
Output: annotated.mp4, raw_data.json, processed_data.json, metadata.json
```

### Module Responsibilities

1. **config.py** - Configuration management using dataclasses
   - Auto-device detection (CUDA/MPS/CPU)
   - Centralized settings for all modules
   - No AI/ML models, just configuration

2. **detector.py** - Object detection wrapper (YOLO)
   - Detects: Player, Goalkeeper, Ball, Referee, Staff
   - Uses dual confidence thresholds (low for tracking, high for display)
   - Returns: bounding boxes, confidences, class labels

3. **tracker.py** - Multi-object tracking wrapper (BoTSORT)
   - Assigns persistent IDs across frames
   - Uses ReID features for robust matching
   - Handles occlusions and re-identification
   - Fallback: SimpleTracker (IoU-based)

4. **team_assigner.py** - Color-based team clustering
   - Two-pass algorithm: color collection → team assignment
   - Uses KMeans for player segmentation from background
   - HSV color space for jersey color extraction
   - Handles outliers and ambiguous cases

5. **processor.py** - Data processing and ID merging
   - Creates DataFrame from raw detections
   - Interpolates missing positions (linear)
   - Merges fragmented IDs (same player, different IDs)
   - Optional trajectory smoothing

6. **visualizer.py** - Drawing and video generation
   - Draws ellipses at player positions
   - Color-codes by team
   - Shows player IDs and ball markers
   - Creates annotated output video

7. **utils.py** - I/O utilities
   - Video reading with FPS downsampling
   - JSON serialization of tracking data
   - Summary statistics

8. **main.py** - Pipeline orchestrator
   - Coordinates all modules
   - CLI argument parsing
   - Progress reporting

## Model Architecture

### Detection Model: YOLOv8

**Type:** Single-stage object detector (anchor-free)

**Model Variants:**
- yolov8n.pt (nano) - ~3.2M parameters, fastest
- yolov8m.pt (medium) - ~25.9M parameters, balanced
- yolov8l.pt (large) - ~43.7M parameters, most accurate

**Classes Detected:**
```python
{
  0: "Player",
  1: "Goalkeeper",
  2: "Ball",
  3: "Referee",
  4: "Staff"
}
```

**Detection Process:**
1. Input: BGR frame (H, W, 3)
2. Model runs inference with low_confidence_threshold (0.15)
3. Output: bounding boxes [x1, y1, x2, y2], confidences, class_labels
4. Bottom-center point calculated: (center_x, bottom_y) for ground contact
5. Format for tracker: [x1, y1, x2, y2, conf, class_idx]

**Location:** `detector.py:41-68`

### Tracking Model: BoTSORT

**Type:** Multi-object tracker with appearance features

**Components:**
- Motion model: Kalman filter for state prediction
- Appearance model: OSNet ReID network (osnet_x0_25_msmt17.pt)
- Association: IoU + appearance similarity matching
- Track management: Birth, life, death states

**Tracking Process:**
1. Input: Detection array [x1, y1, x2, y2, conf, class_idx] + frame
2. Extract ReID features from detection crops
3. Predict track positions (Kalman filter)
4. Match detections to tracks (Hungarian algorithm)
5. Update tracks with matched detections
6. Handle unmatched: new tracks (births) or lost tracks (deaths)
7. Output: [x1, y1, x2, y2, track_id, conf, class_idx, detection_idx]

**Location:** `tracker.py:44-65`

### Team Assignment: Color-Based Clustering

**Type:** Rule-based color segmentation (no neural network)

**Algorithm Steps:**
1. **First Pass - Color Collection:**
   - For each player detection across all frames
   - Segment player from background (KMeans on RGB, n_clusters=2)
   - Extract dominant colors (HSV color ranges)
   - Weight by (1 - overlap_ratio) to reduce noise
   - Accumulate color counts per player

2. **Color Determination:**
   - Find dominant color per player (most frequent)
   - Identify 2 most common colors → team colors
   - Create color-to-team mapping {color → 0 or 1}

3. **Second Pass - Team Assignment:**
   - Assign players to teams based on dominant color
   - Handle outliers using secondary color preferences
   - Default to team 0 if no match found

**HSV Color Ranges:**
```python
{
  "red": [(0,100,100), (10,255,255)] + [(160,100,100), (179,255,255)],
  "blue": [(96,100,100), (125,255,255)],
  "green": [(36,100,100), (85,255,255)],
  "yellow": [(26,100,100), (35,255,255)],
  "white": [(0,0,200), (180,30,255)],
  "black": [(0,0,0), (180,255,50)]
}
```

**Location:** `team_assigner.py:30-137`

## Processing Pipeline - Detailed Breakdown

### Step 1: Video Reading (utils.py:13-58)

**Purpose:** Load video and downsample to target FPS

```python
frames, fps = read_video(video_path, target_fps=24)
```

**Process:**
1. Open video with OpenCV VideoCapture
2. Calculate skip rate: `skip = max(1, int(native_fps / target_fps))`
3. Read frames, keeping every Nth frame
4. Return frames list and actual FPS

**Output:** List of BGR frames, actual FPS

---

### Step 2: Detection & Tracking (main.py:119-172)

**Purpose:** Detect objects and assign persistent IDs

**For each frame:**

```python
# 2a. Detection
boxes, confidences, class_labels = detector.detect(frame)
```
- YOLO inference with low_confidence_threshold
- Returns arrays: boxes (N,4), confidences (N,), class_labels (N,)

```python
# 2b. Format for tracker
detection_array = detector.get_detection_array(boxes, confidences, class_labels)
```
- Combines into (N, 6) array: [x1, y1, x2, y2, conf, class]

```python
# 2c. Update tracker
tracks = tracker.update(detection_array, frame)
```
- BoTSORT matches detections to existing tracks
- Assigns new IDs or maintains existing IDs
- Returns: [x1, y1, x2, y2, track_id, conf, class_idx, detection_idx]

```python
# 2d. Organize by class
frame_detections = tracker.organize_tracks(tracks, CLASS_NAMES, conf_threshold, frame_shape)
```
- Groups by: Player, Goalkeeper
- Filters by confidence threshold
- Calculates bottom_center points

```python
# 2e. Add ball (not tracked, just detected)
ball_detections = detector.filter_detections(boxes, confidences, class_labels, frame_shape)["Ball"]
frame_detections["Ball"] = ball_detections
```

**Output:** `detections_per_frame` - List of detection dicts per frame

**Necessary Components:**
- YOLO model weights (yolov8n.pt)
- OSNet ReID weights (osnet_x0_25_msmt17.pt)
- GPU recommended for speed (CPU works but slower)

---

### Step 3: Team Assignment (team_assigner.py:30-137)

**Purpose:** Cluster players into 2 teams based on jersey colors

```python
team_mapping = team_assigner.assign_teams(frames, detections_per_frame)
```

**Detailed Process:**

**3a. First Pass - Accumulate Colors:**
```python
for frame, detections in zip(frames, detections_per_frame):
    for player_id, detection in detections["Player"].items():
        # Extract player crop
        bbox = detection["bbox"]
        crop = frame[y1:y2, x1:x2]

        # Calculate overlap with other players
        overlap_ratio = _calculate_max_overlap_ratio(bbox, all_bboxes)

        # Skip heavily overlapped (ambiguous)
        if overlap_ratio > 0.35:
            continue

        # Detect colors in crop
        color_counts = _detect_colors(crop)  # Returns [(color, pixel_count), ...]

        # Accumulate weighted by (1 - overlap_ratio)
        for color, count in color_counts:
            player_color_counts[player_id][color] += count * (1 - overlap_ratio)
```

**3b. Determine Team Colors:**
```python
# Find dominant color per player
player_dominant_colors = {
    player_id: max(color_counts, key=color_counts.get)
    for player_id, color_counts in player_color_counts.items()
}

# Find 2 most common colors = team colors
color_frequency = Counter(player_dominant_colors.values())
team_colors = [color for color, _ in color_frequency.most_common(2)]
color_to_team = {color: i for i, color in enumerate(team_colors)}
```

**3c. Assign Teams:**
```python
for player_id, dominant_color in player_dominant_colors.items():
    if dominant_color in color_to_team:
        team_mapping[player_id] = color_to_team[dominant_color]
    else:
        # Outlier: use secondary color preference
        team_color_counts = [(c, cnt) for c, cnt in player_color_counts[player_id].items() if c in color_to_team]
        best_color = max(team_color_counts, key=lambda x: x[1])[0]
        team_mapping[player_id] = color_to_team[best_color]
```

**Output:** `team_mapping` - Dict {player_id → team_id (0 or 1)}

**Necessary Components:**
- scikit-learn for KMeans clustering
- OpenCV for HSV color conversion

**Limitations:**
- Fails when jersey colors are very similar
- Affected by lighting conditions
- No jersey number recognition

---

### Step 4: Data Processing (processor.py:28-70)

**Purpose:** Create clean DataFrame with interpolated trajectories

```python
df, team_mapping = processor.process(detections_per_frame, team_mapping)
```

**4a. Create DataFrame:**
```python
# Convert detections to DataFrame format
data = {}
for frame_idx, detections in enumerate(detections_per_frame):
    frame_data = {}
    for class_name in ["Player", "Goalkeeper"]:
        for obj_id, detection in detections[class_name].items():
            col_name = f"{class_name}_{obj_id}"
            frame_data[col_name] = tuple(detection["bottom_center"])

    # Only include frames with at least one person
    if has_person:
        data[frame_idx] = frame_data

df = pd.DataFrame(data).T
```
- Columns: Player_0, Player_1, ..., Goalkeeper_0, Ball
- Each cell: (x, y) tuple or NaN
- Index: frame number

**4b. Interpolate Ball:**
```python
df = _interpolate_column(df, "Ball", fill=True)
```
- Linear interpolation for missing positions
- `fill=True` → fill all gaps (bfill + ffill)
- Ball typically has gaps due to occlusions

**4c. Merge Fragmented IDs:**
```python
df, team_mapping = _merge_fragmented_ids(df, team_mapping)
```
- Identifies same player tracked with different IDs
- **Merge Conditions:**
  1. Temporal proximity: gap < fps * 1.1 seconds
  2. Spatial continuity: distance < gap * 10 pixels
  3. Team consistency: same team assignment
- Updates DataFrame columns and team_mapping

**4d. Interpolate Players:**
```python
for col in df.columns:
    if col == "Ball": continue
    df = _interpolate_column(df, col, fill=False)
```
- `fill=False` → only interpolate inside (preserve entry/exit)
- Maintains player appearance/disappearance timing

**4e. Optional Smoothing:**
```python
if config.smooth:
    df = _smooth_column(df, col)
```
- Set every 2nd point to NaN, then interpolate
- Reduces jitter from detection noise

**Output:** Processed DataFrame with clean trajectories

**Necessary Components:**
- pandas for DataFrame operations
- numpy for numerical calculations

---

### Step 5: Visualization (visualizer.py:197-243)

**Purpose:** Create annotated video with tracking overlay

```python
visualizer.create_annotated_video(frames, df, team_mapping, output_path, fps)
```

**Process:**
```python
for frame_idx, frame in enumerate(frames):
    if frame_idx in df.index:
        annotated_frame = draw_from_dataframe(frame, frame_idx, df, team_mapping)
    else:
        annotated_frame = frame.copy()

    annotated_frames.append(annotated_frame)

# Write video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
for frame in annotated_frames:
    out.write(frame)
```

**Drawing Elements:**

1. **Player Ellipse:**
   ```python
   cv2.ellipse(frame, (x, y), (35, 18), 0, -45, 235, color, 2)
   ```
   - Position: bottom_center (x, y)
   - Size: 35x18 pixels
   - Color: team_colors[team_id]

2. **Player ID:**
   ```python
   cv2.putText(frame, str(obj_id), (x-10, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
   ```

3. **Ball Marker:**
   ```python
   triangle_pts = [(x, y-20), (x-5, y-30), (x+5, y-30)]
   cv2.drawContours(frame, [triangle_pts], 0, ball_color, -1)
   ```
   - Green triangle pointing to ball

4. **Optional Bounding Boxes:**
   ```python
   if show_bboxes:
       cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
   ```

**Output:** annotated.mp4 (mp4v codec)

---

### Step 6: Save Results (utils.py:96-174)

**Purpose:** Export tracking data to JSON files

```python
save_tracking_data(df, team_mapping, output_dir, fps)
```

**Output Files:**

1. **metadata.json**
   ```json
   {
     "fps": 24,
     "num_frames": 360,
     "team_mapping": {"5": 0, "7": 1, "12": 0}
   }
   ```

2. **raw_data.json**
   - DataFrame in JSON format
   - Columns as keys, rows as records

3. **processed_data.json**
   ```json
   [
     {
       "frame": 0,
       "time": "00:00",
       "detections": [
         {"id": 5, "type": "Player", "team": 0, "x": 450.5, "y": 620.3},
         {"id": "Ball", "type": "Ball", "x": 300.2, "y": 400.1}
       ]
     }
   ]
   ```

**Directory Structure:**
```
output/
└── video_name/
    ├── annotated.mp4
    ├── raw_data.json
    ├── processed_data.json
    └── metadata.json
```

## How to Use the System

### Basic Usage

```python
from main import FootballTracker
from config import MainConfig

# Create tracker with default config
tracker = FootballTracker()

# Process video
output_dir = tracker.process_video("input.mp4")
```

### Custom Configuration

```python
config = MainConfig()

# Detector settings
config.detector.model_path = "yolov8m.pt"  # Use medium model
config.detector.confidence_threshold = 0.4  # Higher confidence
config.detector.device = "cuda"  # Force GPU

# Tracker settings
config.tracker.tracker_type = "botsort"

# Team assignment settings
config.team_assigner.n_clusters = 2
config.team_assigner.overlap_threshold = 0.35

# Processing settings
config.processor.interpolate = True
config.processor.smooth = False
config.processor.temporal_threshold_seconds = 1.1
config.processor.spatial_threshold_per_frame = 10.0

# Visualization settings
config.visualizer.show_ids = True
config.visualizer.show_bboxes = True
config.visualizer.show_ball = True
config.visualizer.team_colors = {
    0: (0, 0, 255),   # Red in BGR
    1: (255, 0, 0)    # Blue in BGR
}

# Video processing
config.fps = 24
config.output_dir = "my_results"

tracker = FootballTracker(config)
output_dir = tracker.process_video("input.mp4")
```

### Swapping Components (Modularity)

```python
# Swap detector
from detector import ObjectDetector
class CustomDetector:
    def detect(self, frame):
        # Your detection logic
        return boxes, confidences, class_labels

tracker = FootballTracker()
tracker.detector = CustomDetector()

# Swap tracker
from tracker import SimpleTracker
tracker.tracker = SimpleTracker()  # Use IoU-based instead of BoTSORT

# Swap team assigner
class MLTeamAssigner:
    def assign_teams(self, frames, detections_per_frame):
        # Your ML model logic
        return team_mapping

tracker.team_assigner = MLTeamAssigner()
```

### Loading and Re-visualizing

```python
from utils import load_tracking_data
from visualizer import Visualizer

# Load saved data
df, team_mapping, fps = load_tracking_data("output/video_name")

# Re-create video with different settings
config = VisualizerConfig()
config.show_bboxes = False  # Hide bounding boxes
config.team_colors = {0: (0, 255, 0), 1: (255, 0, 255)}  # New colors

visualizer = Visualizer(config)
visualizer.create_annotated_video(frames, df, team_mapping, "new_output.mp4", fps)
```

## Necessary Requirements

### Dependencies
- torch >= 2.0.0 (deep learning framework)
- ultralytics >= 8.0.0 (YOLO detection)
- boxmot >= 10.0.0 (BoTSORT tracking)
- opencv-python >= 4.8.0 (video I/O and visualization)
- pandas >= 2.0.0 (data processing)
- scikit-learn >= 1.3.0 (KMeans clustering)
- numpy >= 1.24.0 (numerical operations)

### Model Weights (auto-downloaded on first run)
- yolov8n.pt (or yolov8m.pt, yolov8l.pt)
- osnet_x0_25_msmt17.pt (ReID model for BoTSORT)

### Hardware
- Minimum: CPU, 4GB RAM (slow: 5-7 FPS)
- Recommended: GPU with 4GB+ VRAM, 8GB+ RAM (fast: 11-20 FPS)

## Performance Characteristics

### Processing Speed
- **YOLOv8n + BoTSORT (GPU):** 11-20 FPS
  - Detection: ~30-50ms/frame
  - Tracking: ~20-40ms/frame

- **YOLOv8n + SimpleTracker (CPU):** 5-7 FPS
  - Detection: ~150-200ms/frame
  - Tracking: ~1-5ms/frame

### Accuracy Notes
- Detection depends on YOLO model size (n < m < l)
- Tracking quality depends on ReID model
- Team assignment accuracy: 85-95% (color-based, fails with similar jerseys)

## Common Issues

### ID Fragmentation
**Problem:** Same player gets different IDs across frames

**Solution:** Adjust merging thresholds in ProcessorConfig:
```python
config.processor.temporal_threshold_seconds = 1.5  # Increase to merge wider gaps
config.processor.spatial_threshold_per_frame = 15.0  # Increase to allow more movement
```

### Wrong Team Assignments
**Problem:** Players assigned to wrong team

**Cause:** Jersey colors too similar or lighting variations

**Solution:** Implement deep learning-based team assignment (DeepLearningTeamAssigner is a placeholder)

### Low Detection Accuracy
**Problem:** Missing players or false detections

**Solution:**
- Increase confidence: `config.detector.confidence_threshold = 0.5`
- Use larger model: `config.detector.model_path = "yolov8l.pt"`

### GPU Out of Memory
**Problem:** CUDA out of memory error

**Solution:**
- Use smaller model: yolov8n.pt instead of yolov8l.pt
- Lower FPS: `config.fps = 12`
- Force CPU: `config.detector.device = "cpu"`

## Key Files Reference

- `main.py:46-117` - Main processing pipeline
- `detector.py:41-68` - YOLO detection
- `tracker.py:44-65` - BoTSORT tracking
- `team_assigner.py:30-137` - Team assignment algorithm
- `processor.py:28-70` - Data processing pipeline
- `visualizer.py:197-243` - Video annotation
- `config.py:10-133` - All configuration options

## What's NOT Included (vs Eagle Project)

This is a simplified version. The following are excluded:

- Camera calibration (homography matrix)
- Keypoint detection (HRNet)
- Pitch coordinate transformation
- Minimap visualization
- Voronoi diagrams
- Pass trajectory analysis
- Tactical formations

All tracking is done in image coordinates, not real-world (pitch) coordinates.

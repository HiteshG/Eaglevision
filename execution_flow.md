# Football Tracker - Detailed Execution Flow (Debug Mode)

## Scenario: User runs command
```bash
python -m football_tracker.main --video input.mp4 --fps 24
```

---

## PHASE 1: INITIALIZATION (Entry Point)

### Step 1: Python Module Execution
**File**: `football_tracker/main.py`  
**Line**: 245 (bottom of file)
```python
if __name__ == "__main__":
    sys.exit(main())
```
**What happens**: Python executes the `main()` function

---

### Step 2: Parse Command-Line Arguments
**File**: `football_tracker/main.py`  
**Lines**: 151-205

```python
def main():
    parser = argparse.ArgumentParser(...)  # Line 151
    
    # Parse arguments
    parser.add_argument("--video", type=str, required=True)  # Line 156
    parser.add_argument("--fps", type=int, default=24)       # Line 173
    # ... more arguments ...
    
    args = parser.parse_args()  # Line 205
```

**What happens**: 
- Creates argument parser
- Parses: `args.video = "input.mp4"`, `args.fps = 24`

---

### Step 3: Create Configuration
**File**: `football_tracker/main.py`  
**Lines**: 207-217

```python
# Create configuration
config = MainConfig()                                    # Line 208
config.fps = args.fps                                    # Line 209
config.detector.model_path = args.model                  # Line 210
config.detector.confidence_threshold = args.detector_conf # Line 211
```

**Triggers**: `football_tracker/config.py`  
**Lines**: 93-112

```python
@dataclass
class MainConfig:
    detector: DetectorConfig = None          # Line 95
    tracker: TrackerConfig = None            # Line 96
    # ... other fields ...
    
    def __post_init__(self):                 # Line 103
        if self.detector is None:
            self.detector = DetectorConfig() # Line 104 - Creates detector config
        if self.tracker is None:
            self.tracker = TrackerConfig()   # Line 106 - Creates tracker config
        # ... initialize others ...
```

**Triggers**: `football_tracker/config.py` - DetectorConfig initialization  
**Lines**: 11-24

```python
@dataclass
class DetectorConfig:
    model_path: str = "yolov8n.pt"          # Line 13
    confidence_threshold: float = 0.35       # Line 14
    device: Optional[str] = None             # Line 16
    
    def __post_init__(self):                 # Line 18
        if self.device is None:
            self.device = self._get_device() # Line 19 - Auto-detect GPU/CPU
    
    @staticmethod
    def _get_device() -> str:                # Line 21
        if torch.cuda.is_available():
            return "cuda"                    # Return CUDA if available
        # ... check MPS, fallback to CPU
```

**Console output**:
```
(No output yet - just configuration)
```

---

### Step 4: Initialize FootballTracker
**File**: `football_tracker/main.py`  
**Line**: 224

```python
tracker = FootballTracker(config)  # Line 224
```

**Triggers**: `football_tracker/main.py` - FootballTracker.__init__  
**Lines**: 18-41

```python
class FootballTracker:
    def __init__(self, config: MainConfig = None):       # Line 21
        self.config = config or MainConfig()             # Line 27
        
        # Print header
        print("\n" + "="*50)                             # Line 30
        print("INITIALIZING FOOTBALL TRACKER")
        print("="*50 + "\n")
        
        # Initialize components
        self.detector = ObjectDetector(self.config.detector)     # Line 35
        self.tracker = ObjectTracker(self.config.tracker)        # Line 36
        self.team_assigner = TeamAssigner(self.config.team_assigner) # Line 37
        self.visualizer = Visualizer(self.config.visualizer)     # Line 39
```

**Console output**:
```
==================================================
INITIALIZING FOOTBALL TRACKER
==================================================
```

---

### Step 5: Initialize ObjectDetector
**File**: `football_tracker/detector.py`  
**Lines**: 23-40

```python
def __init__(self, config: DetectorConfig):
    self.config = config
    print(f"Loading detector on device: {config.device}")  # Line 33
    
    # Load YOLO model
    self.model = YOLO(config.model_path)                   # Line 36 - LOAD YOLO
    if config.device != "cpu":
        self.model.to(config.device)                       # Line 38 - Move to GPU
    
    self.device = config.device
```

**Console output**:
```
Loading detector on device: cuda
```

**What happens**: 
- Ultralytics YOLO downloads weights if needed
- Loads model into GPU memory (~500MB for YOLOv8n)

---

### Step 6: Initialize ObjectTracker
**File**: `football_tracker/tracker.py`  
**Lines**: 17-41

```python
def __init__(self, config: TrackerConfig):
    self.config = config
    print(f"Loading tracker: {config.tracker_type} on device: {config.device}") # Line 26
    
    # Convert device string
    if config.device == "cuda":
        device = 0                                         # Line 30
    elif config.device == "mps":
        device = "cpu"  # BoTSORT doesn't support MPS    # Line 32
    
    # Initialize BoTSORT
    self.tracker = BotSort(                                # Line 36
        reid_weights=Path(config.reid_weights),
        device=device,
        half=False
    )
```

**Console output**:
```
Loading tracker: botsort on device: cuda
```

**What happens**: 
- BoTSORT loads ReID model (~500MB)
- Initializes tracking state

---

### Step 7: Initialize TeamAssigner
**File**: `football_tracker/team_assigner.py`  
**Lines**: 16-24

```python
def __init__(self, config: TeamAssignerConfig):
    self.config = config                                   # Line 23
    self.color_ranges = config.color_ranges                # Line 24
```

**Console output**: (none - silent initialization)

---

### Step 8: Initialize Visualizer
**File**: `football_tracker/visualizer.py`  
**Lines**: 14-22

```python
def __init__(self, config: VisualizerConfig):
    self.config = config                                   # Line 20
```

**Console output**:
```

All components initialized successfully!
```

---

## PHASE 2: VIDEO PROCESSING

### Step 9: Call process_video
**File**: `football_tracker/main.py`  
**Line**: 225

```python
output_dir = tracker.process_video(args.video, args.output_dir)  # Line 225
```

**Triggers**: `football_tracker/main.py` - FootballTracker.process_video  
**Lines**: 43-102

```python
def process_video(self, video_path: str, output_dir: str = None) -> str:
    # Create output directory
    if output_dir is None:
        output_dir = create_output_directory(              # Line 54
            video_path,
            self.config.output_dir
        )
    
    print("\n" + "="*50)                                   # Line 58
    print(f"PROCESSING VIDEO: {os.path.basename(video_path)}")
    print("="*50 + "\n")
```

**Triggers**: `football_tracker/utils.py` - create_output_directory  
**Lines**: 182-195

```python
def create_output_directory(video_path: str, base_dir: str = "output") -> str:
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Line 192
    output_dir = os.path.join(base_dir, video_name)                  # Line 193
    os.makedirs(output_dir, exist_ok=True)                           # Line 194
    return output_dir
```

**Console output**:
```

==================================================
PROCESSING VIDEO: input.mp4
==================================================
```

---

### Step 10: Read Video
**File**: `football_tracker/main.py`  
**Line**: 64

```python
print("Step 1/5: Reading video...")                        # Line 64
frames, fps = read_video(video_path, self.config.fps)     # Line 65
```

**Triggers**: `football_tracker/utils.py` - read_video  
**Lines**: 10-60

```python
def read_video(video_path: str, fps: int = 24) -> Tuple[List[np.ndarray], int]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")  # Line 21
    
    print(f"Reading video: {video_path}")                   # Line 23
    
    cap = cv2.VideoCapture(video_path)                      # Line 25 - OPEN VIDEO
    native_fps = cap.get(cv2.CAP_PROP_FPS)                  # Line 26
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # Line 27
    
    print(f"Native FPS: {native_fps:.2f}, Total frames: {total_frames}")  # Line 29
    
    # Calculate skip rate
    skip = max(1, int(native_fps / fps))                    # Line 32
    actual_fps = native_fps / skip                          # Line 33
    
    print(f"Sampling at {actual_fps:.2f} FPS (every {skip} frames)")  # Line 35
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()                             # Line 41 - READ FRAME
        if not ret:
            break
        
        if frame_count % skip == 0:
            frames.append(frame)                            # Line 46 - STORE FRAME
        
        frame_count += 1
    
    cap.release()                                           # Line 50
    
    print(f"Read {len(frames)} frames")                     # Line 52
    
    return frames, int(actual_fps)                          # Line 54
```

**Console output**:
```
Step 1/5: Reading video...
Reading video: input.mp4
Native FPS: 30.00, Total frames: 300
Sampling at 24.00 FPS (every 1 frames)
Read 240 frames
```

**What happens**: 
- Opens video with OpenCV
- Samples frames at target FPS
- Stores 240 frames in memory (~200-500MB depending on resolution)

---

### Step 11: Initialize DataProcessor
**File**: `football_tracker/main.py`  
**Lines**: 67-71

```python
if not frames:
    raise ValueError("No frames read from video")          # Line 68
    
# Initialize processor with actual FPS
self.processor = DataProcessor(self.config.processor, fps) # Line 71
```

**Triggers**: `football_tracker/processor.py` - DataProcessor.__init__  
**Lines**: 19-30

```python
def __init__(self, config: ProcessorConfig, fps: int):
    self.config = config                                   # Line 28
    self.fps = fps                                         # Line 29
```

---

## PHASE 3: DETECTION AND TRACKING

### Step 12: Start Detection Loop
**File**: `football_tracker/main.py`  
**Line**: 74

```python
print("\nStep 2/5: Detecting and tracking objects...")     # Line 74
detections_per_frame = self._detect_and_track(frames)     # Line 75
```

**Triggers**: `football_tracker/main.py` - FootballTracker._detect_and_track  
**Lines**: 104-148

```python
def _detect_and_track(self, frames: List[np.ndarray]) -> List[Dict]:
    detections_per_frame = []                              # Line 112
    
    print(f"Processing {len(frames)} frames...")           # Line 114
    
    for i, frame in enumerate(frames):                     # Line 116 - LOOP FRAMES
        if i % 100 == 0:
            print(f"  Frame {i}/{len(frames)}")            # Line 118
        
        # Detect objects
        boxes, confidences, class_labels = self.detector.detect(frame)  # Line 121
```

**Console output**:
```

Step 2/5: Detecting and tracking objects...
Processing 240 frames...
  Frame 0/240
```

---

### Step 13: Detect Objects (Frame 0)
**File**: `football_tracker/detector.py` - ObjectDetector.detect  
**Lines**: 42-64

```python
def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():                                  # Line 54
        # Use low confidence threshold for robust tracking
        results = self.model(                              # Line 56 - RUN YOLO
            frame, 
            verbose=False,
            conf=self.config.low_confidence_threshold      # 0.15
        )
    
    # Extract detections
    boxes = results[0].boxes                               # Line 62
    coords = boxes.xyxy.cpu().numpy()                     # Line 63 - [x1,y1,x2,y2]
    confidences = boxes.conf.cpu().numpy()                 # Line 64 - scores
    class_labels = boxes.cls.cpu().numpy().astype(int)    # Line 65 - classes
    
    return coords, confidences, class_labels               # Line 67
```

**What happens**: 
- YOLO forward pass (~30-50ms on GPU)
- Detects players, goalkeepers, ball
- Returns bounding boxes

**Example output** (internal):
```python
boxes = array([[120, 150, 180, 320],    # Player 1
               [250, 140, 310, 330],    # Player 2
               [400, 200, 420, 220]])   # Ball
confidences = array([0.89, 0.85, 0.65])
class_labels = array([0, 0, 2])  # 0=Player, 2=Ball
```

---

### Step 14: Format for Tracker
**File**: `football_tracker/main.py`  
**Lines**: 123-128

```python
# Prepare for tracker
detection_array = self.detector.get_detection_array(      # Line 124
    boxes,
    confidences,
    class_labels
)
```

**Triggers**: `football_tracker/detector.py` - get_detection_array  
**Lines**: 106-121

```python
def get_detection_array(
    self,
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_labels: np.ndarray
) -> np.ndarray:
    return np.hstack((                                     # Line 119
        boxes,                         # [x1, y1, x2, y2]
        confidences.reshape(-1, 1),    # [conf]
        class_labels.reshape(-1, 1)    # [class]
    ))
```

**What happens**: Creates array of shape (N, 6):
```python
detection_array = array([[120, 150, 180, 320, 0.89, 0],
                         [250, 140, 310, 330, 0.85, 0],
                         [400, 200, 420, 220, 0.65, 2]])
```

---

### Step 15: Update Tracker
**File**: `football_tracker/main.py`  
**Line**: 131

```python
tracks = self.tracker.update(detection_array, frame)      # Line 131
```

**Triggers**: `football_tracker/tracker.py` - ObjectTracker.update  
**Lines**: 45-61

```python
def update(
    self,
    detections: np.ndarray,
    frame: np.ndarray
) -> List[np.ndarray]:
    try:
        tracks = self.tracker.update(detections, frame)    # Line 60 - BoTSORT
        return tracks
    except Exception as e:
        print(f"Tracker error: {e}")
        return np.array([])
```

**What happens**: 
- BoTSORT matches detections to existing tracks
- Assigns track IDs (first frame: creates new tracks)
- Returns tracked objects with IDs (~20-40ms)

**Example output** (internal):
```python
tracks = array([[120, 150, 180, 320, 1, 0.89, 0, 0],   # ID=1
                [250, 140, 310, 330, 2, 0.85, 0, 1]])  # ID=2
# Format: [x1, y1, x2, y2, track_id, conf, class, det_idx]
```

---

### Step 16: Organize Tracks
**File**: `football_tracker/main.py`  
**Lines**: 133-139

```python
# Organize tracked objects
frame_detections = self.tracker.organize_tracks(          # Line 134
    tracks,
    self.detector.CLASS_NAMES,
    self.config.detector.confidence_threshold,
    frame.shape[:2]
)
```

**Triggers**: `football_tracker/tracker.py` - organize_tracks  
**Lines**: 63-119

```python
def organize_tracks(
    self,
    tracks: np.ndarray,
    class_names: Dict[int, str],
    confidence_threshold: float,
    frame_shape: Tuple[int, int]
) -> Dict[str, Dict[int, Dict]]:
    height, width = frame_shape                            # Line 78
    result = {
        "Player": {},                                      # Line 79
        "Goalkeeper": {}                                   # Line 80
    }
    
    if len(tracks) == 0:
        return result                                      # Line 84
    
    for track in tracks:                                   # Line 86 - LOOP TRACKS
        x1, y1, x2, y2, track_id, conf, class_idx, _ = track  # Line 87
        
        # Convert to proper types
        class_idx = int(class_idx)                         # Line 90
        track_id = int(track_id)                           # Line 91
        conf = float(conf)                                 # Line 92
        
        # Get class name
        class_name = class_names.get(class_idx, None)     # Line 95
        
        # Skip if not a class we track
        if class_name not in result:
            continue                                       # Line 99
        
        # Skip low confidence
        if conf < confidence_threshold:
            continue                                       # Line 103
        
        # Clip coordinates
        x1 = int(np.clip(x1, 0, width - 1))              # Line 106
        # ... clip others ...
        
        # Calculate bottom center
        bottom_center = [int((x1 + x2) / 2), y2]         # Line 113
        
        result[class_name][track_id] = {                  # Line 115
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "bottom_center": bottom_center
        }
```

**What happens**: Organizes into dictionary structure:
```python
frame_detections = {
    "Player": {
        1: {"bbox": [120,150,180,320], "confidence": 0.89, "bottom_center": [150,320]},
        2: {"bbox": [250,140,310,330], "confidence": 0.85, "bottom_center": [280,330]}
    },
    "Goalkeeper": {},
    "Ball": {}
}
```

---

### Step 17: Add Ball Detections
**File**: `football_tracker/main.py`  
**Lines**: 141-147

```python
# Add ball detections (not tracked, just detected)
ball_detections = self.detector.filter_detections(        # Line 142
    boxes,
    confidences,
    class_labels,
    frame.shape[:2]
).get("Ball", {})

frame_detections["Ball"] = ball_detections                # Line 147
```

**What happens**: Ball is detected but not tracked (too small/fast)

---

### Step 18: Store Frame Detections
**File**: `football_tracker/main.py`  
**Line**: 149

```python
detections_per_frame.append(frame_detections)             # Line 149
```

---

### Step 19: Continue Loop for Remaining Frames
**Loop continues**: Frames 1-239 follow same process:
- Detect (Step 13)
- Format (Step 14)
- Track (Step 15)
- Organize (Step 16)
- Add ball (Step 17)
- Store (Step 18)

**Console output**:
```
  Frame 100/240
  Frame 200/240
  Completed all 240 frames
```

**Time**: ~50-90ms per frame × 240 = 12-22 seconds total

---

## PHASE 4: TEAM ASSIGNMENT

### Step 20: Start Team Assignment
**File**: `football_tracker/main.py`  
**Line**: 78

```python
print("\nStep 3/5: Assigning teams...")                   # Line 78
team_mapping = self.team_assigner.assign_teams(           # Line 79
    frames,
    detections_per_frame
)
```

**Console output**:
```

Step 3/5: Assigning teams...
Assigning teams based on jersey colors...
```

**Triggers**: `football_tracker/team_assigner.py` - assign_teams  
**Lines**: 27-103

```python
def assign_teams(
    self,
    frames: List[np.ndarray],
    detections_per_frame: List[Dict]
) -> Dict[int, int]:
    print("Assigning teams based on jersey colors...")    # Line 39
    
    # First pass: collect color frequencies for each player
    player_color_counts = {}                               # Line 42
    
    for frame, detections in zip(frames, detections_per_frame):  # Line 44
        if "Player" not in detections:
            continue                                       # Line 46
        
        players = detections["Player"]                     # Line 48
        all_bboxes = [item["bbox"] for item in players.values()]  # Line 49
        
        for player_id, detection in players.items():      # Line 51 - EACH PLAYER
            bbox = detection["bbox"]                       # Line 52
            x1, y1, x2, y2 = bbox
            
            # Calculate overlap with other detections
            overlap_ratio = self._calculate_max_overlap_ratio(  # Line 56
                bbox, all_bboxes
            )
            
            # Skip heavily overlapped detections
            if overlap_ratio > self.config.overlap_threshold:
                continue                                   # Line 62
            
            # Extract player crop
            crop = frame[y1:y2, x1:x2]                    # Line 65
            if crop.size == 0:
                continue
            
            # Detect dominant colors
            color_counts = self._detect_colors(crop)       # Line 70
```

---

### Step 21: Detect Colors (for Player 1, Frame 0)
**File**: `football_tracker/team_assigner.py` - _detect_colors  
**Lines**: 195-268

```python
def _detect_colors(self, image: np.ndarray) -> List[Tuple[str, int]]:
    if image.size == 0:
        return []                                          # Line 207
    
    # Convert to RGB for KMeans clustering
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # Line 210
    
    # Use KMeans to segment player from background
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)  # Line 213
    kmeans.fit(rgb_image.reshape(-1, 3))                  # Line 214 - CLUSTER
    labels = kmeans.labels_.reshape(image.shape[:2])      # Line 215
    
    # Determine which cluster is background
    corners = [
        labels[0, 0], labels[0, -1],
        labels[-1, 0], labels[-1, -1]
    ]
    background_cluster = max(set(corners), key=corners.count)  # Line 222
    player_cluster = 1 if background_cluster == 0 else 0   # Line 223
    
    # Create mask for player region
    player_mask = (labels == player_cluster).astype(np.uint8) * 255  # Line 226
    
    # Convert to HSV for color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    # Line 229
    hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=player_mask)  # Line 230
    
    # Count pixels in each color range
    color_counts = {}                                      # Line 233
    
    for color_name, (lower, upper) in self.color_ranges.items():  # Line 235
        lower_bound = np.array(lower, dtype=np.uint8)     # Line 236
        upper_bound = np.array(upper, dtype=np.uint8)     # Line 237
        
        # Create color mask
        color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)  # Line 240
        color_mask = cv2.bitwise_and(color_mask, player_mask)  # Line 241
        
        # Count non-zero pixels
        count = cv2.countNonZero(color_mask)              # Line 244
        
        if count > 0:
            color_counts[color_name] = count              # Line 247
    
    # Combine red and red2
    if "red2" in color_counts:
        color_counts["red"] = color_counts.get("red", 0) + color_counts.pop("red2")  # Line 251
    
    # Sort by count
    color_counts = sorted(
        color_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )                                                      # Line 258
    
    return color_counts                                    # Line 260
```

**Example output** (internal):
```python
color_counts = [('red', 3500), ('white', 800), ('black', 200)]
# Player 1 wears red jersey
```

---

### Step 22: Accumulate Colors
**File**: `football_tracker/team_assigner.py`  
**Lines**: 72-79

```python
# Accumulate color counts for this player
if player_id not in player_color_counts:
    player_color_counts[player_id] = {}                   # Line 75
    
for color, count in color_counts:                         # Line 77
    if color not in player_color_counts[player_id]:
        player_color_counts[player_id][color] = 0         # Line 79
    # Weight by (1 - overlap_ratio)
    player_color_counts[player_id][color] += count * (1 - overlap_ratio)  # Line 81
```

**Loop continues**: Process all players in all frames

---

### Step 23: Determine Team Colors
**File**: `football_tracker/team_assigner.py`  
**Lines**: 84-93

```python
# Determine dominant color for each player
player_dominant_colors = {}                                # Line 85
for player_id, color_counts in player_color_counts.items():  # Line 86
    if color_counts:
        dominant_color = max(color_counts, key=color_counts.get)  # Line 88
        player_dominant_colors[player_id] = dominant_color  # Line 89

# Find the two most common colors
all_colors = list(player_dominant_colors.values())         # Line 92
color_frequency = Counter(all_colors)                      # Line 98
most_common_colors = color_frequency.most_common(2)        # Line 99
team_colors = [color for color, _ in most_common_colors]  # Line 100
```

**Example** (internal):
```python
player_dominant_colors = {
    1: 'red',
    2: 'red',
    3: 'blue',
    4: 'blue',
    5: 'red',
    # ...
}

most_common_colors = [('red', 6), ('blue', 5)]
team_colors = ['red', 'blue']
```

---

### Step 24: Assign Teams
**File**: `football_tracker/team_assigner.py`  
**Lines**: 103-131

```python
# Create color to team mapping
color_to_team = {color: i for i, color in enumerate(team_colors)}  # Line 105

# Second pass: assign teams
team_mapping = {}                                          # Line 108
for player_id, dominant_color in player_dominant_colors.items():  # Line 109
    if dominant_color in color_to_team:
        team_mapping[player_id] = color_to_team[dominant_color]  # Line 112
    else:
        # Outlier handling...                              # Line 114-125
        
print(f"Assigned {len(team_mapping)} players to teams")   # Line 127
```

**Console output**:
```
Assigned 11 players to teams
Team 0: 6 players
Team 1: 5 players
```

**Result**:
```python
team_mapping = {
    1: 0,  # Player 1 → Team 0 (red)
    2: 0,  # Player 2 → Team 0 (red)
    3: 1,  # Player 3 → Team 1 (blue)
    4: 1,  # Player 4 → Team 1 (blue)
    # ...
}
```

---

## PHASE 5: DATA PROCESSING

### Step 25: Start Processing
**File**: `football_tracker/main.py`  
**Lines**: 83-87

```python
print("\nStep 4/5: Processing tracking data...")          # Line 83
df, team_mapping = self.processor.process(                # Line 84
    detections_per_frame,
    team_mapping
)
```

**Console output**:
```

Step 4/5: Processing tracking data...
Processing tracking data...
```

**Triggers**: `football_tracker/processor.py` - DataProcessor.process  
**Lines**: 32-66

```python
def process(
    self,
    detections_per_frame: List[Dict],
    team_mapping: Dict[int, int]
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    print("Processing tracking data...")                   # Line 41
    
    # Create initial DataFrame
    df = self._create_dataframe(detections_per_frame)     # Line 44
```

---

### Step 26: Create DataFrame
**File**: `football_tracker/processor.py` - _create_dataframe  
**Lines**: 68-146

```python
def _create_dataframe(self, detections_per_frame: List[Dict]) -> pd.DataFrame:
    data = {}                                              # Line 77
    
    for frame_idx, detections in enumerate(detections_per_frame):  # Line 79
        frame_data = {}                                    # Line 80
        
        has_person = False                                 # Line 82
        
        # Add player and goalkeeper detections
        for class_name in ["Player", "Goalkeeper"]:        # Line 85
            if class_name not in detections:
                continue
            
            for obj_id, detection in detections[class_name].items():  # Line 89
                col_name = f"{class_name}_{obj_id}"        # Line 90
                bottom_center = detection["bottom_center"] # Line 91
                frame_data[col_name] = tuple(bottom_center)  # Line 92
                has_person = True                          # Line 93
        
        # Add ball detection
        if "Ball" in detections and len(detections["Ball"]) > 0:  # Line 96
            ball_detections = detections["Ball"]           # Line 97
            # Sort by confidence and take the best
            best_ball = max(
                ball_detections.values(),
                key=lambda x: x["confidence"]
            )                                              # Line 102
            frame_data["Ball"] = tuple(best_ball["bottom_center"])  # Line 103
        
        # Only include frames with at least one person
        if has_person:
            data[frame_idx] = frame_data                   # Line 107
    
    # Create DataFrame
    df = pd.DataFrame(data).T                              # Line 110
    
    # Remove sparse columns (<1% non-null)
    if len(df) > 0:
        threshold = 0.01 * len(df)                         # Line 114
        df = df.loc[:, df.notna().sum() >= threshold]     # Line 115
    
    return df                                              # Line 117
```

**What happens**: Creates DataFrame like:
```
        Player_1      Player_2      Ball
0    (150, 320)    (280, 330)   (450, 250)
1    (152, 321)    (282, 331)   (452, 248)
2    (154, 322)          NaN    (454, 246)
...
```

---

### Step 27: Interpolate Ball Positions
**File**: `football_tracker/processor.py`  
**Lines**: 49-51

```python
# Interpolate missing ball positions
if "Ball" in df.columns:
    df = self._interpolate_column(df, "Ball", fill=True)  # Line 51
```

**Triggers**: `football_tracker/processor.py` - _interpolate_column  
**Lines**: 148-186

```python
def _interpolate_column(
    self,
    df: pd.DataFrame,
    col_name: str,
    fill: bool = False
) -> pd.DataFrame:
    if col_name not in df.columns:
        return df                                          # Line 161
    
    s = df[col_name]                                       # Line 163
    
    # Extract x and y coordinates
    x = s.apply(lambda v: v[0] if isinstance(v, (list, tuple)) else np.nan)  # Line 166
    y = s.apply(lambda v: v[1] if isinstance(v, (list, tuple)) else np.nan)  # Line 167
    
    # Interpolate
    if fill:
        x = x.interpolate(method="linear").bfill().ffill()  # Line 171
        y = y.interpolate(method="linear").bfill().ffill()  # Line 172
    else:
        x = x.interpolate(method="linear", limit_area="inside")  # Line 174
        y = y.interpolate(method="linear", limit_area="inside")  # Line 175
    
    # Combine back
    combined = pd.Series(
        [(xi, yi) if not (math.isnan(xi) or math.isnan(yi)) else np.nan
         for xi, yi in zip(x, y)],
        index=s.index
    )                                                      # Line 182
    
    df[col_name] = combined                                # Line 184
    return df                                              # Line 185
```

**What happens**: Fills gaps in ball trajectory using linear interpolation

---

### Step 28: Merge Fragmented IDs
**File**: `football_tracker/processor.py`  
**Line**: 54

```python
df, team_mapping = self._merge_fragmented_ids(df, team_mapping)  # Line 54
```

**Triggers**: `football_tracker/processor.py` - _merge_fragmented_ids  
**Lines**: 212-279

```python
def _merge_fragmented_ids(
    self,
    df: pd.DataFrame,
    team_mapping: Dict[int, int]
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    # Merge goalkeepers with players if same ID
    goalkeeper_cols = [c for c in df.columns if "Goalkeeper" in c]  # Line 224
    for col in goalkeeper_cols:                            # Line 225
        player_id = col.split("_")[1]                      # Line 226
        player_col = f"Player_{player_id}"                 # Line 227
        
        if player_col in df.columns:
            df[col] = df[player_col].combine_first(df[col])  # Line 230
            df.drop(columns=[player_col], inplace=True)    # Line 231
    
    # Find columns to merge
    temporal_threshold = int(self.fps * self.config.temporal_threshold_seconds)  # Line 234
    
    to_merge = []                                          # Line 239
    
    # Check each pair
    for col_type, cols in [("Player", player_cols), ...]:  # Line 242
        for i, col1 in enumerate(cols):                    # Line 243
            for col2 in cols[i+1:]:                        # Line 244
                if self._should_merge(df, col1, col2, temporal_threshold, team_mapping):  # Line 245
                    to_merge.append((col1, col2))          # Line 246
    
    # Perform merges
    merged_cols = {}                                       # Line 251
    
    for col1, col2 in to_merge:                            # Line 256
        root1 = self._find_root(col1, merged_cols)        # Line 257
        root2 = self._find_root(col2, merged_cols)        # Line 258
        
        if root1 != root2:
            df[root1] = df[root1].combine_first(df[root2])  # Line 261
            df.drop(columns=[root2], inplace=True)         # Line 262
            merged_cols[root2] = root1                     # Line 263
    
    return df, team_mapping                                # Line 271
```

**What happens**: 
- Finds IDs that represent same player
- Merges based on temporal/spatial/team criteria
- Updates DataFrame and team mapping

**Console output** (if merges found):
```
Merging 2 columns
To Merge: [('Player_5', 'Player_8'), ('Player_3', 'Player_9')]
```

---

### Step 29: Interpolate Player Positions
**File**: `football_tracker/processor.py`  
**Lines**: 57-63

```python
# Interpolate player positions
for col in df.columns:                                     # Line 58
    if col == "Ball":
        continue                                           # Line 60
    df = self._interpolate_column(df, col, fill=False)    # Line 61
    
    # Optional smoothing
    if self.config.smooth:
        df = self._smooth_column(df, col)                  # Line 64
```

**Console output**:
```
Processed data: 240 frames, 12 tracked objects
```

---

## PHASE 6: VISUALIZATION AND SAVING

### Step 30: Save Tracking Data
**File**: `football_tracker/main.py`  
**Lines**: 90-94

```python
print("\nStep 5/5: Saving results...")                    # Line 90

# Save tracking data
save_tracking_data(df, team_mapping, output_dir, fps)     # Line 93
```

**Console output**:
```

Step 5/5: Saving results...
```

**Triggers**: `football_tracker/utils.py` - save_tracking_data  
**Lines**: 73-151

```python
def save_tracking_data(
    df: pd.DataFrame,
    team_mapping: Dict[int, int],
    output_dir: str,
    fps: int
):
    os.makedirs(output_dir, exist_ok=True)                # Line 85
    
    # Save metadata
    metadata = {
        "fps": fps,                                        # Line 89
        "num_frames": len(df),                             # Line 90
        "team_mapping": {str(k): int(v) for k, v in team_mapping.items()}  # Line 91
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")  # Line 94
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)                   # Line 96
    print(f"Saved metadata: {metadata_path}")              # Line 97
    
    # Save raw DataFrame
    raw_data_path = os.path.join(output_dir, "raw_data.json")  # Line 100
    df.to_json(raw_data_path, orient="records", indent=2)  # Line 101
    print(f"Saved raw data: {raw_data_path}")              # Line 102
```

**Console output**:
```
Saved metadata: output/input/metadata.json
Saved raw data: output/input/raw_data.json
```

---

### Step 31: Format and Save Processed Data
**File**: `football_tracker/utils.py`  
**Lines**: 105-148

```python
# Save formatted data
formatted_data = []                                        # Line 106
for frame_idx in df.index:                                 # Line 107
    frame_data = {
        "frame": int(frame_idx),                           # Line 109
        "time": f"{frame_idx // fps // 60:02d}:{frame_idx // fps % 60:02d}",  # Line 110
        "detections": []                                   # Line 111
    }
    
    row = df.loc[frame_idx]                                # Line 114
    
    for col in df.columns:                                 # Line 116
        val = row[col]                                     # Line 117
        if pd.isna(val):
            continue                                       # Line 119
        
        if col == "Ball":
            detection = {                                  # Line 122
                "id": "Ball",
                "type": "Ball",
                "x": float(val[0]),
                "y": float(val[1])
            }
        else:
            # Parse player/goalkeeper...                   # Line 128-137
    
    formatted_data.append(frame_data)                      # Line 145

formatted_path = os.path.join(output_dir, "processed_data.json")  # Line 147
with open(formatted_path, "w") as f:
    json.dump(formatted_data, f, indent=2)                 # Line 149
print(f"Saved processed data: {formatted_path}")           # Line 150
```

**Console output**:
```
Saved processed data: output/input/processed_data.json
```

---

### Step 32: Create Annotated Video
**File**: `football_tracker/main.py`  
**Lines**: 96-102

```python
# Create annotated video
annotated_path = os.path.join(output_dir, "annotated.mp4")  # Line 97
self.visualizer.create_annotated_video(                   # Line 98
    frames,
    df,
    team_mapping,
    annotated_path,
    fps
)
```

**Triggers**: `football_tracker/visualizer.py` - create_annotated_video  
**Lines**: 134-177

```python
def create_annotated_video(
    self,
    frames: List[np.ndarray],
    df: pd.DataFrame,
    team_mapping: Dict[int, int],
    output_path: str,
    fps: int
) -> str:
    print(f"Creating annotated video: {output_path}")     # Line 150
    
    annotated_frames = []                                  # Line 152
    
    for frame_idx, frame in enumerate(frames):             # Line 154 - LOOP FRAMES
        if frame_idx in df.index:
            annotated_frame = self.draw_from_dataframe(    # Line 156
                frame, frame_idx, df, team_mapping
            )
        else:
            annotated_frame = frame.copy()                 # Line 160
        
        annotated_frames.append(annotated_frame)           # Line 162
    
    # Write video
    height, width = frames[0].shape[:2]                    # Line 165
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")              # Line 166
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Line 167
    
    for frame in annotated_frames:                         # Line 169 - WRITE FRAMES
        out.write(frame)                                   # Line 170
    
    out.release()                                          # Line 172
    print(f"Saved annotated video: {output_path}")         # Line 173
    
    return output_path                                     # Line 175
```

---

### Step 33: Draw Single Frame (Frame 0)
**File**: `football_tracker/visualizer.py` - draw_from_dataframe  
**Lines**: 84-132

```python
def draw_from_dataframe(
    self,
    frame: np.ndarray,
    frame_idx: int,
    df: pd.DataFrame,
    team_mapping: Dict[int, int]
) -> np.ndarray:
    if frame_idx not in df.index:
        return frame                                       # Line 100
    
    annotated_frame = frame.copy()                         # Line 102
    row = df.loc[frame_idx]                                # Line 103
    
    for col in df.columns:                                 # Line 105 - EACH OBJECT
        val = row[col]                                     # Line 106
        if pd.isna(val):
            continue                                       # Line 108
        
        x, y = int(val[0]), int(val[1])                   # Line 110
        
        if col == "Ball":
            # Draw ball
            if self.config.show_ball:
                bottom_point = (x, y - 20)                 # Line 114
                top_left = (x - 5, y - 30)                 # Line 115
                top_right = (x + 5, y - 30)                # Line 116
                pts = np.array([bottom_point, top_left, top_right])  # Line 117
                cv2.drawContours(                          # Line 118
                    annotated_frame,
                    [pts],
                    0,
                    self.config.ball_color,
                    -1
                )
        else:
            # Parse object type and ID
            parts = col.split("_")                         # Line 126
            obj_type = parts[0]                            # Line 127
            obj_id = int(parts[1])                         # Line 128
            
            # Determine color
            if obj_type == "Goalkeeper":
                color = self.config.goalkeeper_color       # Line 132
            else:
                team_id = team_mapping.get(obj_id, 0)     # Line 134
                color = self.config.team_colors.get(team_id, (255, 255, 255))  # Line 135
            
            # Draw ellipse
            cv2.ellipse(                                   # Line 138
                annotated_frame,
                (x, y),
                (35, 18),
                0,
                -45,
                235,
                color,
                2
            )
            
            # Draw ID
            if self.config.show_ids:
                cv2.putText(                               # Line 149
                    annotated_frame,
                    str(obj_id),
                    (x - 10, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
    
    return annotated_frame                                 # Line 159
```

**What happens**: Draws on frame:
- Red ellipse + ID "1" at (150, 320) - Team 0 player
- Blue ellipse + ID "3" at (280, 330) - Team 1 player
- Green triangle at (450, 250) - Ball

**Loop continues**: Annotate all 240 frames

**Console output**:
```
Creating annotated video: output/input/annotated.mp4
Saved annotated video: output/input/annotated.mp4
```

**Time**: ~5-10ms per frame × 240 = 1-2 seconds

---

### Step 34: Print Summary
**File**: `football_tracker/main.py`  
**Line**: 105

```python
print_summary(df, team_mapping, fps)                      # Line 105
```

**Triggers**: `football_tracker/utils.py` - print_summary  
**Lines**: 198-249

```python
def print_summary(
    df: pd.DataFrame,
    team_mapping: Dict[int, int],
    fps: int
):
    print("\n" + "="*50)                                   # Line 207
    print("TRACKING SUMMARY")                              # Line 208
    print("="*50)                                          # Line 209
    
    # Time information
    duration_seconds = len(df) / fps                       # Line 212
    print(f"\nVideo Duration: {duration_seconds:.2f} seconds")  # Line 213
    print(f"Frames Processed: {len(df)}")                  # Line 214
    print(f"FPS: {fps}")                                   # Line 215
    
    # Player information
    player_cols = [c for c in df.columns if "Player" in c or "Goalkeeper" in c]  # Line 218
    print(f"\nTotal Players Tracked: {len(player_cols)}")  # Line 219
    
    # Team distribution
    team_counts = {}                                       # Line 222
    for col in player_cols:                                # Line 223
        player_id = int(col.split("_")[1])                # Line 224
        team_id = team_mapping.get(player_id, -1)         # Line 225
        team_counts[team_id] = team_counts.get(team_id, 0) + 1  # Line 226
    
    print(f"\nTeam Distribution:")                         # Line 228
    for team_id, count in sorted(team_counts.items()):    # Line 229
        print(f"  Team {team_id}: {count} players")       # Line 230
    
    # Ball tracking
    if "Ball" in df.columns:
        ball_frames = df["Ball"].notna().sum()            # Line 234
        ball_percentage = (ball_frames / len(df)) * 100   # Line 235
        print(f"\nBall Detection:")                        # Line 236
        print(f"  Frames with ball: {ball_frames} ({ball_percentage:.1f}%)")  # Line 237
    
    print("="*50 + "\n")                                   # Line 239
```

**Console output**:
```

==================================================
TRACKING SUMMARY
==================================================

Video Duration: 10.00 seconds
Frames Processed: 240
FPS: 24

Total Players Tracked: 11

Team Distribution:
  Team 0: 6 players
  Team 1: 5 players

Ball Detection:
  Frames with ball: 220 (91.7%)
==================================================
```

---

### Step 35: Final Success Message
**File**: `football_tracker/main.py`  
**Line**: 107

```python
print(f"\n✓ Processing complete! Results saved to: {output_dir}\n")  # Line 107

return output_dir                                          # Line 109
```

**Console output**:
```

✓ Processing complete! Results saved to: output/input

```

---

### Step 36: Return to Main Entry Point
**File**: `football_tracker/main.py`  
**Lines**: 227-242

```python
print("\n" + "="*50)                                       # Line 227
print("SUCCESS!")                                          # Line 228
print("="*50)                                              # Line 229
print(f"\nResults saved to: {output_dir}")                # Line 230
print("\nOutput files:")                                   # Line 231
print("  - annotated.mp4       : Video with tracking visualization")  # Line 232
print("  - raw_data.json       : Raw tracking data")      # Line 233
print("  - processed_data.json : Processed tracking data")  # Line 234
print("  - metadata.json       : Video and team metadata")  # Line 235
print("\n")                                                # Line 236

return 0                                                   # Line 238
```

**Console output**:
```

==================================================
SUCCESS!
==================================================

Results saved to: output/input

Output files:
  - annotated.mp4       : Video with tracking visualization
  - raw_data.json       : Raw tracking data
  - processed_data.json : Processed tracking data
  - metadata.json       : Video and team metadata

```

---

### Step 37: Program Exits
**File**: `football_tracker/main.py`  
**Line**: 245

```python
sys.exit(main())  # Exit with code 0 (success)
```

**Program terminates successfully**

---

## EXECUTION TIME BREAKDOWN

Total execution for 10-second video (240 frames at 24 FPS) on GPU:

| Phase | Time | Operations |
|-------|------|------------|
| **Initialization** | ~3-5 seconds | Load YOLO, BoTSORT, configs |
| **Video Reading** | ~1-2 seconds | Load and decode 240 frames |
| **Detection & Tracking** | ~12-22 seconds | 240 iterations × 50-90ms |
| **Team Assignment** | ~5-10 seconds | Color analysis on all frames |
| **Data Processing** | ~1-2 seconds | DataFrame ops, interpolation |
| **Visualization** | ~1-2 seconds | Draw + encode video |
| **Saving** | ~1-2 seconds | Write JSON files |
| **TOTAL** | **~24-45 seconds** | End-to-end |

---

## KEY EXECUTION POINTS

### Critical Loops:
1. **Frame Loop** (main.py:116) - Processes each frame
2. **Team Color Loop** (team_assigner.py:44) - Analyzes all players
3. **Visualization Loop** (visualizer.py:154) - Draws all frames

### Memory Usage:
- **Frames**: ~200-500MB (depends on resolution)
- **YOLO Model**: ~500MB
- **BoTSORT ReID**: ~500MB
- **DataFrame**: ~10-50MB
- **Peak**: ~1.5-2GB total

### CPU/GPU Usage:
- **Detection**: 90% GPU utilization
- **Tracking**: 70% GPU + CPU
- **Team Assignment**: 100% CPU (one core)
- **Processing**: 100% CPU (one core)

---

## SUMMARY

**Entry**: `main.py:245` → `main()` → `FootballTracker.process_video()`

**Exit**: `main.py:245` ← `return 0` ← `sys.exit()`

**Total Lines Executed**: ~2,000+ lines across 8 modules

**Output**: 4 files in `output/input/` directory

---

This is the complete, line-by-line execution flow! 🎯
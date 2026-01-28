# Football Player Tracker - Google Colab Notebook
#
# This notebook provides an easy-to-use interface for running
# the football tracker in Google Colab.

"""
# Football Player Tracker - Colab

Easy-to-use interface for detecting, tracking, and assigning teams to football players.

## Quick Start

1. Run the installation cell
2. Upload your video
3. Run the tracking cell
4. Download results!
"""

# Cell 1: Installation
print("Installing Football Player Tracker...")
!git clone https://github.com/your-username/football-tracker.git
%cd football-tracker
!pip install -q -r requirements.txt
print("Installation complete!")

# Cell 1a: GTA-Link Installation (Optional)
# Toggle this to enable/disable GTA-Link post-processing
USE_GTA_LINK = True

if USE_GTA_LINK:
    print("\nInstalling GTA-Link for tracklet refinement...")
    !git clone https://github.com/sjc042/gta-link.git
    !pip install -q scikit-learn loguru seaborn Cython
    %cd gta-link/reid
    !python setup.py develop --quiet
    %cd ../..
    print("GTA-Link installation complete!")
    print("Note: Model checkpoint expected at gta-link/reid_checkpoints/sports_model.pth.tar-60")

# Cell 1b: GTA-Link Configuration
from dataclasses import dataclass

@dataclass
class GTALinkConfig:
    """Configuration for GTA-Link post-processing."""
    model_path: str = "gta-link/reid_checkpoints/sports_model.pth.tar-60"
    # Split parameters (DBSCAN clustering)
    eps: float = 0.6           # DBSCAN epsilon (cosine distance threshold)
    min_samples: int = 10      # DBSCAN min points for core sample
    max_k: int = 3             # Max clusters after splitting
    min_len: int = 100         # Min tracklet length to attempt split
    # Merge parameters
    merge_dist_thres: float = 0.4  # Max cosine distance for merging
    spatial_factor: float = 1.0     # Scaling for spatial constraint ranges
    # Processing options
    use_split: bool = True     # Enable tracklet splitting
    use_connect: bool = True   # Enable tracklet merging
    batch_size: int = 64       # Batch size for feature extraction

gta_config = GTALinkConfig()
print("GTA-Link Configuration:")
print(f"  Model: {gta_config.model_path}")
print(f"  Split: eps={gta_config.eps}, min_samples={gta_config.min_samples}, max_k={gta_config.max_k}")
print(f"  Merge: threshold={gta_config.merge_dist_thres}, spatial_factor={gta_config.spatial_factor}")

# Cell 2: Import and Setup
from football_tracker import FootballTracker, MainConfig
from google.colab import files
import os

if USE_GTA_LINK:
    from gta_link_processor import run_gta_link_refinement

print("Ready to process videos!")
print("\nNext step: Upload your video in the next cell")

# Cell 3: Upload Video
print("Please select your video file...")
uploaded = files.upload()

if not uploaded:
    raise ValueError("No file uploaded. Please upload a video file.")

video_path = list(uploaded.keys())[0]
print(f"\nUploaded: {video_path}")

# Cell 4: Configure Tracker
# Adjust these settings as needed
config = MainConfig()
config.fps = 24  # Processing FPS (lower = faster)
config.detector.confidence_threshold = 0.35  # Detection confidence
config.visualizer.show_bboxes = False  # Show bounding boxes
config.visualizer.show_ids = True  # Show player IDs

print("Configuration:")
print(f"  FPS: {config.fps}")
print(f"  Confidence: {config.detector.confidence_threshold}")
print(f"  Show bboxes: {config.visualizer.show_bboxes}")
print(f"  GTA-Link enabled: {USE_GTA_LINK}")

# Cell 5: Run Tracking
print("\nStarting tracking pipeline...")
print("="*50)

# Initialize tracker
tracker = FootballTracker(config)

# Read video
from utils import read_video, create_output_directory, save_tracking_data, print_summary
frames, fps = read_video(video_path, config.fps)
output_dir = create_output_directory(video_path, config.output_dir)

# Initialize processor with actual FPS
from processor import DataProcessor
tracker.processor = DataProcessor(config.processor, fps)

# Step 1: Detection and Tracking
print("\nStep 1/5: Detecting and tracking objects...")
detections_per_frame = tracker._detect_and_track(frames)

# Step 1.5: GTA-Link Refinement (Optional)
if USE_GTA_LINK:
    print("\nStep 1.5: Running GTA-Link tracklet refinement...")
    detections_per_frame, _ = run_gta_link_refinement(
        detections_per_frame,
        frames,
        {},  # Empty team_mapping - will be assigned after refinement
        gta_config
    )

# Step 2: Team Assignment
print("\nStep 2/5: Assigning teams...")
team_mapping = tracker.team_assigner.assign_teams(frames, detections_per_frame)

# Step 3: Data Processing
print("\nStep 3/5: Processing tracking data...")
df, team_mapping = tracker.processor.process(detections_per_frame, team_mapping)

# Step 4: Save Results
print("\nStep 4/5: Saving results...")
save_tracking_data(df, team_mapping, output_dir, fps)

# Step 5: Create Annotated Video
print("\nStep 5/5: Creating annotated video...")
annotated_path = os.path.join(output_dir, "annotated.mp4")
tracker.visualizer.create_annotated_video(
    frames, df, team_mapping, annotated_path, fps
)

print_summary(df, team_mapping, fps)

print("\n" + "="*50)
print("Processing complete!")
print(f"\nResults saved to: {output_dir}")

# Cell 6: Preview Results
# Display sample frames from annotated video
import cv2
from IPython.display import Image, display
import matplotlib.pyplot as plt

annotated_path = os.path.join(output_dir, "annotated.mp4")
cap = cv2.VideoCapture(annotated_path)

# Get 4 evenly spaced frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_indices = [int(i * total_frames / 5) for i in range(1, 5)]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, frame_num in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(frame_rgb)
        axes[idx].set_title(f"Frame {frame_num}")
        axes[idx].axis('off')

cap.release()
plt.tight_layout()
plt.show()

print("\nPreview of annotated video shown above")

# Cell 7: View Tracking Statistics
import json
import pandas as pd

# Load metadata
metadata_path = os.path.join(output_dir, "metadata.json")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print("Tracking Statistics")
print("="*50)
print(f"Total Frames: {metadata['num_frames']}")
print(f"FPS: {metadata['fps']}")
print(f"Duration: {metadata['num_frames'] / metadata['fps']:.2f} seconds")

team_mapping = metadata['team_mapping']
team_counts = {}
for player_id, team_id in team_mapping.items():
    team_counts[team_id] = team_counts.get(team_id, 0) + 1

print(f"\nPlayers Tracked: {len(team_mapping)}")
print(f"Team 0: {team_counts.get(0, 0)} players")
print(f"Team 1: {team_counts.get(1, 0)} players")

# Cell 8: View Sample Tracking Data
# Load processed data
processed_path = os.path.join(output_dir, "processed_data.json")
with open(processed_path, 'r') as f:
    data = json.load(f)

# Show first few frames
print("\nSample Tracking Data (First 3 Frames)")
print("="*50)
for frame_data in data[:3]:
    print(f"\nFrame {frame_data['frame']} (Time: {frame_data['time']})")
    print(f"  Detections: {len(frame_data['detections'])}")
    for det in frame_data['detections'][:3]:  # Show first 3 detections
        print(f"    {det['type']} {det['id']}: ({det['x']:.1f}, {det['y']:.1f})")

# Cell 9: Download Results
print("Preparing download...")

# Zip the results
!zip -r results.zip {output_dir}

print("\nDownloading results...")
files.download(f"results.zip")

print("\nDownload complete!")
print("\nThe zip file contains:")
print("  - annotated.mp4       : Video with tracking")
print("  - raw_data.json       : Raw tracking data")
print("  - processed_data.json : Processed data")
print("  - metadata.json       : Metadata")

# Cell 10: Advanced - Custom Configuration
# For advanced users who want more control

"""
Advanced Configuration Example:

from football_tracker import (
    MainConfig,
    DetectorConfig,
    TrackerConfig,
    TeamAssignerConfig,
    ProcessorConfig,
    VisualizerConfig
)

# Custom detector settings
detector_config = DetectorConfig(
    model_path="yolov8n.pt",
    confidence_threshold=0.4,
    device="cuda"
)

# Custom tracker settings
tracker_config = TrackerConfig(
    tracker_type="botsort",
    device="cuda"
)

# Custom team assignment
team_config = TeamAssignerConfig(
    n_clusters=2,
    overlap_threshold=0.35
)

# Custom processing
processor_config = ProcessorConfig(
    interpolate=True,
    smooth=False,
    temporal_threshold_seconds=1.5
)

# Custom visualization
visualizer_config = VisualizerConfig(
    show_ids=True,
    show_bboxes=True,
    team_colors={
        0: (0, 0, 255),    # Red for team 0
        1: (255, 0, 0)     # Blue for team 1
    }
)

# Combine all configs
config = MainConfig(
    detector=detector_config,
    tracker=tracker_config,
    team_assigner=team_config,
    processor=processor_config,
    visualizer=visualizer_config,
    fps=24
)

# Use custom config
tracker = FootballTracker(config)
output_dir = tracker.process_video(video_path)

# GTA-Link with custom settings
gta_config = GTALinkConfig(
    eps=0.5,           # Tighter clustering
    min_samples=15,    # More robust clusters
    max_k=2,           # Max 2 identities per tracklet
    min_len=50,        # Process shorter tracklets
    merge_dist_thres=0.35,  # More aggressive merging
    spatial_factor=1.2
)
"""

print("See the cell code for advanced configuration examples")

# Cell 11: Troubleshooting
"""
Common Issues and Solutions:

1. Low Detection Accuracy
   - Increase detector confidence threshold
   - Use larger YOLO model (yolov8m.pt or yolov8l.pt)

2. ID Switching
   - Adjust temporal_threshold_seconds
   - Adjust spatial_threshold_per_frame
   - Enable GTA-Link for better ID consistency

3. Wrong Team Assignment
   - Check if jersey colors are too similar
   - Consider manual correction of team_mapping

4. Out of Memory
   - Lower FPS (e.g., fps=12)
   - Use smaller model (yolov8n.pt)
   - Reduce GTA-Link batch_size

5. Slow Processing
   - Lower FPS
   - Use smaller YOLO model
   - Enable GPU (automatic in Colab)

6. GTA-Link Issues
   - Ensure model checkpoint exists at gta-link/reid_checkpoints/
   - Try adjusting eps (0.5-0.7) and merge_dist_thres (0.3-0.5)
   - Set USE_GTA_LINK = False to disable
"""

print("Troubleshooting tips available in cell source")

# Cell 12: Clean Up (Optional)
"""
# Uncomment to clean up files after download
!rm -rf {output_dir}
!rm results.zip
!rm {video_path}
if USE_GTA_LINK:
    !rm -rf gta-link
print("Cleanup complete")
"""

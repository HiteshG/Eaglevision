"""
Example usage scripts for Football Tracker.
Demonstrates various use cases and configurations.
"""

# Example 1: Basic Usage
def example_basic():
    """Most simple usage - just process a video with defaults."""
    from football_tracker import FootballTracker
    
    tracker = FootballTracker()
    output_dir = tracker.process_video("input_video.mp4")
    print(f"Results saved to: {output_dir}")


# Example 2: Custom Configuration
def example_custom_config():
    """Use custom configuration for all components."""
    from football_tracker import (
        FootballTracker,
        MainConfig,
        DetectorConfig,
        TrackerConfig,
        TeamAssignerConfig,
        ProcessorConfig,
        VisualizerConfig
    )
    
    # Configure detector
    detector_config = DetectorConfig(
        model_path="yolov8n.pt",
        confidence_threshold=0.4,
        device="cuda"
    )
    
    # Configure tracker
    tracker_config = TrackerConfig(
        tracker_type="botsort",
        device="cuda"
    )
    
    # Configure team assigner
    team_config = TeamAssignerConfig(
        n_clusters=2,
        overlap_threshold=0.3
    )
    
    # Configure processor
    processor_config = ProcessorConfig(
        interpolate=True,
        smooth=True,
        temporal_threshold_seconds=1.5,
        spatial_threshold_per_frame=12.0
    )
    
    # Configure visualizer
    visualizer_config = VisualizerConfig(
        show_ids=True,
        show_bboxes=True,
        team_colors={
            0: (0, 0, 255),    # Red
            1: (255, 0, 0)     # Blue
        }
    )
    
    # Combine configurations
    config = MainConfig(
        detector=detector_config,
        tracker=tracker_config,
        team_assigner=team_config,
        processor=processor_config,
        visualizer=visualizer_config,
        fps=24,
        output_dir="custom_output"
    )
    
    # Process video
    tracker = FootballTracker(config)
    output_dir = tracker.process_video("input_video.mp4")
    print(f"Results: {output_dir}")


# Example 3: Load and Visualize Existing Results
def example_load_results():
    """Load previously processed results and create new visualization."""
    from football_tracker import load_tracking_data, Visualizer, VisualizerConfig
    from football_tracker.utils import read_video
    
    # Load tracking data
    df, team_mapping, fps = load_tracking_data("output/my_video")
    
    # Load original video
    frames, _ = read_video("input_video.mp4", fps)
    
    # Create new visualization with different colors
    config = VisualizerConfig(
        team_colors={
            0: (0, 255, 255),    # Yellow
            1: (255, 0, 255)     # Magenta
        },
        show_bboxes=False
    )
    
    visualizer = Visualizer(config)
    visualizer.create_annotated_video(
        frames,
        df,
        team_mapping,
        "output/my_video/recolored.mp4",
        fps
    )


# Example 4: Process Multiple Videos
def example_batch_processing():
    """Process multiple videos in batch."""
    from football_tracker import FootballTracker, MainConfig
    import os
    
    # List of video files
    video_files = [
        "match1.mp4",
        "match2.mp4",
        "match3.mp4"
    ]
    
    # Create tracker with config
    config = MainConfig(fps=24)
    tracker = FootballTracker(config)
    
    # Process each video
    for video_file in video_files:
        if not os.path.exists(video_file):
            print(f"Skipping {video_file} - file not found")
            continue
        
        print(f"\nProcessing: {video_file}")
        try:
            output_dir = tracker.process_video(video_file)
            print(f"✓ Success: {output_dir}")
        except Exception as e:
            print(f"✗ Error processing {video_file}: {e}")


# Example 5: Custom Detector
def example_custom_detector():
    """Use a custom detector implementation."""
    import numpy as np
    from football_tracker import FootballTracker
    
    class CustomDetector:
        """Example custom detector."""
        
        def __init__(self):
            # Initialize your custom model here
            pass
        
        def detect(self, frame):
            """Detect objects in frame."""
            # Your detection logic here
            # Must return: boxes, confidences, class_labels
            boxes = np.array([[100, 100, 200, 300], [300, 200, 400, 400]])
            confidences = np.array([0.9, 0.85])
            class_labels = np.array([0, 0])  # 0 = Player
            return boxes, confidences, class_labels
        
        def get_detection_array(self, boxes, confidences, class_labels):
            """Format for tracker."""
            return np.hstack((
                boxes,
                confidences.reshape(-1, 1),
                class_labels.reshape(-1, 1)
            ))
        
        CLASS_NAMES = {0: "Player", 1: "Goalkeeper", 2: "Ball"}
    
    # Use custom detector
    tracker = FootballTracker()
    tracker.detector = CustomDetector()
    output_dir = tracker.process_video("input_video.mp4")


# Example 6: Extract Statistics
def example_extract_statistics():
    """Extract detailed statistics from tracking results."""
    from football_tracker import load_tracking_data
    import pandas as pd
    import numpy as np
    
    # Load data
    df, team_mapping, fps = load_tracking_data("output/my_video")
    
    # Calculate statistics per player
    player_cols = [c for c in df.columns if "Player" in c or "Goalkeeper" in c]
    
    print("\nPlayer Statistics:")
    print("=" * 60)
    
    for col in player_cols:
        player_id = int(col.split("_")[1])
        team_id = team_mapping.get(player_id, -1)
        
        # Get valid positions
        positions = df[col].dropna()
        
        if len(positions) < 2:
            continue
        
        # Calculate distance traveled
        coords = np.array([list(pos) for pos in positions])
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        total_distance = np.sum(distances)
        
        # Calculate time on field
        time_on_field = len(positions) / fps
        
        # Average speed (pixels per second)
        avg_speed = total_distance / time_on_field if time_on_field > 0 else 0
        
        print(f"\nPlayer {player_id} (Team {team_id}):")
        print(f"  Time on field: {time_on_field:.1f}s")
        print(f"  Distance: {total_distance:.1f} pixels")
        print(f"  Avg speed: {avg_speed:.1f} px/s")
        print(f"  Frames visible: {len(positions)}/{len(df)} ({len(positions)/len(df)*100:.1f}%)")


# Example 7: Filter and Export Specific Players
def example_filter_players():
    """Export tracking data for specific players only."""
    from football_tracker import load_tracking_data
    import json
    
    # Load data
    df, team_mapping, fps = load_tracking_data("output/my_video")
    
    # Filter for specific players
    target_players = [5, 7, 12]
    
    # Create filtered output
    filtered_data = []
    
    for frame_idx in df.index:
        frame_data = {
            "frame": int(frame_idx),
            "time": f"{frame_idx // fps // 60:02d}:{frame_idx // fps % 60:02d}",
            "detections": []
        }
        
        row = df.loc[frame_idx]
        
        for col in df.columns:
            if col == "Ball":
                continue
            
            val = row[col]
            if pd.isna(val):
                continue
            
            player_id = int(col.split("_")[1])
            
            if player_id not in target_players:
                continue
            
            detection = {
                "id": player_id,
                "x": float(val[0]),
                "y": float(val[1])
            }
            
            frame_data["detections"].append(detection)
        
        if frame_data["detections"]:
            filtered_data.append(frame_data)
    
    # Save filtered data
    with open("filtered_players.json", "w") as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Exported {len(filtered_data)} frames for players: {target_players}")


# Example 8: Create Highlight Clip
def example_create_highlight():
    """Create a highlight clip from specific frames."""
    from football_tracker import load_tracking_data, Visualizer, VisualizerConfig
    from football_tracker.utils import read_video, write_video
    
    # Load data
    df, team_mapping, fps = load_tracking_data("output/my_video")
    frames, _ = read_video("input_video.mp4", fps)
    
    # Define highlight frames (e.g., frames 100-200)
    start_frame = 100
    end_frame = 200
    
    # Filter frames
    highlight_frames = frames[start_frame:end_frame]
    highlight_df = df.loc[start_frame:end_frame]
    
    # Create visualization
    config = VisualizerConfig(show_ids=True, show_bboxes=False)
    visualizer = Visualizer(config)
    
    annotated_frames = []
    for i, frame in enumerate(highlight_frames):
        frame_idx = start_frame + i
        if frame_idx in highlight_df.index:
            annotated = visualizer.draw_from_dataframe(
                frame, frame_idx, highlight_df, team_mapping
            )
        else:
            annotated = frame
        annotated_frames.append(annotated)
    
    # Save highlight
    write_video(annotated_frames, "highlight_clip.mp4", fps)
    print(f"Created highlight clip: frames {start_frame}-{end_frame}")


# Example 9: Compare Team Positioning
def example_team_comparison():
    """Compare team positioning over time."""
    from football_tracker import load_tracking_data
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load data
    df, team_mapping, fps = load_tracking_data("output/my_video")
    
    # Separate players by team
    team0_cols = []
    team1_cols = []
    
    for col in df.columns:
        if col == "Ball":
            continue
        player_id = int(col.split("_")[1])
        team_id = team_mapping.get(player_id, -1)
        
        if team_id == 0:
            team0_cols.append(col)
        elif team_id == 1:
            team1_cols.append(col)
    
    # Calculate average x position per team over time
    team0_avg_x = []
    team1_avg_x = []
    
    for frame_idx in df.index:
        row = df.loc[frame_idx]
        
        # Team 0
        team0_x = [row[col][0] for col in team0_cols if not pd.isna(row[col])]
        if team0_x:
            team0_avg_x.append(np.mean(team0_x))
        else:
            team0_avg_x.append(np.nan)
        
        # Team 1
        team1_x = [row[col][0] for col in team1_cols if not pd.isna(row[col])]
        if team1_x:
            team1_avg_x.append(np.mean(team1_x))
        else:
            team1_avg_x.append(np.nan)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(team0_avg_x, label="Team 0", color="red")
    plt.plot(team1_avg_x, label="Team 1", color="blue")
    plt.xlabel("Frame")
    plt.ylabel("Average X Position (pixels)")
    plt.title("Team Positioning Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("team_comparison.png", dpi=300, bbox_inches="tight")
    print("Saved team comparison plot")


# Example 10: Export to Different Format
def example_export_csv():
    """Export tracking data to CSV format."""
    from football_tracker import load_tracking_data
    import pandas as pd
    
    # Load data
    df, team_mapping, fps = load_tracking_data("output/my_video")
    
    # Create CSV-friendly format
    csv_data = []
    
    for frame_idx in df.index:
        row = df.loc[frame_idx]
        
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                continue
            
            if col == "Ball":
                csv_row = {
                    "frame": frame_idx,
                    "time_seconds": frame_idx / fps,
                    "object_type": "Ball",
                    "object_id": "Ball",
                    "team": -1,
                    "x": val[0],
                    "y": val[1]
                }
            else:
                parts = col.split("_")
                obj_type = parts[0]
                obj_id = int(parts[1])
                team_id = team_mapping.get(obj_id, -1)
                
                csv_row = {
                    "frame": frame_idx,
                    "time_seconds": frame_idx / fps,
                    "object_type": obj_type,
                    "object_id": obj_id,
                    "team": team_id,
                    "x": val[0],
                    "y": val[1]
                }
            
            csv_data.append(csv_row)
    
    # Create DataFrame and save
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv("tracking_data.csv", index=False)
    print(f"Exported {len(csv_data)} detections to CSV")


if __name__ == "__main__":
    print("Football Tracker - Example Usage")
    print("=" * 60)
    print("\nAvailable examples:")
    print("1. example_basic() - Simple usage")
    print("2. example_custom_config() - Custom configuration")
    print("3. example_load_results() - Load and re-visualize")
    print("4. example_batch_processing() - Process multiple videos")
    print("5. example_custom_detector() - Custom detector")
    print("6. example_extract_statistics() - Extract statistics")
    print("7. example_filter_players() - Filter specific players")
    print("8. example_create_highlight() - Create highlight clip")
    print("9. example_team_comparison() - Compare teams")
    print("10. example_export_csv() - Export to CSV")
    print("\nRun individual examples by importing and calling them.")

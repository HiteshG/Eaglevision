"""
Utility Module.
Handles video I/O and helper functions.
"""
import cv2
import os
import json
import pandas as pd
from typing import List, Tuple, Dict
import numpy as np


def read_video(video_path: str, fps: int = 24) -> Tuple[List[np.ndarray], int]:
    """
    Read video file and return frames.
    
    Args:
        video_path: Path to video file
        fps: Target frames per second (downsampling if needed)
        
    Returns:
        Tuple of (frames_list, actual_fps)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"Reading video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Native FPS: {native_fps:.2f}, Total frames: {total_frames}")
    
    # Calculate skip rate
    skip = max(1, int(native_fps / fps))
    actual_fps = native_fps / skip
    
    print(f"Sampling at {actual_fps:.2f} FPS (every {skip} frames)")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skip == 0:
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    
    print(f"Read {len(frames)} frames")
    
    return frames, int(actual_fps)


def write_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 24
) -> str:
    """
    Write frames to video file.
    
    Args:
        frames: List of frames in BGR format
        output_path: Path to save video
        fps: Frames per second
        
    Returns:
        Path to saved video
    """
    if not frames:
        raise ValueError("No frames to write")
    
    print(f"Writing video: {output_path}")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    
    print(f"Video saved: {output_path}")
    
    return output_path


def save_tracking_data(
    df: pd.DataFrame,
    team_mapping: Dict[int, int],
    output_dir: str,
    fps: int
):
    """
    Save tracking data to JSON files.
    
    Args:
        df: Processed DataFrame
        team_mapping: Team mapping dictionary
        output_dir: Directory to save files
        fps: Frames per second
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metadata
    metadata = {
        "fps": fps,
        "num_frames": len(df),
        "team_mapping": {str(k): int(v) for k, v in team_mapping.items()}
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")
    
    # Save raw DataFrame
    raw_data_path = os.path.join(output_dir, "raw_data.json")
    df.to_json(raw_data_path, orient="records", indent=2)
    print(f"Saved raw data: {raw_data_path}")
    
    # Save formatted data
    formatted_data = []
    for frame_idx in df.index:
        frame_data = {
            "frame": int(frame_idx),
            "time": f"{frame_idx // fps // 60:02d}:{frame_idx // fps % 60:02d}",
            "detections": []
        }
        
        row = df.loc[frame_idx]
        
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                continue
            
            if col == "Ball":
                detection = {
                    "id": "Ball",
                    "type": "Ball",
                    "x": float(val[0]),
                    "y": float(val[1])
                }
            else:
                parts = col.split("_")
                obj_type = parts[0]
                obj_id = int(parts[1])
                team_id = team_mapping.get(obj_id, -1)
                
                detection = {
                    "id": obj_id,
                    "type": obj_type,
                    "team": team_id,
                    "x": float(val[0]),
                    "y": float(val[1])
                }
            
            frame_data["detections"].append(detection)
        
        formatted_data.append(frame_data)
    
    formatted_path = os.path.join(output_dir, "processed_data.json")
    with open(formatted_path, "w") as f:
        json.dump(formatted_data, f, indent=2)
    print(f"Saved processed data: {formatted_path}")


def load_tracking_data(output_dir: str) -> Tuple[pd.DataFrame, Dict[int, int], int]:
    """
    Load tracking data from saved files.
    
    Args:
        output_dir: Directory containing saved files
        
    Returns:
        Tuple of (DataFrame, team_mapping, fps)
    """
    # Load metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    fps = metadata["fps"]
    team_mapping = {int(k): int(v) for k, v in metadata["team_mapping"].items()}
    
    # Load DataFrame
    raw_data_path = os.path.join(output_dir, "raw_data.json")
    df = pd.read_json(raw_data_path, orient="records")
    df.index = df.index.astype(int)
    
    print(f"Loaded data from {output_dir}")
    print(f"FPS: {fps}, Frames: {len(df)}, Players: {len(team_mapping)}")
    
    return df, team_mapping, fps


def create_output_directory(video_path: str, base_dir: str = "output") -> str:
    """
    Create output directory for a video.
    
    Args:
        video_path: Path to input video
        base_dir: Base output directory
        
    Returns:
        Path to created output directory
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(base_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def print_summary(
    df: pd.DataFrame,
    team_mapping: Dict[int, int],
    fps: int
):
    """
    Print summary of tracking results.
    
    Args:
        df: Processed DataFrame
        team_mapping: Team mapping
        fps: Frames per second
    """
    print("\n" + "="*50)
    print("TRACKING SUMMARY")
    print("="*50)
    
    # Time information
    duration_seconds = len(df) / fps
    print(f"\nVideo Duration: {duration_seconds:.2f} seconds")
    print(f"Frames Processed: {len(df)}")
    print(f"FPS: {fps}")
    
    # Player information
    player_cols = [c for c in df.columns if "Player" in c or "Goalkeeper" in c]
    print(f"\nTotal Players Tracked: {len(player_cols)}")
    
    # Team distribution
    team_counts = {}
    for col in player_cols:
        player_id = int(col.split("_")[1])
        team_id = team_mapping.get(player_id, -1)
        team_counts[team_id] = team_counts.get(team_id, 0) + 1
    
    print(f"\nTeam Distribution:")
    for team_id, count in sorted(team_counts.items()):
        print(f"  Team {team_id}: {count} players")
    
    # Ball tracking
    if "Ball" in df.columns:
        ball_frames = df["Ball"].notna().sum()
        ball_percentage = (ball_frames / len(df)) * 100
        print(f"\nBall Detection:")
        print(f"  Frames with ball: {ball_frames} ({ball_percentage:.1f}%)")
    
    print("="*50 + "\n")


def download_model_weights():
    """
    Download required model weights if not present.
    This is a placeholder - implement based on your hosting solution.
    """
    # Check if weights exist
    weights_dir = "weights"
    required_files = ["yolov8n.pt", "osnet_x0_25_msmt17.pt"]
    
    os.makedirs(weights_dir, exist_ok=True)
    
    for file in required_files:
        file_path = os.path.join(weights_dir, file)
        if not os.path.exists(file_path):
            print(f"Warning: {file} not found in {weights_dir}")
            print(f"Please download from: https://github.com/ultralytics/assets/releases")
    
    return weights_dir

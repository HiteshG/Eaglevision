"""
Visualization Module.
Handles drawing of detection results on video frames.
"""
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .config import VisualizerConfig


class Visualizer:
    """
    Draws tracking results on video frames.
    """
    
    def __init__(self, config: VisualizerConfig):
        """
        Initialize visualizer.
        
        Args:
            config: VisualizerConfig object
        """
        self.config = config
        
    def draw_frame(
        self,
        frame: np.ndarray,
        detections: Dict[str, Dict[int, Dict]],
        team_mapping: Dict[int, int]
    ) -> np.ndarray:
        """
        Draw detections on a frame.
        
        Args:
            frame: Input frame in BGR format
            detections: Detection dictionary for this frame
            team_mapping: Player ID to team ID mapping
            
        Returns:
            Frame with drawn annotations
        """
        annotated_frame = frame.copy()
        
        # Draw players and goalkeepers
        for class_name in ["Player", "Goalkeeper"]:
            if class_name not in detections:
                continue
            
            for obj_id, detection in detections[class_name].items():
                bbox = detection["bbox"]
                bottom_center = detection["bottom_center"]
                
                # Determine color
                if class_name == "Goalkeeper":
                    color = self.config.goalkeeper_color
                else:
                    team_id = team_mapping.get(obj_id, 0)
                    color = self.config.team_colors.get(team_id, (255, 255, 255))
                
                # Draw bounding box
                if self.config.show_bboxes:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ellipse at bottom center
                x, y = bottom_center
                cv2.ellipse(
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
                    cv2.putText(
                        annotated_frame,
                        str(obj_id),
                        (x - 10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
        
        # Draw ball
        if self.config.show_ball and "Ball" in detections:
            for ball_detection in detections["Ball"].values():
                bottom_center = ball_detection["bottom_center"]
                x, y = bottom_center
                
                # Draw triangle marker for ball
                bottom_point = (x, y - 20)
                top_left = (x - 5, y - 30)
                top_right = (x + 5, y - 30)
                pts = np.array([bottom_point, top_left, top_right])
                cv2.drawContours(
                    annotated_frame,
                    [pts],
                    0,
                    self.config.ball_color,
                    -1
                )
        
        return annotated_frame
    
    def draw_from_dataframe(
        self,
        frame: np.ndarray,
        frame_idx: int,
        df: pd.DataFrame,
        team_mapping: Dict[int, int]
    ) -> np.ndarray:
        """
        Draw from processed DataFrame.
        
        Args:
            frame: Input frame
            frame_idx: Frame index
            df: Processed DataFrame
            team_mapping: Team mapping
            
        Returns:
            Annotated frame
        """
        if frame_idx not in df.index:
            return frame
        
        annotated_frame = frame.copy()
        row = df.loc[frame_idx]
        
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                continue
            
            x, y = int(val[0]), int(val[1])
            
            if col == "Ball":
                # Draw ball
                if self.config.show_ball:
                    bottom_point = (x, y - 20)
                    top_left = (x - 5, y - 30)
                    top_right = (x + 5, y - 30)
                    pts = np.array([bottom_point, top_left, top_right])
                    cv2.drawContours(
                        annotated_frame,
                        [pts],
                        0,
                        self.config.ball_color,
                        -1
                    )
            else:
                # Parse object type and ID
                parts = col.split("_")
                obj_type = parts[0]
                obj_id = int(parts[1])
                
                # Determine color
                if obj_type == "Goalkeeper":
                    color = self.config.goalkeeper_color
                else:
                    team_id = team_mapping.get(obj_id, 0)
                    color = self.config.team_colors.get(team_id, (255, 255, 255))
                
                # Draw ellipse
                cv2.ellipse(
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
                    cv2.putText(
                        annotated_frame,
                        str(obj_id),
                        (x - 10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
        
        return annotated_frame
    
    def create_annotated_video(
        self,
        frames: List[np.ndarray],
        df: pd.DataFrame,
        team_mapping: Dict[int, int],
        output_path: str,
        fps: int
    ) -> str:
        """
        Create annotated video from processed data.
        
        Args:
            frames: List of original frames
            df: Processed DataFrame
            team_mapping: Team mapping
            output_path: Path to save video
            fps: Frames per second
            
        Returns:
            Path to saved video
        """
        print(f"Creating annotated video: {output_path}")
        
        annotated_frames = []
        
        for frame_idx, frame in enumerate(frames):
            if frame_idx in df.index:
                annotated_frame = self.draw_from_dataframe(
                    frame, frame_idx, df, team_mapping
                )
            else:
                annotated_frame = frame.copy()
            
            annotated_frames.append(annotated_frame)
        
        # Write video
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in annotated_frames:
            out.write(frame)
        
        out.release()
        print(f"Saved annotated video: {output_path}")
        
        return output_path
    
    def draw_info_panel(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: int,
        team_stats: Dict[int, int]
    ) -> np.ndarray:
        """
        Draw information panel on frame.
        
        Args:
            frame: Input frame
            frame_idx: Current frame index
            fps: Frames per second
            team_stats: Player count per team
            
        Returns:
            Frame with info panel
        """
        # Calculate time
        seconds = frame_idx // fps
        time_str = f"{seconds // 60:02d}:{seconds % 60:02d}"
        
        # Draw semi-transparent panel
        panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (300, 10 + panel_height),
            (0, 0, 0),
            -1
        )
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw text
        y_offset = 35
        cv2.putText(
            frame,
            f"Time: {time_str}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        y_offset += 30
        team0_count = team_stats.get(0, 0)
        team1_count = team_stats.get(1, 0)
        cv2.putText(
            frame,
            f"Team 0: {team0_count} | Team 1: {team1_count}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return frame

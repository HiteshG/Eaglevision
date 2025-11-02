"""
Main script for football player tracking.
Orchestrates detection, tracking, team assignment (with memory), and visualization.
"""
import os
import sys
import argparse
from typing import List, Dict
import numpy as np

from processor import DataProcessor
from visualizer import Visualizer
from tracker import ObjectTracker
from detector import ObjectDetector
from team_assigner import create_team_assigner, resolve_goalkeepers_team_id
from config import MainConfig
from utils import read_video, create_output_directory, save_tracking_data, print_summary


class FootballTracker:
    """
    Main tracking pipeline with team memory support.
    """
    
    def __init__(self, config: MainConfig = None):
        """
        Initialize the tracking pipeline.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or MainConfig()
        
        # Initialize components
        print("\n" + "="*50)
        print("INITIALIZING FOOTBALL TRACKER")
        print("="*50 + "\n")
        
        self.detector = ObjectDetector(self.config.detector)
        self.tracker = ObjectTracker(
            self.config.tracker,
            memory_decay_frames=self.config.team_assigner.memory_decay_frames
        )
        self.team_assigner = create_team_assigner(self.config.team_assigner)
        self.processor = None  # Will be initialized with FPS
        self.visualizer = Visualizer(self.config.visualizer)
        
        print("\nAll components initialized successfully!\n")
    
    def process_video(self, video_path: str, output_dir: str = None) -> str:
        """
        Process a video file and generate tracking results.
        
        Args:
            video_path: Path to input video
            output_dir: Output directory (auto-generated if None)
            
        Returns:
            Path to output directory
        """
        # Create output directory
        if output_dir is None:
            output_dir = create_output_directory(
                video_path,
                self.config.output_dir
            )
        
        print("\n" + "="*50)
        print(f"PROCESSING VIDEO: {os.path.basename(video_path)}")
        print("="*50 + "\n")
        
        # Step 1: Read video
        print("Step 1/5: Reading video...")
        frames, fps = read_video(video_path, self.config.fps)
        
        if not frames:
            raise ValueError("No frames read from video")
        
        # Initialize processor with actual FPS
        self.processor = DataProcessor(self.config.processor, fps)
        
        # Step 2: Detection and Tracking
        print("\nStep 2/5: Detecting and tracking objects...")
        detections_per_frame = self._detect_and_track(frames)
        
        # Step 3: Team Assignment with Memory
        print("\nStep 3/5: Assigning teams with memory...")
        team_mapping = self._assign_teams_with_memory(frames, detections_per_frame)
        
        # Step 4: Assign Goalkeepers to Teams
        print("\nStep 4/5: Assigning goalkeepers to teams...")
        team_mapping = self._assign_goalkeepers(detections_per_frame, team_mapping)
        
        # Step 5: Data Processing
        print("\nStep 5/5: Processing tracking data...")
        df, team_mapping = self.processor.process(
            detections_per_frame,
            team_mapping
        )
        
        # Step 6: Save Results
        print("\nStep 6/6: Saving results...")
        
        # Save tracking data
        save_tracking_data(df, team_mapping, output_dir, fps)
        
        # Create annotated video
        annotated_path = os.path.join(output_dir, "annotated.mp4")
        self.visualizer.create_annotated_video(
            frames,
            df,
            team_mapping,
            annotated_path,
            fps
        )
        
        # Print summary
        print_summary(df, team_mapping, fps)
        
        print(f"\n✓ Processing complete! Results saved to: {output_dir}\n")
        
        return output_dir
    
    def _detect_and_track(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Run detection and tracking on all frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of detection dictionaries per frame
        """
        detections_per_frame = []
        
        print(f"Processing {len(frames)} frames...")
        
        for i, frame in enumerate(frames):
            if i % 100 == 0:
                print(f"  Frame {i}/{len(frames)}")
            
            # Detect objects
            boxes, confidences, class_labels = self.detector.detect(frame)
            
            # Prepare for tracker
            detection_array = self.detector.get_detection_array(
                boxes,
                confidences,
                class_labels
            )
            
            # Update tracker
            tracks = self.tracker.update(detection_array, frame)
            
            # Organize tracked objects
            frame_detections = self.tracker.organize_tracks(
                tracks,
                self.detector.CLASS_NAMES,
                self.config.detector.confidence_threshold,
                frame.shape[:2]
            )
            
            # Add ball detections (not tracked, just detected)
            ball_detections = self.detector.filter_detections(
                boxes,
                confidences,
                class_labels,
                frame.shape[:2]
            ).get("Ball", {})
            
            frame_detections["Ball"] = ball_detections
            
            detections_per_frame.append(frame_detections)
        
        print(f"  Completed all {len(frames)} frames")
        
        return detections_per_frame
    
    def _assign_teams_with_memory(
        self,
        frames: List[np.ndarray],
        detections_per_frame: List[Dict]
    ) -> Dict[int, int]:
        """
        Assign teams using the configured method and tracker memory.
        
        Args:
            frames: List of video frames
            detections_per_frame: List of detection dictionaries
            
        Returns:
            Dictionary mapping player_id -> team_id
        """
        # First, run the team assignment algorithm on all frames
        initial_team_mapping = self.team_assigner.assign_teams(
            frames,
            detections_per_frame
        )
        
        # Update tracker memory with initial assignments
        for player_id, team_id in initial_team_mapping.items():
            self.tracker.update_team_memory(player_id, team_id)
        
        # Now refine with memory-based consensus
        # For each player, check if we have memory from tracker
        final_team_mapping = {}
        
        for player_id in initial_team_mapping.keys():
            # Check tracker memory first
            memory_team = self.tracker.get_team_from_memory(player_id)
            
            if memory_team is not None:
                # Use memory if available
                final_team_mapping[player_id] = memory_team
            else:
                # Use initial assignment
                final_team_mapping[player_id] = initial_team_mapping[player_id]
                # Update memory
                self.tracker.update_team_memory(player_id, initial_team_mapping[player_id])
        
        return final_team_mapping
    
    def _assign_goalkeepers(
        self,
        detections_per_frame: List[Dict],
        player_team_mapping: Dict[int, int]
    ) -> Dict[int, int]:
        """
        Assign goalkeepers to teams using distance-weighted voting.
        
        Args:
            detections_per_frame: List of detection dictionaries
            player_team_mapping: Existing player team assignments
            
        Returns:
            Updated team mapping including goalkeepers
        """
        # Collect all goalkeeper and player positions across frames
        goalkeeper_positions = {}  # {gk_id: [(x, y), ...]}
        player_positions = {}  # {player_id: [(x, y), ...]}
        
        for detections in detections_per_frame:
            # Collect player positions
            if "Player" in detections:
                for player_id, detection in detections["Player"].items():
                    pos = detection["bottom_center"]
                    if player_id not in player_positions:
                        player_positions[player_id] = []
                    player_positions[player_id].append(pos)
            
            # Collect goalkeeper positions
            if "Goalkeeper" in detections:
                for gk_id, detection in detections["Goalkeeper"].items():
                    pos = detection["bottom_center"]
                    if gk_id not in goalkeeper_positions:
                        goalkeeper_positions[gk_id] = []
                    goalkeeper_positions[gk_id].append(pos)
        
        if not goalkeeper_positions:
            return player_team_mapping
        
        # Average positions for each player and goalkeeper
        avg_player_positions = {}
        for player_id, positions in player_positions.items():
            if player_id in player_team_mapping:
                avg_player_positions[player_id] = np.mean(positions, axis=0)
        
        avg_gk_positions = {}
        for gk_id, positions in goalkeeper_positions.items():
            avg_gk_positions[gk_id] = np.mean(positions, axis=0)
        
        if not avg_player_positions or not avg_gk_positions:
            return player_team_mapping
        
        # Prepare arrays for distance calculation
        player_ids = list(avg_player_positions.keys())
        players_xy = np.array([avg_player_positions[pid] for pid in player_ids])
        players_team_id = np.array([player_team_mapping[pid] for pid in player_ids])
        
        gk_ids = list(avg_gk_positions.keys())
        goalkeepers_xy = np.array([avg_gk_positions[gid] for gid in gk_ids])
        
        # Assign goalkeeper teams
        gk_teams = resolve_goalkeepers_team_id(
            players_xy,
            players_team_id,
            goalkeepers_xy
        )
        
        # Update team mapping
        goalkeeper_team_mapping = {
            gk_ids[i]: gk_teams[i]
            for i in range(len(gk_ids))
        }
        
        # Update tracker memory for goalkeepers
        for gk_id, team_id in goalkeeper_team_mapping.items():
            self.tracker.update_team_memory(gk_id, team_id)
        
        # Combine with player mapping
        combined_mapping = {**player_team_mapping, **goalkeeper_team_mapping}
        
        print(f"Assigned {len(goalkeeper_team_mapping)} goalkeepers to teams")
        for team_id in [0, 1]:
            count = sum(1 for t in goalkeeper_team_mapping.values() if t == team_id)
            if count > 0:
                print(f"  Team {team_id}: {count} goalkeeper(s)")
        
        return combined_mapping


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Football Player Detection, Tracking, and Team Assignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Target frames per second for processing"
    )
    
    parser.add_argument(
        "--detector-conf",
        type=float,
        default=0.35,
        help="Detector confidence threshold"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model weights"
    )
    
    parser.add_argument(
        "--team-method",
        type=str,
        choices=["color", "embedding"],
        default="color",
        help="Team assignment method: color-based or embedding-based"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )
    
    parser.add_argument(
        "--show-bboxes",
        action="store_true",
        help="Show bounding boxes in annotated video"
    )
    
    parser.add_argument(
        "--memory-decay",
        type=int,
        default=150,
        help="Frames before forgetting team assignment (default: 150)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = MainConfig()
    config.fps = args.fps
    config.detector.model_path = args.model
    config.detector.confidence_threshold = args.detector_conf
    config.visualizer.show_bboxes = args.show_bboxes
    
    # Team assignment configuration
    config.team_assigner.team_method = args.team_method
    config.team_assigner.memory_decay_frames = args.memory_decay
    
    if args.no_gpu:
        config.detector.device = "cpu"
        config.tracker.device = "cpu"
        config.team_assigner.device = "cpu"
    
    # Create tracker and process video
    try:
        tracker = FootballTracker(config)
        output_dir = tracker.process_video(args.video, args.output_dir)
        
        print("\n" + "="*50)
        print("SUCCESS!")
        print("="*50)
        print(f"\nResults saved to: {output_dir}")
        print("\nOutput files:")
        print("  - annotated.mp4       : Video with tracking visualization")
        print("  - raw_data.json       : Raw tracking data")
        print("  - processed_data.json : Processed tracking data")
        print("  - metadata.json       : Video and team metadata")
        print("\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
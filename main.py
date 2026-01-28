"""
Main script for football player tracking.
Orchestrates detection, tracking, team assignment, and visualization.
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
from team_assigner import TeamAssigner
from config import MainConfig
from utils import read_video, create_output_directory, save_tracking_data, print_summary

class FootballTracker:
    """
    Main tracking pipeline that coordinates all components.
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
        self.tracker = ObjectTracker(self.config.tracker)
        self.team_assigner = TeamAssigner(self.config.team_assigner)
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
        
        # Step 3: Team Assignment
        print("\nStep 3/5: Assigning teams...")
        team_mapping = self.team_assigner.assign_teams(
            frames,
            detections_per_frame
        )
        
        # Step 4: Data Processing
        print("\nStep 4/5: Processing tracking data...")
        df, team_mapping = self.processor.process(
            detections_per_frame,
            team_mapping
        )
        
        # Step 5: Save Results
        print("\nStep 5/5: Saving results...")
        
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
        "--cmc-method",
        type=str,
        default="orb",
        choices=["orb", "ecc", "sof", "sparseOptFlow", "none"],
        help="Camera Motion Compensation method (orb, ecc, sof, sparseOptFlow, none)"
    )

    parser.add_argument(
        "--track-buffer",
        type=int,
        default=30,
        help="Frames to keep lost tracks (lower = fewer ghost tracks)"
    )

    parser.add_argument(
        "--new-track-thresh",
        type=float,
        default=0.6,
        help="Confidence threshold for creating new tracks (higher = fewer false tracks)"
    )

    args = parser.parse_args()
    
    # Create configuration
    config = MainConfig()
    config.fps = args.fps
    config.detector.model_path = args.model
    config.detector.confidence_threshold = args.detector_conf
    config.visualizer.show_bboxes = args.show_bboxes

    # Camera Motion Compensation settings
    config.tracker.cmc_method = args.cmc_method if args.cmc_method != "none" else None
    config.tracker.track_buffer = args.track_buffer
    config.tracker.new_track_thresh = args.new_track_thresh

    if args.no_gpu:
        config.detector.device = "cpu"
        config.tracker.device = "cpu"
    
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

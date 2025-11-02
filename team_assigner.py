"""
Team Assignment Module.
Assigns players to teams based on jersey color clustering.
Designed to be easily swappable with other team assignment methods (e.g., deep learning).
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from typing import Dict, List, Tuple
from .config import TeamAssignerConfig


class TeamAssigner:
    """
    Color-based team assignment using KMeans clustering.
    Can be replaced with deep learning models or other methods.
    """
    
    def __init__(self, config: TeamAssignerConfig):
        """
        Initialize team assigner with configuration.
        
        Args:
            config: TeamAssignerConfig object
        """
        self.config = config
        self.color_ranges = config.color_ranges
        
    def assign_teams(
        self,
        frames: List[np.ndarray],
        detections_per_frame: List[Dict]
    ) -> Dict[int, int]:
        """
        Assign teams to all detected players.
        
        Args:
            frames: List of video frames in BGR format
            detections_per_frame: List of detection dictionaries per frame
            
        Returns:
            Dictionary mapping player_id -> team_id (0 or 1)
        """
        print("Assigning teams based on jersey colors...")
        
        # First pass: collect color frequencies for each player
        player_color_counts = {}
        
        for frame, detections in zip(frames, detections_per_frame):
            if "Player" not in detections:
                continue
            
            players = detections["Player"]
            all_bboxes = [item["bbox"] for item in players.values()]
            
            for player_id, detection in players.items():
                bbox = detection["bbox"]
                x1, y1, x2, y2 = bbox
                
                # Calculate overlap with other detections
                overlap_ratio = self._calculate_max_overlap_ratio(
                    bbox, all_bboxes
                )
                
                # Skip heavily overlapped detections
                if overlap_ratio > self.config.overlap_threshold:
                    continue
                
                # Extract player crop
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Detect dominant colors
                color_counts = self._detect_colors(crop)
                
                # Accumulate color counts for this player
                if player_id not in player_color_counts:
                    player_color_counts[player_id] = {}
                
                for color, count in color_counts:
                    if color not in player_color_counts[player_id]:
                        player_color_counts[player_id][color] = 0
                    # Weight by (1 - overlap_ratio) to reduce noise
                    player_color_counts[player_id][color] += count * (1 - overlap_ratio)
        
        # Determine dominant color for each player
        player_dominant_colors = {}
        for player_id, color_counts in player_color_counts.items():
            if color_counts:
                dominant_color = max(color_counts, key=color_counts.get)
                player_dominant_colors[player_id] = dominant_color
        
        # Find the two most common colors (representing the two teams)
        all_colors = list(player_dominant_colors.values())
        if len(all_colors) < 2:
            print("Warning: Not enough players detected for team assignment")
            return {pid: 0 for pid in player_dominant_colors.keys()}
        
        color_frequency = Counter(all_colors)
        most_common_colors = color_frequency.most_common(2)
        team_colors = [color for color, _ in most_common_colors]
        
        # Create color to team mapping
        color_to_team = {color: i for i, color in enumerate(team_colors)}
        
        # Second pass: assign teams, handling outliers
        team_mapping = {}
        for player_id, dominant_color in player_dominant_colors.items():
            if dominant_color in color_to_team:
                # Player's color matches one of the two main team colors
                team_mapping[player_id] = color_to_team[dominant_color]
            else:
                # Outlier: assign to team based on secondary color preference
                color_counts = player_color_counts[player_id]
                # Filter to only team colors
                team_color_counts = [
                    (color, count) 
                    for color, count in color_counts.items() 
                    if color in color_to_team
                ]
                
                if team_color_counts:
                    # Assign to team with highest secondary color count
                    best_color = max(team_color_counts, key=lambda x: x[1])[0]
                    team_mapping[player_id] = color_to_team[best_color]
                else:
                    # Default to team 0 if no team colors found
                    print(f"Warning: Could not determine team for player {player_id}")
                    team_mapping[player_id] = 0
        
        print(f"Assigned {len(team_mapping)} players to teams")
        print(f"Team 0: {sum(1 for t in team_mapping.values() if t == 0)} players")
        print(f"Team 1: {sum(1 for t in team_mapping.values() if t == 1)} players")
        
        return team_mapping
    
    def _calculate_max_overlap_ratio(
        self,
        bbox: List[int],
        all_bboxes: List[List[int]]
    ) -> float:
        """
        Calculate maximum overlap ratio with other bounding boxes.
        
        Args:
            bbox: [x1, y1, x2, y2]
            all_bboxes: List of all bounding boxes
            
        Returns:
            Maximum overlap ratio (0 to 1)
        """
        x1, y1, x2, y2 = bbox
        curr_area = (x2 - x1) * (y2 - y1)
        
        if curr_area == 0:
            return 0.0
        
        max_overlap = 0.0
        
        for other_bbox in all_bboxes:
            if other_bbox == bbox:
                continue
            
            x1_, y1_, x2_, y2_ = other_bbox
            
            # Calculate intersection
            x_overlap = max(0, min(x2, x2_) - max(x1, x1_))
            y_overlap = max(0, min(y2, y2_) - max(y1, y1_))
            overlap_area = x_overlap * y_overlap
            
            overlap_ratio = overlap_area / curr_area
            max_overlap = max(max_overlap, overlap_ratio)
        
        return max_overlap
    
    def _detect_colors(self, image: np.ndarray) -> List[Tuple[str, int]]:
        """
        Detect colors in an image crop using HSV color ranges.
        
        Args:
            image: BGR image crop
            
        Returns:
            List of (color_name, pixel_count) tuples, sorted by count
        """
        if image.size == 0:
            return []
        
        # Convert to RGB for KMeans clustering
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use KMeans to segment player from background
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=3, init="k-means++")
        kmeans.fit(rgb_image.reshape(-1, 3))
        labels = kmeans.labels_.reshape(image.shape[:2])
        
        # Determine which cluster is background (most common in corners)
        corners = [
            labels[0, 0], labels[0, -1],
            labels[-1, 0], labels[-1, -1]
        ]
        background_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 if background_cluster == 0 else 0
        
        # Create mask for player region
        player_mask = (labels == player_cluster).astype(np.uint8) * 255
        
        # Convert to HSV for color detection
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=player_mask)
        
        # Count pixels in each color range
        color_counts = {}
        
        for color_name, (lower, upper) in self.color_ranges.items():
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)
            
            # Create color mask
            color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            color_mask = cv2.bitwise_and(color_mask, player_mask)
            
            # Count non-zero pixels
            count = cv2.countNonZero(color_mask)
            
            if count > 0:
                color_counts[color_name] = count
        
        # Combine red and red2
        if "red2" in color_counts:
            color_counts["red"] = color_counts.get("red", 0) + color_counts.pop("red2")
        
        # Sort by count
        color_counts = sorted(
            color_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return color_counts
    
    def __repr__(self):
        return f"TeamAssigner(n_clusters={self.config.n_clusters})"


class DeepLearningTeamAssigner:
    """
    Placeholder for deep learning-based team assignment.
    Can be implemented using models like ResNet, EfficientNet, etc.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        raise NotImplementedError(
            "Deep learning team assignment not yet implemented. "
            "Use TeamAssigner for color-based assignment."
        )
    
    def assign_teams(self, frames, detections_per_frame):
        """Assign teams using deep learning model."""
        pass

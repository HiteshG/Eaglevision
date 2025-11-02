"""
Data Processor Module.
Handles interpolation, smoothing, and ID merging for tracking data.
"""
import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple
from config import ProcessorConfig


class DataProcessor:
    """
    Processes tracking data including interpolation and ID merging.
    """
    
    def __init__(self, config: ProcessorConfig, fps: int):
        """
        Initialize data processor.
        
        Args:
            config: ProcessorConfig object
            fps: Frames per second of the video
        """
        self.config = config
        self.fps = fps
        
    def process(
        self,
        detections_per_frame: List[Dict],
        team_mapping: Dict[int, int]
    ) -> Tuple[pd.DataFrame, Dict[int, int]]:
        """
        Process detection data into a clean DataFrame.
        
        Args:
            detections_per_frame: List of detection dictionaries per frame
            team_mapping: Dictionary mapping player_id -> team_id
            
        Returns:
            Tuple of (processed_dataframe, updated_team_mapping)
        """
        print("Processing tracking data...")
        
        # Create initial DataFrame
        df = self._create_dataframe(detections_per_frame)
        
        if df.empty:
            print("Warning: No data to process")
            return df, team_mapping
        
        # Interpolate missing ball positions
        if "Ball" in df.columns:
            df = self._interpolate_column(df, "Ball", fill=True)
        
        # Merge fragmented IDs (same player with different IDs)
        df, team_mapping = self._merge_fragmented_ids(df, team_mapping)
        
        # Interpolate player positions
        for col in df.columns:
            if col == "Ball":
                continue
            df = self._interpolate_column(df, col, fill=False)
            
            # Optional smoothing
            if self.config.smooth:
                df = self._smooth_column(df, col)
        
        print(f"Processed data: {len(df)} frames, {len(df.columns)} tracked objects")
        return df, team_mapping
    
    def _create_dataframe(self, detections_per_frame: List[Dict]) -> pd.DataFrame:
        """
        Create DataFrame from detection data.
        
        Args:
            detections_per_frame: List of detection dictionaries
            
        Returns:
            DataFrame with columns for each tracked object
        """
        data = {}
        
        for frame_idx, detections in enumerate(detections_per_frame):
            frame_data = {}
            
            has_person = False
            
            # Add player and goalkeeper detections
            for class_name in ["Player", "Goalkeeper"]:
                if class_name not in detections:
                    continue
                
                for obj_id, detection in detections[class_name].items():
                    col_name = f"{class_name}_{obj_id}"
                    bottom_center = detection["bottom_center"]
                    frame_data[col_name] = tuple(bottom_center)
                    has_person = True
            
            # Add ball detection (take highest confidence if multiple)
            if "Ball" in detections and len(detections["Ball"]) > 0:
                ball_detections = detections["Ball"]
                # Sort by confidence and take the best
                best_ball = max(
                    ball_detections.values(),
                    key=lambda x: x["confidence"]
                )
                frame_data["Ball"] = tuple(best_ball["bottom_center"])
            
            # Only include frames with at least one person detection
            if has_person:
                data[frame_idx] = frame_data
        
        # Create DataFrame
        df = pd.DataFrame(data).T
        
        # Remove columns with less than 1% non-null values
        if len(df) > 0:
            threshold = 0.01 * len(df)
            df = df.loc[:, df.notna().sum() >= threshold]
        
        return df
    
    def _interpolate_column(
        self,
        df: pd.DataFrame,
        col_name: str,
        fill: bool = False
    ) -> pd.DataFrame:
        """
        Interpolate missing values in a column.
        
        Args:
            df: Input DataFrame
            col_name: Column name to interpolate
            fill: If True, fill all NaN values; if False, only interpolate inside
            
        Returns:
            DataFrame with interpolated values
        """
        if col_name not in df.columns:
            return df
        
        s = df[col_name]
        
        # Extract x and y coordinates
        x = s.apply(lambda v: v[0] if isinstance(v, (list, tuple)) else np.nan)
        y = s.apply(lambda v: v[1] if isinstance(v, (list, tuple)) else np.nan)
        
        # Interpolate
        if fill:
            x = x.interpolate(method="linear").bfill().ffill()
            y = y.interpolate(method="linear").bfill().ffill()
        else:
            x = x.interpolate(method="linear", limit_area="inside")
            y = y.interpolate(method="linear", limit_area="inside")
        
        # Combine back
        combined = pd.Series(
            [(xi, yi) if not (math.isnan(xi) or math.isnan(yi)) else np.nan
             for xi, yi in zip(x, y)],
            index=s.index
        )
        
        df[col_name] = combined
        return df
    
    def _smooth_column(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        Smooth trajectories by interpolating every other point.
        
        Args:
            df: Input DataFrame
            col_name: Column name to smooth
            
        Returns:
            DataFrame with smoothed values
        """
        if col_name not in df.columns:
            return df
        
        s = df[col_name]
        
        # Extract coordinates
        x = s.apply(lambda v: v[0] if isinstance(v, (list, tuple)) else np.nan)
        y = s.apply(lambda v: v[1] if isinstance(v, (list, tuple)) else np.nan)
        
        # Set every other value to NaN
        x.iloc[::2] = np.nan
        y.iloc[::2] = np.nan
        
        # Interpolate
        x = x.interpolate(method="linear", limit_area="inside")
        y = y.interpolate(method="linear", limit_area="inside")
        
        # Combine
        combined = pd.Series(
            [(xi, yi) if not (math.isnan(xi) or math.isnan(yi)) else np.nan
             for xi, yi in zip(x, y)],
            index=s.index
        )
        
        df[col_name] = combined
        return df
    
    def _merge_fragmented_ids(
        self,
        df: pd.DataFrame,
        team_mapping: Dict[int, int]
    ) -> Tuple[pd.DataFrame, Dict[int, int]]:
        """
        Merge fragmented IDs (same player tracked with different IDs).
        
        Args:
            df: Input DataFrame
            team_mapping: Current team mapping
            
        Returns:
            Tuple of (updated DataFrame, updated team_mapping)
        """
        # Merge goalkeepers with players if same ID
        goalkeeper_cols = [c for c in df.columns if "Goalkeeper" in c]
        for col in goalkeeper_cols:
            player_id = col.split("_")[1]
            player_col = f"Player_{player_id}"
            
            if player_col in df.columns:
                # Merge goalkeeper into player
                df[col] = df[player_col].combine_first(df[col])
                df.drop(columns=[player_col], inplace=True)
        
        # Find columns to merge
        temporal_threshold = int(self.fps * self.config.temporal_threshold_seconds)
        
        player_cols = [c for c in df.columns if "Player" in c]
        goalkeeper_cols = [c for c in df.columns if "Goalkeeper" in c]
        
        to_merge = []
        
        # Check each pair of columns
        for col_type, cols in [("Player", player_cols), ("Goalkeeper", goalkeeper_cols)]:
            for i, col1 in enumerate(cols):
                for col2 in cols[i+1:]:
                    if self._should_merge(df, col1, col2, temporal_threshold, team_mapping):
                        to_merge.append((col1, col2))
        
        # Perform merges
        merged_cols = {}
        
        for col1, col2 in to_merge:
            root1 = self._find_root(col1, merged_cols)
            root2 = self._find_root(col2, merged_cols)
            
            if root1 != root2:
                # Merge root2 into root1
                df[root1] = df[root1].combine_first(df[root2])
                df.drop(columns=[root2], inplace=True)
                merged_cols[root2] = root1
                
                # Update team mapping
                id1 = int(root1.split("_")[1])
                id2 = int(root2.split("_")[1])
                if id2 in team_mapping:
                    team_mapping[id1] = team_mapping[id2]
        
        return df, team_mapping
    
    def _should_merge(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str,
        temporal_threshold: int,
        team_mapping: Dict[int, int]
    ) -> bool:
        """
        Determine if two columns should be merged.
        
        Args:
            df: DataFrame
            col1, col2: Column names
            temporal_threshold: Maximum frame gap
            team_mapping: Team assignments
            
        Returns:
            True if columns should be merged
        """
        # Get valid indices
        idx1_first = df[col1].first_valid_index()
        idx1_last = df[col1].last_valid_index()
        idx2_first = df[col2].first_valid_index()
        idx2_last = df[col2].last_valid_index()
        
        if any(x is None for x in [idx1_first, idx1_last, idx2_first, idx2_last]):
            return False
        
        # Check for overlap
        if (idx1_last >= idx2_first or idx2_last >= idx1_first):
            return False
        
        # Determine order
        if idx2_first < idx1_first:
            first_idx = idx1_first
            last_idx = idx2_last
            first_val = df[col1].loc[first_idx]
            last_val = df[col2].loc[last_idx]
        else:
            first_idx = idx2_first
            last_idx = idx1_last
            first_val = df[col2].loc[first_idx]
            last_val = df[col1].loc[last_idx]
        
        # Condition 1: Temporal proximity
        if abs(last_idx - first_idx) > temporal_threshold:
            return False
        
        # Condition 2: Spatial continuity
        threshold = abs(last_idx - first_idx) * self.config.spatial_threshold_per_frame
        distance = np.linalg.norm(np.array(last_val) - np.array(first_val))
        if distance > threshold:
            return False
        
        # Condition 3: Team consistency
        id1 = int(col1.split("_")[1])
        id2 = int(col2.split("_")[1])
        
        if id1 in team_mapping and id2 in team_mapping:
            if team_mapping[id1] != team_mapping[id2]:
                return False
        
        return True
    
    def _find_root(self, col: str, merged_cols: Dict[str, str]) -> str:
        """Find root column in merge chain."""
        while col in merged_cols:
            col = merged_cols[col]
        return col
    
    def format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format DataFrame for output.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Formatted DataFrame
        """
        output = []
        
        for frame_idx in df.index:
            frame_data = {
                "frame": int(frame_idx),
                "time": f"{frame_idx // self.fps // 60:02d}:{frame_idx // self.fps % 60:02d}",
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
                    
                    detection = {
                        "id": obj_id,
                        "type": obj_type,
                        "x": float(val[0]),
                        "y": float(val[1])
                    }
                
                frame_data["detections"].append(detection)
            
            output.append(frame_data)
        
        return pd.DataFrame(output)

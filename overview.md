# Integration Guide - Team Memory & Embedding System

## Architecture Overview

This document explains how the team memory system integrates with both color-based and embedding-based team assignment methods.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Football Tracker                        │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Detector   │    │   Tracker    │    │Team Assigner │
│   (YOLO)     │───▶│  (BoTSORT)   │◀──▶│(Color/Embed) │
└──────────────┘    └──────────────┘    └──────────────┘
                            │
                    ┌───────┴────────┐
                    ▼                ▼
            ┌──────────────┐  ┌──────────────┐
            │Team Memory   │  │  Goalkeeper  │
            │  Manager     │  │  Assignment  │
            └──────────────┘  └──────────────┘
```

---

## Data Flow

### Frame-by-Frame Processing

```
Frame N
  │
  ├─▶ YOLO Detection
  │     │ Outputs: boxes, confidences, classes
  │     ▼
  ├─▶ BoTSORT Tracking
  │     │ Outputs: tracks with IDs
  │     │ Updates: track_id assignments
  │     ▼
  ├─▶ Team Memory Lookup
  │     │ Check: tracker.get_team_from_memory(track_id)
  │     │ Result: team_id (if in memory) or None
  │     │
  │     ├─▶ If in memory and not expired:
  │     │     └─▶ Use cached team assignment
  │     │
  │     └─▶ If not in memory or expired:
  │           └─▶ Will be assigned in bulk later
  │
  └─▶ Store detections for batch assignment
```

### Batch Team Assignment (After All Frames)

```
All Frames Processed
  │
  ├─▶ Team Assignment Method
  │     │
  │     ├─▶ Color-Based:
  │     │     │ 1. Collect player crops
  │     │     │ 2. Segment player from background (KMeans)
  │     │     │ 3. Extract HSV colors
  │     │     │ 4. Find 2 dominant colors = 2 teams
  │     │     │ 5. Assign players to teams
  │     │     └─▶ team_mapping
  │     │
  │     └─▶ Embedding-Based:
  │           │ 1. Collect player crops (stride sampling)
  │           │ 2. Extract SigLip embeddings
  │           │ 3. Average embeddings per player
  │           │ 4. UMAP dimensionality reduction
  │           │ 5. KMeans++ clustering
  │           └─▶ team_mapping
  │
  ├─▶ Update Tracker Memory
  │     │ For each player_id, team_id in mapping:
  │     └─▶ tracker.update_team_memory(player_id, team_id)
  │
  ├─▶ Goalkeeper Assignment
  │     │ 1. Collect goalkeeper positions
  │     │ 2. Collect player positions + teams
  │     │ 3. Distance-weighted voting (k=5 nearest)
  │     │ 4. Assign goalkeeper to majority team
  │     └─▶ Update team_mapping with goalkeepers
  │
  └─▶ Final team_mapping ready for visualization
```

---

## Team Memory System Details

### Memory Structure

```python
tracker.team_memory = {
    track_id: (team_id, last_seen_frame),
    # Example:
    5: (0, 1234),  # Player 5, Team 0, last seen at frame 1234
    7: (1, 1235),  # Player 7, Team 1, last seen at frame 1235
    12: (0, 1100), # Player 12, Team 0, last seen at frame 1100
}
```

### Memory Lifecycle

```
Track ID 5 appears at frame 100
  │
  ├─▶ Memory check: Not found
  │     └─▶ Will be assigned later in batch
  │
  └─▶ After batch assignment: Team 0
        └─▶ Memory updated: {5: (0, 100)}


Track ID 5 reappears at frame 150
  │
  ├─▶ Memory check: Found (Team 0, frame 100)
  │     │ Age: 150 - 100 = 50 frames
  │     │ Decay threshold: 150 frames
  │     │ 50 < 150 ✓ Valid
  │     └─▶ Use Team 0 from memory
  │
  └─▶ Memory updated: {5: (0, 150)}


Track ID 5 reappears at frame 300
  │
  ├─▶ Memory check: Found (Team 0, frame 150)
  │     │ Age: 300 - 150 = 150 frames
  │     │ Decay threshold: 150 frames
  │     │ 150 >= 150 ✗ Expired
  │     └─▶ Memory deleted, will reassign in batch
  │
  └─▶ After batch reassignment:
        └─▶ Memory updated: {5: (0, 300)}
```

### Automatic Cleanup

```python
# Called every frame
def _cleanup_memory():
    current_frame = tracker.current_frame_idx
    
    for track_id, (team_id, last_seen) in tracker.team_memory.items():
        age = current_frame - last_seen
        
        if age > memory_decay_frames:
            # Remove expired entry
            del tracker.team_memory[track_id]
```

---

## Color-Based Assignment Flow

```
Input: frames, detections_per_frame
  │
  ├─▶ For each frame with players:
  │     │
  │     ├─▶ For each player:
  │     │     │
  │     │     ├─▶ Extract player crop
  │     │     │     └─▶ bbox = [x1, y1, x2, y2]
  │     │     │
  │     │     ├─▶ Skip if heavily overlapped
  │     │     │     └─▶ overlap_ratio > threshold
  │     │     │
  │     │     ├─▶ Segment player from background
  │     │     │     └─▶ KMeans(n_clusters=2) on RGB
  │     │     │
  │     │     ├─▶ Convert to HSV color space
  │     │     │
  │     │     ├─▶ Count pixels in each color range
  │     │     │     └─▶ red, blue, green, etc.
  │     │     │
  │     │     └─▶ Accumulate color counts per player
  │     │
  │     └─▶ Player color profiles collected
  │
  ├─▶ Find dominant color per player
  │     └─▶ player_dominant_colors = {player_id: color}
  │
  ├─▶ Find 2 most common colors
  │     └─▶ team_colors = [color1, color2]
  │
  ├─▶ Create mapping: color → team
  │     └─▶ color_to_team = {color1: 0, color2: 1}
  │
  └─▶ Assign players to teams
        └─▶ team_mapping = {player_id: team_id}
```

---

## Embedding-Based Assignment Flow

```
Input: frames, detections_per_frame
  │
  ├─▶ Collect crops (with stride sampling)
  │     │
  │     ├─▶ For every Nth frame (stride=3):
  │     │     │
  │     │     ├─▶ For each player:
  │     │     │     │
  │     │     │     ├─▶ Get bbox
  │     │     │     │     └─▶ [x1, y1, x2, y2]
  │     │     │     │
  │     │     │     ├─▶ Shrink bbox (scale=0.7)
  │     │     │     │     └─▶ Focus on jersey area
  │     │     │     │
  │     │     │     ├─▶ Crop image
  │     │     │     │     └─▶ crop = frame[y1:y2, x1:x2]
  │     │     │     │
  │     │     │     └─▶ Store crop for player_id
  │     │     │
  │     │     └─▶ crops_per_player[player_id].append(crop)
  │     │
  │     └─▶ Filter: Keep players with >= 3 crops
  │
  ├─▶ Extract embeddings
  │     │
  │     ├─▶ Convert crops to PIL images
  │     │     └─▶ RGB format for SigLip
  │     │
  │     ├─▶ Create batches (batch_size=256)
  │     │
  │     ├─▶ For each batch:
  │     │     │
  │     │     ├─▶ Preprocess with SigLip processor
  │     │     │     └─▶ Resize, normalize
  │     │     │
  │     │     ├─▶ Run through SigLip model
  │     │     │     └─▶ outputs.last_hidden_state
  │     │     │
  │     │     ├─▶ Mean pooling over spatial dims
  │     │     │     └─▶ (batch, H*W, 768) → (batch, 768)
  │     │     │
  │     │     └─▶ Collect embeddings
  │     │
  │     └─▶ embeddings = (N_crops, 768)
  │
  ├─▶ Average embeddings per player
  │     └─▶ player_embeddings[player_id] = mean(embeddings)
  │
  ├─▶ UMAP dimensionality reduction
  │     │ Input: (N_players, 768)
  │     │ Output: (N_players, 3)
  │     └─▶ projections
  │
  ├─▶ KMeans++ clustering
  │     │ n_clusters = 2
  │     │ init = "k-means++"
  │     └─▶ labels = [0, 1, 0, 1, ...]
  │
  └─▶ Create team mapping
        └─▶ team_mapping = {player_id: label}
```

---

## Goalkeeper Assignment Flow

```
Input: detections_per_frame, player_team_mapping
  │
  ├─▶ Collect positions across all frames
  │     │
  │     ├─▶ For each frame:
  │     │     │
  │     │     ├─▶ Collect player positions
  │     │     │     └─▶ player_positions[player_id].append((x, y))
  │     │     │
  │     │     └─▶ Collect goalkeeper positions
  │     │           └─▶ gk_positions[gk_id].append((x, y))
  │     │
  │     └─▶ Position histories collected
  │
  ├─▶ Compute average positions
  │     │
  │     ├─▶ For each player:
  │     │     └─▶ avg_pos = mean(all_positions)
  │     │
  │     └─▶ For each goalkeeper:
  │           └─▶ avg_pos = mean(all_positions)
  │
  ├─▶ Prepare arrays
  │     │
  │     ├─▶ players_xy = [(x1, y1), (x2, y2), ...]
  │     ├─▶ players_team_id = [0, 1, 0, 1, ...]
  │     └─▶ goalkeepers_xy = [(gx1, gy1), (gx2, gy2), ...]
  │
  ├─▶ For each goalkeeper:
  │     │
  │     ├─▶ Compute distances to all players
  │     │     └─▶ distances = ||gk_pos - player_pos||
  │     │
  │     ├─▶ Find k=5 nearest players
  │     │     └─▶ nearest_indices = argpartition(distances, k)
  │     │
  │     ├─▶ Get nearest players' teams and distances
  │     │     ├─▶ nearest_teams = [0, 0, 1, 0, 1]
  │     │     └─▶ nearest_distances = [50, 80, 200, 120, 250]
  │     │
  │     ├─▶ Compute weights (inverse distance)
  │     │     └─▶ weights = 1 / (distances + epsilon)
  │     │
  │     ├─▶ Sum weights per team
  │     │     ├─▶ Team 0: w1 + w2 + w4 = 0.041
  │     │     └─▶ Team 1: w3 + w5 = 0.009
  │     │
  │     └─▶ Assign to team with max weight
  │           └─▶ gk_team = argmax(team_weights) = 0
  │
  └─▶ Update team_mapping with goalkeepers
        └─▶ team_mapping[gk_id] = gk_team
```

---

## Complete Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Video Input                                         │
├─────────────────────────────────────────────────────────────┤
│ • Read video at target FPS                                  │
│ • Extract frames                                            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Detection & Tracking (Frame-by-Frame)              │
├─────────────────────────────────────────────────────────────┤
│ For each frame:                                             │
│   1. YOLO detection → boxes, confidences, classes           │
│   2. BoTSORT tracking → track_ids                           │
│   3. Organize by class → Player, Goalkeeper, Ball           │
│   4. Store detections_per_frame                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Team Assignment (Batch)                            │
├─────────────────────────────────────────────────────────────┤
│ Method: Color or Embedding (configured)                     │
│   1. Run assignment algorithm on all frames                 │
│   2. Get initial_team_mapping                               │
│   3. Update tracker memory                                  │
│   4. Refine with memory consensus                           │
│   5. Return final_team_mapping                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Goalkeeper Assignment                              │
├─────────────────────────────────────────────────────────────┤
│   1. Collect player & goalkeeper positions                  │
│   2. Distance-weighted voting                               │
│   3. Update team_mapping with goalkeepers                   │
│   4. Update tracker memory                                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Data Processing                                    │
├─────────────────────────────────────────────────────────────┤
│   1. Create DataFrame from detections                       │
│   2. Interpolate missing positions                          │
│   3. Merge fragmented IDs                                   │
│   4. Smooth trajectories (optional)                         │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Output Generation                                  │
├─────────────────────────────────────────────────────────────┤
│   1. Create annotated video                                 │
│   2. Save JSON data files                                   │
│   3. Save metadata                                          │
│   4. Print summary                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Why Batch Team Assignment?
**Reason:** More robust than frame-by-frame
- Analyzes entire video for consistent colors/features
- Reduces noise from single-frame variations
- Allows averaging embeddings across multiple observations

### 2. Why Memory System?
**Reason:** Consistency and performance
- Prevents team switching due to temporary occlusions
- Reduces re-computation for recurring track IDs
- Balances consistency with adaptability (via decay)

### 3. Why Distance-Weighted for Goalkeepers?
**Reason:** More accurate than simple majority voting
- Closer players have more influence (more likely same team)
- Robust to outliers (far players weighted less)
- Works well even with unbalanced player distributions

### 4. Why Factory Pattern for Team Assigners?
**Reason:** Easy extensibility
- Can add new methods without changing main pipeline
- Clean separation of concerns
- Configuration-driven selection

---

## Performance Optimization Tips

### 1. Embedding-Based Assignment
```python
# Optimize batch size for your GPU
config.team_assigner.embedding_batch_size = 256  # RTX 3090
config.team_assigner.embedding_batch_size = 128  # RTX 2080
config.team_assigner.embedding_batch_size = 64   # GTX 1080

# Reduce stride for more samples (better quality, slower)
config.team_assigner.stride = 2  # More crops, better quality
config.team_assigner.stride = 5  # Fewer crops, faster
```

### 2. Memory Management
```python
# Longer memory for stable teams
config.team_assigner.memory_decay_frames = 300  # More stable

# Shorter memory for dynamic scenarios
config.team_assigner.memory_decay_frames = 75   # More adaptive
```

### 3. Detection Quality
```python
# Higher confidence for fewer false positives
config.detector.confidence_threshold = 0.5  # More precise

# Lower confidence for better recall
config.detector.confidence_threshold = 0.3  # More detections
```

---

## Testing & Validation

### Unit Tests Checklist
- [ ] Test color-based assignment with 10+ players
- [ ] Test embedding-based assignment with 10+ players
- [ ] Test memory storage and retrieval
- [ ] Test memory expiration logic
- [ ] Test goalkeeper assignment with various distributions
- [ ] Test with missing players (occlusions)
- [ ] Test with overlapping detections

### Integration Tests Checklist
- [ ] End-to-end with color method
- [ ] End-to-end with embedding method
- [ ] Memory persistence across 300+ frames
- [ ] Goalkeeper assignment with 2 teams
- [ ] Multiple videos in batch

### Performance Tests Checklist
- [ ] Measure embedding extraction time
- [ ] Measure memory lookup time (should be O(1))
- [ ] Measure goalkeeper assignment time
- [ ] Monitor GPU memory usage
- [ ] Profile batch processing

---

## Debugging Guide

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Memory State
```python
# After processing
print(f"Active memories: {len(tracker.team_memory)}")
print(f"Memory contents: {tracker.team_memory}")
```

### Visualize Team Assignments
```python
# Print team distribution
for team_id in [0, 1]:
    count = sum(1 for t in team_mapping.values() if t == team_id)
    print(f"Team {team_id}: {count} players")
```

### Validate Goalkeeper Assignments
```python
# Check goalkeeper team distribution
gk_teams = [team_mapping[gid] for gid in goalkeeper_ids]
print(f"Goalkeepers: Team 0={gk_teams.count(0)}, Team 1={gk_teams.count(1)}")
```

---

**This completes the integration guide. All components work together seamlessly through the tracker's team memory system, providing robust and consistent team assignments across the entire video.**
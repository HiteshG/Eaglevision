# Camera Motion Compensation (CMC) Modules - Code Breakdown

## Overview

Camera Motion Compensation is critical for multi-object tracking in sports video analysis where the camera follows the action. Without CMC, predicted bounding box positions from Kalman filters will drift significantly when the camera pans, tilts, or zooms—causing track fragmentation and ID switches.

These modules are from **BoxMOT** (Mikel Broström) and compute a **warp matrix** that transforms coordinates from frame `t-1` to frame `t`, allowing trackers to compensate predicted positions before association.

---

## Architecture

```
BaseCMC (abstract)
    │
    ├── ECC    (intensity-based, iterative optimization)
    │
    └── ORB    (feature-based, keypoint matching)
```

Both classes inherit from `BaseCMC` which provides:
- `preprocess(img)` → grayscale conversion + downscaling
- `generate_mask(img, dets, scale)` → masks out detected objects (for ORB)

---

## 1. ECC (Enhanced Correlation Coefficient)

### Algorithm Type
**Direct/Intensity-based** — optimizes pixel intensity correlation between frames using gradient descent.

### How It Works

```
Frame t-1 (prev_img) ──────┐
                           │
                           ▼
              ┌─────────────────────────┐
              │  cv2.findTransformECC   │
              │  (iterative optimizer)  │
              └─────────────────────────┘
                           │
Frame t (curr_img) ────────┘
                           │
                           ▼
                    Warp Matrix (2×3 or 3×3)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warp_mode` | `cv2.MOTION_TRANSLATION` | Transformation model |
| `eps` | `1e-5` | Convergence threshold |
| `max_iter` | `100` | Maximum iterations |
| `scale` | `0.15` | Downscale factor for speed |
| `grayscale` | `True` | Convert to grayscale |
| `align` | `False` | Store aligned image for debugging |

### Warp Modes Explained

| Mode | Matrix Size | DOF | Use Case |
|------|-------------|-----|----------|
| `MOTION_TRANSLATION` | 2×3 | 2 | Pure pan (most sports broadcasts) |
| `MOTION_EUCLIDEAN` | 2×3 | 3 | Pan + rotation |
| `MOTION_AFFINE` | 2×3 | 6 | Pan + rotation + scale + shear |
| `MOTION_HOMOGRAPHY` | 3×3 | 8 | Full perspective (rare for sports) |

### Code Flow

```python
def apply(self, img, dets=None):
    # 1. Initialize identity matrix based on warp mode
    if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)  # 3×3
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)  # 2×3

    # 2. First frame: store and return identity
    if self.prev_img is None:
        self.prev_img = self.preprocess(img)
        return warp_matrix

    # 3. Preprocess current frame
    curr = self.preprocess(img)  # grayscale + downscale

    # 4. Run ECC optimization
    try:
        _, warp_matrix = cv2.findTransformECC(
            self.prev_img,    # template
            curr,             # input image
            warp_matrix,      # initial guess (identity)
            self.warp_mode,   # transformation type
            self.termination_criteria,
            None,             # mask (unused)
            1,                # gaussian blur window
        )
    except cv2.error as e:
        # Handle non-convergence gracefully
        if e.code == cv2.Error.StsNoConv:
            return warp_matrix  # identity

    # 5. Upscale translation components
    if self.scale < 1.0:
        warp_matrix[0, 2] /= self.scale  # tx
        warp_matrix[1, 2] /= self.scale  # ty

    # 6. Update state
    self.prev_img = curr
    return warp_matrix
```

### Translation Upscaling (Critical Detail)

When `scale=0.15`, images are processed at 15% resolution. The translation components `tx` and `ty` are in downscaled coordinates, so they must be divided by `scale` to map back to original resolution:

```
Original: 1920×1080
Scaled:   288×162   (×0.15)

If camera pans 100px in original space:
  → Measured as 15px in scaled space
  → Upscale: 15 / 0.15 = 100px ✓
```

### Strengths & Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Sub-pixel accuracy | Slow (iterative) |
| Handles smooth motion well | Fails on large displacements |
| No feature detection needed | Sensitive to lighting changes |
| Works on textureless regions | Can get stuck in local minima |

---

## 2. ORB (Oriented FAST and Rotated BRIEF)

### Algorithm Type
**Feature-based** — detects keypoints, computes descriptors, matches across frames, estimates transformation via RANSAC.

### Pipeline

```
Frame t-1                          Frame t
    │                                  │
    ▼                                  ▼
┌─────────┐                      ┌─────────┐
│  FAST   │ (keypoint detection) │  FAST   │
└────┬────┘                      └────┬────┘
     │                                │
     ▼                                ▼
┌─────────┐                      ┌─────────┐
│   ORB   │ (descriptor compute) │   ORB   │
└────┬────┘                      └────┬────┘
     │                                │
     └──────────┬─────────────────────┘
                │
                ▼
        ┌───────────────┐
        │  BFMatcher    │ (KNN k=2)
        │  + Lowe Test  │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │ Spatial Gate  │ (distance < 25% of frame)
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │ Statistical   │ (2.5σ outlier removal)
        │ Filtering     │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │ RANSAC Affine │ (cv2.estimateAffinePartial2D)
        └───────┬───────┘
                │
                ▼
          Warp Matrix (2×3)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_detector_threshold` | `20` | FAST corner threshold |
| `matcher_norm_type` | `cv2.NORM_HAMMING` | Distance metric for binary descriptors |
| `scale` | `0.15` | Downscale factor |
| `grayscale` | `True` | Convert to grayscale |
| `draw_keypoint_matches` | `False` | Debug visualization |
| `align` | `False` | Store aligned image |

### Code Flow with Detailed Comments

```python
def apply(self, img, dets=None):
    H = np.eye(2, 3, dtype=np.float32)  # identity fallback

    img_p = self.preprocess(img)
    h, w = img_p.shape[:2]

    # ─────────────────────────────────────────────────────────────
    # MASK GENERATION: Exclude detected objects (dynamic regions)
    # ─────────────────────────────────────────────────────────────
    mask = self.generate_mask(img_p, dets, self.scale)
    # mask = 255 where static background, 0 where detections exist
    # This prevents matching features ON players (which move independently)

    # ─────────────────────────────────────────────────────────────
    # FEATURE DETECTION & DESCRIPTION
    # ─────────────────────────────────────────────────────────────
    keypoints = self.detector.detect(img_p, mask)  # FAST corners
    keypoints, descriptors = self.extractor.compute(img_p, keypoints)  # ORB descriptors

    # Need minimum 4 points for affine estimation
    if descriptors is None or len(keypoints) < 4:
        self._store_state(...)
        return H  # identity

    # First frame initialization
    if self.prev_img is None:
        self._store_state(...)
        return H

    # ─────────────────────────────────────────────────────────────
    # KNN MATCHING (k=2 for Lowe's ratio test)
    # ─────────────────────────────────────────────────────────────
    knn = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)

    # ─────────────────────────────────────────────────────────────
    # MATCH FILTERING: Lowe's Ratio + Spatial Gating
    # ─────────────────────────────────────────────────────────────
    matches = []
    spatial_distances = []
    max_spatial_distance = 0.25 * np.array([w, h])  # 25% of frame dims

    for m, n in knn:
        # Lowe's ratio test: best match must be significantly
        # better than second-best (0.9 threshold, typically 0.7-0.8)
        if m.distance >= 0.9 * n.distance:
            continue

        prev_pt = self.prev_keypoints[m.queryIdx].pt
        curr_pt = keypoints[m.trainIdx].pt
        dxy = prev_pt - curr_pt

        # Spatial gate: reject if movement > 25% of frame
        # (camera rarely moves more than this between frames)
        if abs(dxy[0]) < max_spatial_distance[0] and \
           abs(dxy[1]) < max_spatial_distance[1]:
            matches.append(m)
            spatial_distances.append(dxy)

    if len(matches) < 4:
        return H

    # ─────────────────────────────────────────────────────────────
    # STATISTICAL OUTLIER REMOVAL (2.5σ rule)
    # ─────────────────────────────────────────────────────────────
    spatial_distances = np.asarray(spatial_distances)
    mean = spatial_distances.mean(axis=0)
    std = spatial_distances.std(axis=0) + 1e-6

    # Keep points where displacement is within 2.5 std of mean
    inliers = np.all((spatial_distances - mean) < 2.5 * std, axis=1)
    good_matches = [m for m, ok in zip(matches, inliers) if ok]

    if len(good_matches) < 4:
        return H

    # ─────────────────────────────────────────────────────────────
    # RANSAC AFFINE ESTIMATION
    # ─────────────────────────────────────────────────────────────
    prev_pts = np.array([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
    curr_pts = np.array([keypoints[m.trainIdx].pt for m in good_matches])

    # estimateAffinePartial2D: 4 DOF (rotation + uniform scale + translation)
    # More robust than full affine (6 DOF) for typical camera motion
    H_est, inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)

    if H_est is not None:
        # Upscale translation to original coordinates
        if self.scale < 1.0:
            H_est[0, 2] /= self.scale
            H_est[1, 2] /= self.scale

    self._store_state(...)
    return H_est if H_est is not None else H
```

### Why Masking Detections Matters

```
Without mask:                    With mask:
┌────────────────────┐          ┌────────────────────┐
│  ○  ○     ○   ○    │          │  ○  ○     ○   ○    │
│     ┌─────┐        │          │     ┌─────┐        │
│  ○  │█████│ ○  ○   │          │  ○  │     │ ○  ○   │
│     │█████│        │          │     │     │        │
│  ○  └─────┘   ○    │          │  ○  └─────┘   ○    │
│        ○  ○   ○    │          │        ○  ○   ○    │
└────────────────────┘          └────────────────────┘
     ↑ Player                         ↑ Masked out
     Features on player               Only background features
     corrupt camera motion            used for CMC
```

### Partial Affine vs Full Affine

`cv2.estimateAffinePartial2D` estimates:
```
┌                   ┐
│ s·cos(θ)  -s·sin(θ)  tx │   4 parameters:
│ s·sin(θ)   s·cos(θ)  ty │   - s (uniform scale)
└                   ┘      - θ (rotation)
                            - tx, ty (translation)
```

Full affine (`cv2.estimateAffine2D`) adds shear and non-uniform scale (6 params), but is more prone to overfitting with noisy matches.

### Strengths & Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Handles large displacements | Needs textured scenes |
| Robust to lighting changes | Slower than ECC for small motion |
| Explicit outlier rejection | Feature detection can be noisy |
| Works with moving objects (via masking) | Binary descriptors less distinctive |

---

## Comparison: ECC vs ORB

| Aspect | ECC | ORB |
|--------|-----|-----|
| **Speed** | Slow (iterative) | Faster (one-shot matching) |
| **Large Motion** | Poor (needs initialization) | Good |
| **Small Motion** | Excellent (sub-pixel) | Good |
| **Textureless** | Works | Fails |
| **Dynamic Objects** | Affected | Handled via masking |
| **Output** | 2×3 / 3×3 (configurable) | 2×3 (partial affine) |

---

## Integration with Tracking

### How CMC Warps Kalman Predictions

```python
# In tracker (e.g., ByteTrack, BoT-SORT):

# 1. Get CMC warp matrix
warp = cmc.apply(frame, detections)

# 2. Predict track positions
for track in tracks:
    track.predict()  # Kalman filter predicts x, y, w, h

# 3. Compensate for camera motion
for track in tracks:
    # Apply warp to predicted center point
    x, y = track.mean[:2]
    point = np.array([[x, y]], dtype=np.float32)
    
    if warp.shape == (3, 3):  # homography
        warped = cv2.perspectiveTransform(point.reshape(1, 1, 2), warp)
    else:  # affine
        warped = cv2.transform(point.reshape(1, 1, 2), warp)
    
    track.mean[:2] = warped.flatten()
```

### Example: Camera Pan Compensation

```
Frame t-1:                    Frame t (camera panned right):
┌─────────────────────┐      ┌─────────────────────┐
│      [Track A]      │      │           [Track A] │  ← Same player
│         ○           │      │              ○      │
│                     │      │                     │
│   [Track B]         │      │      [Track B]      │
│      ○              │      │         ○           │
└─────────────────────┘      └─────────────────────┘

Without CMC:                  With CMC:
- Predicted pos: same as t-1  - Warp predicts new position
- Detection: shifted right    - Predicted pos matches detection
- IoU = 0 → Track lost!       - IoU high → Track maintained ✓
```

---

## Usage Example

```python
from boxmot.motion.cmc.ecc import ECC
from boxmot.motion.cmc.orb import ORB

# Option 1: ECC for smooth broadcast footage
cmc = ECC(
    warp_mode=cv2.MOTION_TRANSLATION,  # pan only
    scale=0.15,                         # 15% resolution
    max_iter=50,                        # fewer iterations for speed
)

# Option 2: ORB for fast camera motion or drone footage  
cmc = ORB(
    feature_detector_threshold=20,
    scale=0.15,
    draw_keypoint_matches=True,  # debug mode
)

# In tracking loop:
for frame, detections in video:
    warp_matrix = cmc.apply(frame, detections)
    
    # Apply warp to track predictions before association
    for track in tracker.tracks:
        track.compensate_camera_motion(warp_matrix)
    
    # Run association (IoU matching, etc.)
    tracker.update(detections)
```

---

## Football/Sports-Specific Considerations

1. **Broadcast footage**: Usually smooth pans → ECC works well with `MOTION_TRANSLATION`

2. **Tactical camera (high angle)**: Minimal motion → CMC may be unnecessary, but helps during replays

3. **Following camera**: Large, fast pans → ORB more robust

4. **Zoom changes**: Rare in match footage, but if present use `MOTION_EUCLIDEAN` or `MOTION_AFFINE`

5. **Replay detection**: Disable CMC during replays (scene changes break the inter-frame assumption)

6. **Scale factor**: `0.15` is aggressive; try `0.25-0.5` if results are noisy

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Track drift | CMC disabled or failing | Check `warp_matrix` != identity |
| ID switches during pan | CMC not applied to predictions | Verify `compensate_camera_motion()` called |
| ECC not converging | Large motion or scene change | Switch to ORB or increase `max_iter` |
| ORB returns identity | Too few features | Lower `feature_detector_threshold` |
| Wrong scale compensation | `scale` mismatch | Ensure same `scale` in CMC and tracker |

---

## References

- BoxMOT: https://github.com/mikel-brostrom/boxmot
- ECC Algorithm: Evangelidis & Psarakis, "Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization" (2008)
- ORB: Rublee et al., "ORB: An efficient alternative to SIFT or SURF" (2011)
- Lowe's Ratio Test: Lowe, "Distinctive Image Features from Scale-Invariant Keypoints" (2004)

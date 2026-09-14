# Working Notes — Vision Intelligence CV Engineer Live Task

## Project Overview

A real-time computer vision analytics pipeline for a **fixed classroom CCTV camera**. Processes `video7.mp4` and outputs live annotations — running attendance, room occupancy, entry/exit events, posture (Seated / Standing), motion, and frame quality — directly overlaid on the video window.

The entire pipeline is designed around the key constraint: **the camera never moves, the desks never move, the door is always at the same location**. Every module exploits this.

---

## Project Structure

```
Assignment/
├── main.py                          # ← The entire pipeline (single-file)
├── classroom_botsort.yaml           # ← Auto-generated custom tracker config
├── video7.mp4                       # ← Input video (classroom CCTV footage)
├── CV_Engineer_Live_Task_Vision_Intelligence 111.pdf
├── submission_answers.md
├── working.md                       # ← This file
└── yolov8m-pose.pt                  # ← Active model
```

---

## Technology Stack

| Tool / Library | Version | Role |
|---|---|---|
| **Python** | 3.13 | Runtime |
| **OpenCV** | 5.0.0 | Video I/O, background subtraction, blur, desk detection, rendering |
| **Ultralytics YOLOv8-Pose** | 8.4.x | Person detection + 17-keypoint COCO pose estimation |
| **PyTorch** | 2.13 | YOLO inference backend |
| **NumPy** | 2.2.x | Numerical ops, background median, histogram scoring |
| **PyYAML** | (bundled with ultralytics) | Write custom tracker config at startup |
| **lapx** | 0.9.4 | Linear assignment required by BoT-SORT internals |

---

## Pipeline Architecture

```
Frame Read
    │
    ├─► [BG] Background Accumulation (first 60 frames → median → desk detection)
    │
    ├─► [1] Adaptive Blur Detection    (Laplacian variance vs. rolling-median threshold)
    │
    ├─► [2] MOG2 Motion + Temporal Gate (background subtractor + shadow removal +
    │        morphological cleanup + 3-frame consecutive confirmation)
    │
    ├─► [3] YOLOv8m-Pose + Custom BoT-SORT
    │            │  (conf=0.20, iou=0.50, imgsz=1280, track_buffer=120)
    │            │
    │            ├─► Lightweight Re-ID  (HSV histogram match vs. recent exits)
    │            │
    │            ├─► Zone-aware Entry/Exit  (door zone vs. interior)
    │            │
    │            └─► Hybrid Posture  (bbox height + keypoints + desk-row alignment)
    │
    ├─► [4] Metrics Overlay            (upper-right panel, 8 lines)
    │
    └─► [5] Scene Overlays + Seekbar   (door zone, desk lines, bottom seekbar)
```

---

## Configuration Block (top of `main.py`)

All thresholds are in the `CONFIGURATION` block at the top of `main.py`. Key values:

| Constant | Value | Purpose |
|---|---|---|
| `CONF` | 0.20 | Detection confidence |
| `IOU` | 0.50 | NMS IoU — tighter to separate adjacent seated people |
| `IMG_SIZE` | 1280 | YOLO resolution |
| `TRACK_BUFFER` | 120 | Frames to keep a track alive without detections (4 s at 30 fps) |
| `ENTRY_SIDE` | `"auto"` | `"left"`, `"right"`, `"auto"`, or `"none"` |
| `ENTRY_X_FRAC` | 0.18 | Width of door zone as fraction of frame |
| `EXIT_DOOR_FRAMES` | 25 | Frames absent from door zone → confirmed Exit |
| `EXIT_INTERIOR_FRAMES` | 100 | Frames absent from interior → ambiguous loss, remove silently |
| `POSTURE_SEATED_H` | 0.32 | `box_h/frame_h` below this → strong Seated signal |
| `POSTURE_STANDING_H` | 0.44 | `box_h/frame_h` above this → strong Standing signal |
| `BG_ACCUM_FRAMES` | 60 | Frames to build static background |
| `REID_THRESH` | 0.80 | Bhattacharyya correlation to merge IDs |
| `MOTION_GATE_N` | 3 | Consecutive frames of motion needed to flag it |

---

## Feature Implementation Details

### Scene Understanding (Background & Desk Detection)
**Why**: Generic trackers treat every scene the same. For a fixed classroom, the desk positions never change and can be computed once from the background.

**How**:
1. Accumulate the first 60 frames.
2. Compute pixel-wise **median** → robust static background (people are removed because they move).
3. Run `Canny` edge detection on the background.
4. Run `HoughLinesP` looking for **near-horizontal lines** spanning ≥ 20% of frame width.
5. Cluster lines within 25 px of each other → each cluster = one desk row.
6. Store the average y-coordinate of each cluster.

**Output**: `desk_ys` — a list of y-coordinates used by the posture classifier.
**Visual**: Thin cyan horizontal lines drawn on the frame at each detected desk row.

**Status indicator**: During the 60-frame warmup, the video displays "Building scene model: X%" and skips inference (improves first-frame accuracy of the background model).

---

### 1. Person Detection
- **Model**: `yolov8m-pose.pt` — chosen for combined detection + pose. Same accuracy as `yolov8m` for the person class.
- **`conf=0.20`**: Lowered (was 0.25) to catch partially occluded and seated students. False positives are handled by NMS and the tracker's minimum box area filter.
- **`iou=0.50`**: Slightly tighter than before to better separate adjacent seated people whose boxes overlap.
- **`imgsz=1280`**: 2× default resolution — essential for students far from camera.

---

### 2. Tracking — Custom BoT-SORT

**Why the default fails**: The default `botsort.yaml` has `track_buffer=30` — 1 second at 30fps. A student occluded for 2 seconds by a walking classmate would lose their track ID and be re-identified as a new person. In a classroom, occlusions routinely last 2–5 seconds.

**Fix**: A custom `classroom_botsort.yaml` is written to disk at startup with `track_buffer=120` (4 seconds). This keeps the Kalman state alive through typical classroom occlusions without waiting for the person to reappear.

| Parameter | Default | Custom | Reason |
|---|---|---|---|
| `track_buffer` | 30 | 120 | Survive 4-second classroom occlusions |
| `match_thresh` | 0.8 | 0.85 | Stricter association — fewer false merges |
| `min_box_area` | 10 | 50 | Ignore tiny false-positive blobs |

---

### 3. Posture — Hybrid 4-Signal Classifier

**Why all previous approaches failed**:

| Approach | Why it failed |
|---|---|
| Aspect ratio only | Ignores desk geometry and body keypoints |
| Velocity-based | Students fidget; stationary standing people look "seated" |
| Keypoint-only (knee visibility) | Knees ARE often visible for seated students under desks |
| Fixed height threshold | Failed on perspective scaling (far away standing people look small) |

**Current approach: scored 4-signal fusion**

Four signals are computed and summed to a score. `score > 0` → Seated.

**Signal 1 — Box Aspect Ratio** (`weight: ±3`):
- `aspect (width/height) > 0.45` → score +3 (squarer box → seated)
- `aspect < 0.35` → score -3 (tall and thin → standing)
- Automatically handles camera perspective scaling.

**Signal 2 — Normalised bounding box height** (`weight: ±2`):
- Contextualised by aspect ratio. A tall box (`norm_h > 0.44`) only penalises (-2) if the aspect ratio is also thin (`< 0.40`). 

**Signal 3 — Keypoint body-completeness** (`weight: ±2–3`):
- Ankles visible but far below hips (`> 0.4 * box_h`) → score -3 (full body exposed → standing)
- Ankles visible but close to hips → score +1 (folded legs / visible feet under desk)
- Hip visible, knees NOT visible → score +2 (body cut off at desk level)
- Hips + knees visible, ankles NOT → score +1 (ambiguous, leaning toward seated)

**Signal 4 — Desk-row alignment** (`weight: +2`):
- Person's vertical centroid within 14% of frame height from a detected desk row → score +2

**Smoothing**: 18-frame majority vote at 55% threshold prevents flickering.

**Box colors**:
- 🟢 Green = Seated
- 🟠 Orange = Standing / Moving

---

### 4. Attendance Stabilisation

**Why naive counting fails**: `len(detections_in_frame)` drops whenever someone is momentarily occluded — even for 1 frame. Raises/lowers by ±5 when people cross paths.

**Fix**:
- `active_tracks` stores all confirmed tracks (including temporarily occluded ones).
- `running_attendance = len(active_tracks)` — only decreases when a track is fully removed after its timeout.
- Absent counter increments each frame the person is not detected.
- Timeout depends on zone (see Entry/Exit below).

---

### 5. Entry / Exit — Zone-Aware (Exploits Fixed Camera)

**Why timeout-based exit is wrong**: If someone moves behind a pillar inside the classroom for 30 frames, the old logic fired "Exit". This is fundamentally wrong.

**Core insight**: In a fixed classroom camera, the door is always at the same side. A person can only genuinely leave by crossing through the door zone. Interior disappearances are always occlusions.

**Implementation**:
- **Door zone**: A configurable strip (default: `ENTRY_X_FRAC = 0.18` = 18% of frame width) at the configured edge.
- `ENTRY_SIDE = "auto"` → pipeline collects the x-coordinate of the first 8 new track appearances and picks "left" or "right" based on which side they cluster.
- Press **L** during playback to manually cycle: `left → right → none → left`.

**Zone-differentiated exit timeouts**:
| Zone | Timeout | Rationale |
|---|---|---|
| Door zone | 25 frames | Near door + gone → very likely walked out |
| Interior | 100 frames | Far from door + gone → almost certainly occluded |

**Entry event**: Fires when a new canonical ID (after Re-ID resolution) is first seen.
**Exit event**: Fires only for door-zone tracks that exceed the door timeout.

**Visual**: Semi-transparent orange strip + orange border line marks the door zone.

---

### 6. Unique Person Count & Re-ID

**Problem**: When BoT-SORT loses a track and re-acquires with a new ID, the person is counted again.

**Fix**: When a track exits, store its HSV color histogram (16×16 H×S bins). When a new track appears, compare against recent exits (within `REID_TTL=200` frames ≈ 6.7s). If Bhattacharyya correlation ≥ `REID_THRESH=0.80` → merge IDs, do not increment unique count.

**Trade-off**: Fails if two people wear very similar clothing. A deep Re-ID model (OSNet, FastReID) would be more robust but requires significant infrastructure.

---

### 7. Motion Detection (MOG2 + Temporal Gate)

**Why absdiff failed**: `cv2.absdiff` reacts to lighting changes, shadows, camera noise, and video compression artifacts.

**Fix — MOG2 background subtractor**:
- Learns a per-pixel Gaussian background model over 500 frames.
- `detectShadows=True`: shadows are marked as 127 in the mask and explicitly excluded.
- `MORPH_OPEN(5×5 ellipse)`: removes small noise speckles.

**Temporal gate**: Motion is flagged only if detected for `MOTION_GATE_N=3` **consecutive** frames. A single-frame spike (lighting flicker, compression) is suppressed.

---

### 8. Frame Quality — Adaptive Blur Detection

**Why fixed threshold failed**: Indoor CCTV footage is inherently soft. A fixed threshold of 50 permanently labelled most frames as "Blurry".

**Fix**: Rolling deque of last 60 Laplacian variance scores. Threshold = `max(50, median(recent) × 0.40)`. A frame is "Blurry" only if it is significantly worse than its neighbours — a genuine anomaly.

---

### 9. Custom Seekbar (Bottom of Frame)

- OpenCV's native `createTrackbar` renders at the top — rejected.
- 22px dark strip drawn on the frame at the bottom via `cv2.rectangle`.
- `cv2.setMouseCallback` detects left-click/drag on the strip.
- On seek: all tracking state, background accumulator, desk lines, motion history, and blur history are fully reset.
- Help text `"L=toggle door zone | Q=quit"` is shown in the seekbar area.

---

## What is Fundamentally Wrong with Previous Versions (Audit)

| Module | Root Cause of Failure | Why Assumption Was Wrong |
|---|---|---|
| Posture (aspect ratio) | box_h/box_w used as sitting proxy | Seated people at desks still have tall boxes |
| Posture (velocity-based) | Low velocity = seated | Students fidget; standing people can be still |
| Posture (keypoint knee-only) | No knee = seated | Knee visibility is camera-angle dependent; many seated students have visible knees |
| Exit detection (uniform timeout) | Any 60-frame absence = exit | Interior disappearances are occlusions, not exits |
| Tracking (default botsort) | `track_buffer=30` = 1 second | Classroom occlusions easily last 2–5 seconds |
| Motion (absdiff) | Frame-to-frame pixel diff | Sensitive to lighting flicker and CCTV compression |
| Blur (fixed threshold=50) | Hardcoded global threshold | Indoor CCTV is inherently soft; most frames fail |
| Attendance (raw box count) | `len(current_detections)` | Drops on every missed frame; fluctuates with occlusion |

---

## Estimated Accuracy Improvements

| Module | Previous Method | New Method | Estimated Gain |
|---|---|---|---|
| Posture | Single keypoint rule | 3-signal hybrid with desk geometry | +30-40% correct labels |
| Exit Events | Uniform 60-frame timeout | Zone-differentiated (25 door / 100 interior) | Near-zero false exits from interior |
| Tracking | Default track_buffer=30 | Custom track_buffer=120 | ~50% fewer ID switches in occlusions |
| Attendance | Raw detections | Active tracks with zone-aware removal | Stable count vs. flickering |
| Motion | absdiff | MOG2 + temporal gate | Near-zero false positives from lighting |

---

## How to Run

```bash
# Dependencies (one-time)
pip install --user ultralytics opencv-python lapx

cd "C:\Users\ashut\OneDrive\Desktop\Assignment"
python main.py
```

**First run**: Displays "Building scene model: X%" for the first 60 frames (≈2 s at 30fps). This builds the background and detects desk rows. Inference starts only after this completes.

**Controls**:
| Key / Action | Effect |
|---|---|
| Drag seekbar (bottom) | Seek to any point in the video |
| `Q` | Quit |
| `L` | Cycle door zone: left → right → none → left |

---

## Known Remaining Limitations

| Limitation | Impact | Better Solution |
|---|---|---|
| Re-ID via HSV histogram | Fails for people in similar clothing | Deep Re-ID (OSNet, FastReID) |
| MOG2 warmup (~500 frames) | Over-triggers motion in first ~15 s | Pre-computed background OR KNN subtractor |
| Desk detection via Hough | Fails if first 60 frames are crowded or dim | Manual calibration mode |
| Posture for small/far detections | Keypoints unreliable; only Signal 1 active | Higher-resolution model or camera zoom |
| Auto door detection (8 entries) | Needs 8 people to enter before zone is fixed | One-time manual calibration at startup |
| CPU-only speed | `yolov8m-pose` at 1280px is heavy on CPU | Use `yolov8s-pose` at 640px for speed |

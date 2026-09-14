# Vision Intelligence CV Engineer Live Task - Project Report

## 1. Project Overview & Design Philosophy

This project is a real-time computer vision analytics pipeline designed specifically for **classroom CCTV monitoring**. It processes the provided video (`video7.mp4`) and outputs live analytics—including running attendance, unique footfall, posture classification (Seated vs. Standing), room occupancy, motion detection, and frame quality—directly onto the video window.

### Why This is NOT a Generic Computer Vision Solution
Most generic object detection and tracking algorithms are designed to work on moving cameras (drones, dashcams) or dynamic environments. **The core engineering philosophy behind this solution was to exploit the fixed constraints of a CCTV camera:**
- The camera never moves.
- The desks never move.
- The entrance/exit location is always the same.

By designing around these constraints, the pipeline uses scene-understanding (like automatic desk-row detection from the background) and zone-aware logic (virtual entry/exit lines) to achieve much higher accuracy than a naive, out-of-the-box YOLO implementation.

---

## 2. Technology Stack

- **Python 3.13 & OpenCV**: For video I/O, frame rendering, traditional CV algorithms (MOG2 background subtraction, Laplacian variance blur detection, Hough Lines), and the custom seekbar UI.
- **Ultralytics YOLOv8m-Pose**: The medium-sized pose model. This was chosen over a standard detection model because accurately classifying complex classroom postures requires both bounding boxes and the 17 COCO skeletal keypoints.
- **BoT-SORT (Customized)**: The primary multi-object tracking algorithm. It handles camera motion well and is highly robust.
- **NumPy**: Used for fast matrix operations, calculating the static background median, and histogram scoring.
- **lapx**: Required internally by BoT-SORT for linear assignment.

---

## 3. How the Pipeline Works

1. **Scene Understanding (Warm-up Phase)**: The first 60 frames are used to build a static background model (using pixel-wise median). Canny Edge Detection and Hough Lines are then run on this background to automatically map out the horizontal rows of desks. This gives the model a geometric understanding of the room.
2. **Blur & Motion Detection**: 
   - **Blur** is detected dynamically using a rolling median of Laplacian variance, ensuring that inherently soft CCTV footage isn't constantly flagged.
   - **Motion** uses OpenCV's MOG2 Background Subtractor with shadow removal, protected by a 3-frame temporal gate to prevent false alarms from lighting flicker.
3. **Detection & Tracking**: The frame is passed to YOLOv8-Pose, and the results are fed into a **custom BoT-SORT configuration**. The default tracker drops IDs after 1 second of occlusion; this custom config extends the memory buffer to 4 seconds, allowing tracks to survive students walking in front of one another.
4. **Lightweight Re-ID**: If the tracker drops an ID, a color histogram (HSV) is saved. When a "new" track appears, it is compared against recent exits. If the correlation is > 0.80, the IDs are merged to prevent double-counting unique entries.
5. **Zone-Aware Entry/Exit**: Exits are only counted if a person disappears while inside a defined "Door Zone" at the edge of the screen. If they disappear in the center of the room, they are simply marked as occluded, keeping the running attendance stable.

---

## 4. The Hardest Challenge: Posture Classification

**The Problem**: Classifying Seated vs. Standing in a classroom is notoriously difficult. Standard approaches fail:
- *Bounding box aspect ratio* fails because a seated student (head and torso visible above a desk) still has a tall bounding box.
- *Velocity-based heuristics* fail because seated students fidget and standing students can stand perfectly still.
- *Simple keypoint checks (e.g., knee visibility)* fail because students in the front row often have their legs visible under the desks.

**The Solution: A 4-Signal Hybrid Classifier**
I engineered a robust scoring system combining four signals to determine posture:
1. **Aspect Ratio**: Seated people are noticeably "squarer" (`width/height > 0.45`), while standing people are "tall and thin". This inherently solves the issue of camera perspective scaling (far away vs. close up).
2. **Normalized Height**: A contextual check. A tall box only penalizes the "Seated" score if it is also thin.
3. **Smart Keypoint Distance**: If YOLO detects ankles, the algorithm measures the vertical distance between the hips and ankles. If the distance is small (legs folded or visible under a desk), it doesn't penalize them. It only flags "Standing" if the ankles are far below the hips.
4. **Desk Alignment**: If the person's vertical centroid aligns with the desk rows automatically found during the warm-up phase, they receive a strong "Seated" bonus.

These scores are smoothed using an 18-frame majority vote to prevent the UI from flickering on a single bad detection.

---

## 5. Known Limitations

- **Re-ID Collisions**: The lightweight HSV histogram Re-ID can fail if two people are wearing nearly identical clothing. A deep Re-ID model (like OSNet) would fix this but requires significantly more compute.
- **CPU Performance**: YOLOv8m-Pose at 1280px resolution is heavy. On a CPU-only machine, it runs below real-time. For faster CPU performance, `yolov8s-pose` at 640px could be substituted at the cost of some accuracy on far-away students.
- **MOG2 Warmup**: The motion detector requires ~500 frames to fully stabilize its background model, meaning it may over-trigger in the first 15 seconds of the feed.

---

## 6. Interview Questions Submission

**1. How would you scale this from one live camera to 500 cameras streaming at once? Where would it break first?**
To scale to 500 cameras, the monolithic script must be decoupled into a distributed microservices architecture using message brokers (e.g., Kafka) and scalable inference servers (e.g., NVIDIA Triton). The system would break first at the hardware inference bottleneck; running 500 parallel YOLO-Pose streams on a single node will exhaust VRAM and compute immediately. Resolving this requires lowering the processing frame rate (e.g., analyzing 3-5 FPS instead of 30), batching frames from multiple streams, and horizontally scaling across a GPU cluster.

**2. How would you avoid double-counting or losing track of a person if they briefly leave the camera's view?**
First, the tracking algorithm's memory buffer (e.g., `track_buffer` in BoT-SORT) must be extended to keep the track identity "alive" for several seconds during occlusions. For longer disappearances where the tracker drops the ID, a lightweight Re-Identification (Re-ID) mechanism is required. By extracting and storing an appearance embedding (like an HSV color histogram or a fast deep embedding) when a track exits, we can compare newly appearing tracks against recent exits and merge them if the similarity is high, preventing double-counting.

**3. How would you handle a camera feed that's consistently blurry or poor quality — flag it, skip it, or something else?**
Consistently poor feeds should not be blindly processed or silently skipped, as both pollute the database and hide hardware failures. The pipeline should calculate a rolling baseline of frame sharpness (using Laplacian variance) and automatically trigger a "Degraded Camera" alert to the IT/maintenance team if the median drops below an acceptable threshold. During this degraded state, the system should continue processing but explicitly tag all generated analytics in the database with a "Low Confidence" flag so downstream consumers know the data is unreliable until the lens is cleaned.

# Vision Intelligence
A computer vision-based classroom analytics system designed to track student attendance, posture, and room occupancy using fixed CCTV cameras. The system utilizes YOLOv8 for detection and pose estimation, coupled with a custom BoT-SORT implementation for robust tracking.

## Features

- **Occupancy & Attendance Tracking**: Monitors the number of students entering and exiting the classroom, and maintains a running attendance count.
- **Entry/Exit Zone Detection**: Automatically detects or allows manual configuration of entry/exit zones to properly log student movements.
- **Posture Classification**: Uses a hybrid approach (bounding box aspect ratio, normalized height, keypoints, and desk row alignment) to classify students as "Seated" or "Standing".
- **Robust Tracking in Classrooms**: A custom BoT-SORT configuration prevents losing tracks when students are occluded or sitting still behind desks for long periods.
- **Lightweight Re-Identification (Re-ID)**: Utilizes HSV color histograms to re-identify students who temporarily leave the camera's view, preventing double-counting.
- **Camera Quality & Motion Alerts**: Detects blurry frames (via adaptive Laplacian variance) and motion to ensure data quality.

## Requirements

Ensure you have Python 3 installed. The required libraries are listed in `requirements.txt`.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

### Main Dependencies

- `opencv-python`
- `numpy`
- `ultralytics`
- `PyYAML`
- `fpdf2` (For generating PDF reports)

## Usage

### 1. Running the Main Pipeline

Ensure you have the required YOLO model (e.g., `yolov8m-pose.pt`) and the video file (e.g., `video7.mp4`) in the project directory.

```bash
python main.py
```

**Controls during playback:**

- Click on the progress bar at the bottom to seek through the video.
- Press `L` to toggle the entry door zone side (Left, Right, None, Auto).
- Press `Q` to quit.

### 2. Generating PDF Reports

You can generate a PDF document with submission answers or reports using the provided script:

```bash
python generate_pdf.py
```

## How It Works

1. **Scene Understanding**: Initially, the system builds a static background and detects horizontal desk lines which help in posture classification.
2. **Detection & Tracking**: YOLOv8 detects bounding boxes and keypoints, and BoT-SORT tracks individuals across frames.
3. **Analytics**: The pipeline uses this data to update running metrics like entry/exits, total unique students, and current postures, which are overlaid on the video feed.

## Acknowledgements

- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Uses [BoT-SORT](https://arxiv.org/abs/2206.14651) for tracking.

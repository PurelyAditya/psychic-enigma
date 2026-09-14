import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import yaml

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ─ All tunable values in one place
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH   = "yolov8m-pose.pt"
VIDEO_PATH   = "video7.mp4"
WIN_NAME     = "Vision Intelligence"

# Detection
CONF         = 0.20    # Lowered to catch occluded / partially visible seated students
IOU          = 0.50    # Slightly tighter NMS — prevents merging two adjacent seated people
IMG_SIZE     = 1280

# Tracking (custom BoT-SORT — written to disk on startup)
TRACK_BUFFER = 120     # Frames to keep a track alive without detection (4 s @ 30fps)
                       # Default was 30 — not enough for classroom occlusions

# Entry / Exit  (zone-aware — exploits fixed camera)
# Adjust ENTRY_SIDE to "left" or "right" to match where the door is in your video.
# "auto" → pipeline detects it from where new tracks first appear.
ENTRY_SIDE   = "auto"
ENTRY_X_FRAC = 0.18   # Door zone = this fraction of frame width from the chosen edge

# Zone-differentiated exit timeouts
EXIT_DOOR_FRAMES     = 25   # Person in door zone + gone this long  → confirmed Exit
EXIT_INTERIOR_FRAMES = 100  # Person in interior  + gone this long  → ambiguous loss, remove silently

# Posture — hybrid (bbox height + keypoints + desk geometry)
KP_CONF             = 0.30
POSTURE_WINDOW      = 18   # Frames for majority-vote smoothing
POSTURE_SEATED_H    = 0.32  # box_h / frame_h below this → strong seated signal
POSTURE_STANDING_H  = 0.44  # box_h / frame_h above this → strong standing signal

# Scene understanding — background & desk detection
BG_ACCUM_FRAMES = 60   # Accumulate this many frames to build static background
DESK_MIN_LINE_W = 0.20  # Desk line must span this fraction of frame width

# Re-ID (color histogram — lightweight but sufficient for fixed camera)
REID_BINS    = 16
REID_TTL     = 200   # Frames to keep exited histogram
REID_THRESH  = 0.80

# Motion (MOG2 + temporal gate)
MOG2_HISTORY    = 500
MOG2_VAR_THRESH = 40
MOTION_MIN_PX   = 3000
MOTION_GATE_N   = 3    # Must detect motion for this many consecutive frames

# Blur (adaptive Laplacian)
BLUR_WINDOW  = 60
BLUR_FACTOR  = 0.40

# UI
SEEKBAR_H    = 22


# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM BOT-SORT CONFIG
#  Write a project-local botsort yaml with a much longer track buffer.
#  Default track_buffer=30 loses tracks in 1 second — fatal for a classroom
#  where students sit still for minutes behind desks.
# ═══════════════════════════════════════════════════════════════════════════════

def write_tracker_config():
    cfg = {
        "tracker_type":      "botsort",
        "track_high_thresh": CONF,
        "track_low_thresh":  0.10,
        "new_track_thresh":  CONF,
        "track_buffer":      TRACK_BUFFER,   # 120 — survive 4-s classroom occlusions
        "match_thresh":      0.85,
        "fuse_score":        True,
        # BoT-SORT specific (required by this ultralytics version)
        "gmc_method":        "sparseOptFlow", # global motion compensation
        "proximity_thresh":  0.5,
        "appearance_thresh": 0.8,
        "with_reid":         False,
        "model":             "auto",
    }
    path = "classroom_botsort.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE UNDERSTANDING — Background & Desk Detection
#  Exploit the FIXED camera: desks never move.
#  Compute a static background frame, then find horizontal desk lines via Hough.
#  These y-coordinates inform the posture classifier.
# ═══════════════════════════════════════════════════════════════════════════════

def build_background(frames):
    """Median of N frames → robust background estimate (removes moving people)."""
    stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def detect_desk_rows(bg_frame):
    """
    Detect horizontal desk lines from the static background using Canny + Hough.

    Why this works for a fixed classroom:
      Desks are horizontal surfaces that create consistent horizontal edges
      in the background image. Hough detects these as near-horizontal lines.
      Their y-coordinates are returned and used in posture classification:
      a person whose vertical midpoint aligns with a desk row is almost
      certainly seated.

    Returns: sorted list of y-coordinate floats, or [] if none found.
    """
    gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 25, 75)
    h, w  = gray.shape

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=70,
        minLineLength=int(w * DESK_MIN_LINE_W),
        maxLineGap=50
    )

    raw_ys = []
    if lines is not None:
        for seg in lines:
            # OpenCV 4 returns shape (N,1,4); OpenCV 5 returns (N,4) — handle both
            seg_flat = np.array(seg).flatten()
            x1, y1, x2, y2 = int(seg_flat[0]), int(seg_flat[1]), int(seg_flat[2]), int(seg_flat[3])
            # Keep only near-horizontal lines (slope < 10 °)
            if abs(y2 - y1) < abs(x2 - x1) * 0.18:
                avg_y = (y1 + y2) / 2.0
                # Desks occupy roughly middle 70 % of frame vertically
                if 0.15 * h < avg_y < 0.88 * h:
                    raw_ys.append(avg_y)

    if not raw_ys:
        return []

    # Cluster lines within 25 px of each other → one desk row
    raw_ys.sort()
    clusters = [[raw_ys[0]]]
    for y in raw_ys[1:]:
        if y - clusters[-1][-1] < 25:
            clusters[-1].append(y)
        else:
            clusters.append([y])

    return [float(np.mean(c)) for c in clusters]


# ═══════════════════════════════════════════════════════════════════════════════
#  POSTURE — Hybrid Classifier
#  Three independent signals, scored and summed.
#  Replaces the single-signal keypoint approach that failed in classrooms.
# ═══════════════════════════════════════════════════════════════════════════════

def classify_posture_hybrid(kps_xy, kps_conf, box, frame_h, desk_ys):
    """
    Hybrid 4-signal posture classifier for fixed classroom CCTV.
    
    ── Signal 1: Box Aspect Ratio ───────────────────────────────────────────
    Seated people are squarer (aspect > 0.45). Standing are tall & thin (aspect < 0.35).
    This handles perspective scaling naturally (far away standing people are still thin).
    
    ── Signal 2: Normalised bounding-box height ─────────────────────────────
    Contextualized by aspect ratio. A tall box is only standing if it is also thin.
    
    ── Signal 3: Keypoint body-completeness ─────────────────────────────────
    Checks hip-to-ankle distance to avoid penalizing seated people with visible feet.
    
    ── Signal 4: Desk-row alignment ─────────────────────────────────────────
    Aligns centroid with detected desks.
    """
    x1, y1, x2, y2 = box
    box_h  = max(y2 - y1, 1.0)
    box_w  = max(x2 - x1, 1.0)
    aspect = box_w / box_h
    norm_h = box_h / frame_h
    score  = 0

    # Signal 1: Aspect Ratio
    if aspect > 0.45:
        score += 3
    elif aspect < 0.35:
        score -= 3
    else:
        score += 1

    # Signal 2: Normalized Height
    if norm_h < POSTURE_SEATED_H:
        score += 2
    elif norm_h > POSTURE_STANDING_H and aspect < 0.40:
        score -= 2

    # Signal 3: Keypoints
    if kps_xy is not None and kps_conf is not None:
        hip_vis   = any(kps_conf[i] >= KP_CONF for i in [11, 12])
        knee_vis  = any(kps_conf[i] >= KP_CONF for i in [13, 14])
        ankle_vis = any(kps_conf[i] >= KP_CONF for i in [15, 16])

        if ankle_vis:
            hip_ys = [kps_xy[i][1] for i in [11, 12] if kps_conf[i] >= KP_CONF]
            ankle_ys = [kps_xy[i][1] for i in [15, 16] if kps_conf[i] >= KP_CONF]
            if hip_ys and ankle_ys:
                mean_hip = np.mean(hip_ys)
                mean_ankle = np.mean(ankle_ys)
                # If ankles are far below hips -> standing
                if (mean_ankle - mean_hip) > 0.4 * box_h:
                    score -= 3
                else:
                    score += 1 # Ankles visible but close to hips -> seated/legs folded
            else:
                score -= 3 # Ankles visible but hips not -> probably standing
        elif hip_vis and not knee_vis:
            score += 2
        elif hip_vis and knee_vis and not ankle_vis:
            score += 1

    # Signal 4: Desk Alignment
    if desk_ys:
        cy       = (y1 + y2) / 2.0
        min_dist = min(abs(cy - dy) for dy in desk_ys)
        if min_dist < 0.14 * frame_h:
            score += 2

    return "Seated" if score > 0 else "Standing"


# ═══════════════════════════════════════════════════════════════════════════════
#  RE-ID — Lightweight Color Histogram
# ═══════════════════════════════════════════════════════════════════════════════

def compute_color_hist(frame, box):
    """Normalized 2-D HSV histogram for a person bounding box."""
    x1, y1, x2, y2 = (int(v) for v in box)
    roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if roi.size == 0:
        return None
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None,
                        [REID_BINS, REID_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def hist_similarity(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY / EXIT — Zone-Aware
#  Exploit fixed camera: the door is always at the same edge of the frame.
#  Interior disappearances = occlusions, NOT exits.
# ═══════════════════════════════════════════════════════════════════════════════

def in_door_zone(cx, fw, side, frac):
    """Return True if x-coordinate cx is inside the door entry zone."""
    if side == "left":
        return cx < fw * frac
    if side == "right":
        return cx > fw * (1.0 - frac)
    return False   # "none" or "auto" not resolved yet


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_entry_zone(frame, side, frac):
    """Subtle semi-transparent orange strip marking the door zone."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    bar_h = h - SEEKBAR_H
    if side == "left":
        cv2.rectangle(overlay, (0, 0), (int(w * frac), bar_h), (0, 100, 220), -1)
    elif side == "right":
        cv2.rectangle(overlay, (int(w * (1 - frac)), 0), (w, bar_h), (0, 100, 220), -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    # Draw a bright border line at the zone boundary
    if side == "left":
        cv2.line(frame, (int(w * frac), 0), (int(w * frac), bar_h), (0, 160, 255), 2)
    elif side == "right":
        cv2.line(frame, (int(w * (1 - frac)), 0), (int(w * (1 - frac)), bar_h), (0, 160, 255), 2)


def draw_desk_lines(frame, desk_ys, seekbar_h):
    """Draw detected desk row lines as thin cyan horizontals."""
    w = frame.shape[1]
    for dy in desk_ys:
        y = int(dy)
        cv2.line(frame, (0, y), (w, y), (200, 220, 0), 1)


def draw_overlay(frame, metrics):
    """Compact stats panel pinned to upper-right corner."""
    fh, fw = frame.shape[:2]
    lh, bw = 20, 360
    x0, y0 = fw - bw - 10, 25
    cv2.rectangle(frame, (x0, 10), (fw - 10, 15 + len(metrics) * lh), (0, 0, 0), -1)
    for text in metrics:
        hi    = any(k in text for k in ("detected", "Blurry", "Occupied", "Entry", "Exit"))
        color = (0, 255, 255) if hi else (255, 255, 255)
        cv2.putText(frame, text, (x0 + 8, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)
        y0 += lh


def draw_seekbar(frame, cur, total, fps):
    """Green progress bar + playhead + MM:SS timestamp at the bottom of frame."""
    fh, fw  = frame.shape[:2]
    ratio   = cur / max(total - 1, 1)
    filled  = int(ratio * fw)
    bar_y   = fh - SEEKBAR_H
    fps     = fps or 25

    cv2.rectangle(frame, (0, bar_y), (fw, fh), (30, 30, 30), -1)
    cv2.rectangle(frame, (0, bar_y), (filled, fh), (0, 200, 100), -1)
    cv2.circle(frame, (filled, bar_y + SEEKBAR_H // 2), 8, (255, 255, 255), -1)

    cs = int(cur / fps);   ts = int(total / fps)
    cv2.putText(frame, f"{cs//60:02d}:{cs%60:02d} / {ts//60:02d}:{ts%60:02d}",
                (10, fh - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Press 'L' hint
    cv2.putText(frame, "L=toggle door zone | Q=quit",
                (fw - 260, fh - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def clear_all(dicts, sets):
    for d in dicts: d.clear()
    for s in sets:  s.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    tracker_cfg = write_tracker_config()
    model       = YOLO(MODEL_PATH)
    cap         = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    # ── Seekbar ──────────────────────────────────────────────────────────────
    seek_state    = {"target_frame": -1}
    cb_registered = [False]

    def on_mouse(event, x, y, flags, param):
        fh, fw = param
        if y >= fh - SEEKBAR_H:
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE) \
               and (flags & cv2.EVENT_FLAG_LBUTTON):
                ratio = max(0.0, min(1.0, x / fw))
                seek_state["target_frame"] = int(ratio * (total_frames - 1))

    # ── Scene state (fixed camera — computed once) ────────────────────────────
    bg_accum       = []
    background     = None
    desk_ys        = []
    scene_ready    = False

    # Entry zone auto-detection
    entry_side     = ENTRY_SIDE   # "left", "right", "auto", "none"
    first_cx_log   = []           # first-appearance x-coords for auto-detection
    AUTO_DETECT_N  = 8

    # ── Tracking state ────────────────────────────────────────────────────────
    unique_ids     = set()
    # active_tracks: can_id → {"last_frame": int, "zone": str, "absent": int}
    active_tracks  = {}
    exited_hists   = {}   # can_id → {"hist": arr|None, "frame_exited": int}
    live_hists     = {}   # can_id → current histogram
    id_remap       = {}   # raw_tracker_id → canonical_id
    posture_votes  = {}   # can_id → deque[str]
    posture_labels = {}   # can_id → str

    all_dicts = [active_tracks, exited_hists, live_hists, id_remap,
                 posture_votes, posture_labels]
    all_sets  = [unique_ids]

    # ── Analytics ─────────────────────────────────────────────────────────────
    last_event = "None"
    event_cd   = 0
    frame_count = 0

    # ── Motion (MOG2 + temporal gate) ────────────────────────────────────────
    bg_sub       = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESH, detectShadows=True)
    morph_k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_hist  = deque(maxlen=MOTION_GATE_N)

    # ── Blur (adaptive Laplacian) ─────────────────────────────────────────────
    recent_blur  = deque(maxlen=BLUR_WINDOW)

    # ══════════════════════════════════════════════════════════════════════════
    while cap.isOpened():

        # Seek jump
        if seek_state["target_frame"] >= 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, seek_state["target_frame"])
            frame_count = seek_state["target_frame"]
            seek_state["target_frame"] = -1
            clear_all(all_dicts, all_sets)
            bg_accum.clear()
            scene_ready = False
            desk_ys.clear()
            first_cx_log.clear()
            entry_side = ENTRY_SIDE
            motion_hist.clear()
            recent_blur.clear()
            background = None

        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        fh, fw = frame.shape[:2]

        if not cb_registered[0]:
            cv2.setMouseCallback(WIN_NAME, on_mouse, (fh, fw))
            cb_registered[0] = True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Background accumulation (first BG_ACCUM_FRAMES frames) ───────────
        if not scene_ready:
            bg_accum.append(frame.copy())
            if len(bg_accum) >= BG_ACCUM_FRAMES:
                background  = build_background(bg_accum)
                desk_ys     = detect_desk_rows(background)
                scene_ready = True
            # Show a build-progress banner and skip heavy inference until ready
            pct = int(100 * len(bg_accum) / BG_ACCUM_FRAMES)
            cv2.putText(frame, f"Building scene model: {pct}%",
                        (fw // 2 - 140, fh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            draw_seekbar(frame, int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                         total_frames, fps)
            cv2.imshow(WIN_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # ── 1. Adaptive Blur Detection ────────────────────────────────────────
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        recent_blur.append(lap_var)
        blur_thresh  = max(50.0, float(np.median(recent_blur)) * BLUR_FACTOR)
        frame_quality = "Clear" if lap_var > blur_thresh else "Blurry - flag for review"

        # ── 2. MOG2 Motion + Temporal Gate ───────────────────────────────────
        fg = bg_sub.apply(frame)
        fg = np.where(fg == 255, 255, 0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph_k)
        motion_hist.append(cv2.countNonZero(fg) > MOTION_MIN_PX)
        # Require MOTION_GATE_N consecutive positive frames to avoid flicker
        motion_status = "Motion detected" if all(motion_hist) and len(motion_hist) == MOTION_GATE_N \
                        else "No motion"

        # ── 3. Detection + Tracking ───────────────────────────────────────────
        results = model.track(
            frame, persist=True, tracker=tracker_cfg,
            classes=[0], conf=CONF, iou=IOU, imgsz=IMG_SIZE, verbose=False
        )

        seated_count   = 0
        standing_count = 0
        current_raw    = set()

        has_kps = hasattr(results[0], "keypoints") and results[0].keypoints is not None
        kps_xy   = results[0].keypoints.xy.cpu().numpy()   if has_kps else None
        kps_conf = results[0].keypoints.conf.cpu().numpy() if has_kps else None

        if results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            raw_ids = results[0].boxes.id.int().cpu().tolist()

            for i, (box, raw_id) in enumerate(zip(boxes, raw_ids)):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2.0
                current_raw.add(raw_id)

                # Resolve canonical ID (may be remapped by Re-ID)
                can_id = id_remap.get(raw_id, raw_id)

                # ── Entry + Lightweight Re-ID ─────────────────────────────────
                if can_id not in unique_ids:
                    hist = compute_color_hist(frame, box)
                    best_sim, best_oid = 0.0, None
                    for oid, rec in exited_hists.items():
                        if frame_count - rec["frame_exited"] < REID_TTL:
                            s = hist_similarity(hist, rec["hist"])
                            if s > best_sim:
                                best_sim, best_oid = s, oid

                    if best_sim >= REID_THRESH and best_oid is not None:
                        # Re-identified — merge with old ID, do NOT count as new entry
                        id_remap[raw_id] = best_oid
                        can_id = best_oid
                        del exited_hists[best_oid]
                    else:
                        unique_ids.add(can_id)
                        # Auto-detect door side from clustering first appearances
                        if entry_side == "auto":
                            first_cx_log.append(cx)
                            if len(first_cx_log) >= AUTO_DETECT_N:
                                avg = np.mean(first_cx_log)
                                entry_side = "left" if avg < fw * 0.5 else "right"
                        last_event = f"Entry (ID:{can_id})"
                        event_cd   = 55

                # Determine current zone for this person
                zone = "door" if in_door_zone(cx, fw, entry_side, ENTRY_X_FRAC) \
                               else "interior"

                # Update / initialise track record
                if can_id in active_tracks:
                    active_tracks[can_id]["last_frame"] = frame_count
                    active_tracks[can_id]["zone"]       = zone
                    active_tracks[can_id]["absent"]     = 0
                else:
                    active_tracks[can_id] = {
                        "last_frame": frame_count,
                        "zone":       zone,
                        "absent":     0,
                    }

                live_hists[can_id] = compute_color_hist(frame, box)

                # ── Posture (hybrid) ──────────────────────────────────────────
                kxy_i = kps_xy[i]   if (has_kps and i < len(kps_xy))   else None
                kco_i = kps_conf[i] if (has_kps and i < len(kps_conf)) else None

                raw_posture = classify_posture_hybrid(
                    kxy_i, kco_i, (x1, y1, x2, y2), fh, desk_ys
                )

                if can_id not in posture_votes:
                    posture_votes[can_id]  = deque(maxlen=POSTURE_WINDOW)
                    posture_labels[can_id] = "Seated"

                posture_votes[can_id].append(raw_posture)
                v = list(posture_votes[can_id])
                posture_labels[can_id] = \
                    "Seated" if v.count("Seated") >= len(v) * 0.55 else "Standing"

                posture   = posture_labels[can_id]
                box_color = (0, 220, 0) if posture == "Seated" else (0, 140, 255)
                seated_count   += (posture == "Seated")
                standing_count += (posture == "Standing")

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                cv2.putText(frame, f"ID:{can_id} {posture}",
                            (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1)

        # ── Exit detection (zone-differentiated) ─────────────────────────────
        # KEY INSIGHT (fixed camera):
        #   Door zone disappearance → person likely walked out → shorter timeout
        #   Interior disappearance  → person is occluded       → much longer timeout
        # This prevents false "Exit" events when students momentarily hide behind
        # each other or behind a pillar.
        current_can = {id_remap.get(r, r) for r in current_raw}
        to_remove   = []

        for cid, info in active_tracks.items():
            if cid not in current_can:
                info["absent"] += 1
                timeout = EXIT_DOOR_FRAMES if info["zone"] == "door" \
                          else EXIT_INTERIOR_FRAMES
                if info["absent"] >= timeout:
                    if info["zone"] == "door":
                        last_event = f"Exit (ID:{cid})"
                        event_cd   = 55
                    # Store histogram for future Re-ID regardless of zone
                    exited_hists[cid] = {
                        "hist":          live_hists.pop(cid, None),
                        "frame_exited":  frame_count,
                    }
                    to_remove.append(cid)

        for cid in to_remove:
            active_tracks.pop(cid, None)
            posture_votes.pop(cid, None)
            posture_labels.pop(cid, None)

        # Prune stale exited histograms
        stale = [k for k, v in exited_hists.items()
                 if frame_count - v["frame_exited"] > REID_TTL]
        for k in stale:
            del exited_hists[k]

        if event_cd > 0:
            event_cd -= 1
        else:
            last_event = "None"

        running_attendance = len(active_tracks)
        occupancy          = "Occupied" if running_attendance > 0 else "Empty"
        total_unique       = len(unique_ids)

        # ── Draw scene understanding overlays ────────────────────────────────
        if entry_side in ("left", "right"):
            draw_entry_zone(frame, entry_side, ENTRY_X_FRAC)
        if desk_ys:
            draw_desk_lines(frame, desk_ys, SEEKBAR_H)

        # ── Metrics overlay ───────────────────────────────────────────────────
        entry_label = f"Door: {entry_side}" if entry_side != "auto" else "Door: detecting..."
        metrics = [
            f"Frame quality: {frame_quality}",
            f"Motion: {motion_status}",
            f"Room Status: {occupancy}  [{entry_label}]",
            f"Running attendance: {running_attendance} present",
            f"Total unique entered: {total_unique} entries",
            f"Posture: {seated_count} Seated, {standing_count} Standing",
            f"Desk rows detected: {len(desk_ys)}",
            f"Event: {last_event}",
        ]
        draw_overlay(frame, metrics)

        # ── Seekbar ───────────────────────────────────────────────────────────
        cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        draw_seekbar(frame, cur_pos, total_frames, fps)

        cv2.imshow(WIN_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("l"):
            # Manually cycle door zone: left → right → none → left
            cycle = {"left": "right", "right": "none", "none": "left", "auto": "left"}
            entry_side = cycle.get(entry_side, "left")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

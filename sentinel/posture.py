"""
SENTINEL AI — Human Posture Estimation Module
Analyzes human body keypoints and geometric relationships for posture classification:
1. STANDING
2. SITTING
3. BENDING
4. LYING_DOWN (with temporal confirmation for POSSIBLE_FALL alerts)
5. UNKNOWN (insufficient keypoints or low confidence)

Privacy & Safety Constraints:
- No facial recognition or identity tracking (facial keypoints 0-4 are discarded).
- Posture is NOT treated as a criminal or security threat (pose confidence is decoupled from behaviour risk).
- Possible falls require sustained temporal confirmation before dispatching an alert.
"""

import math
import time
import os
import cv2
import numpy as np

from sentinel.config import SentinelConfig
from sentinel.detection import is_person, calculate_iou


# COCO-17 Keypoint Indices
KPT_NOSE = 0
KPT_LEFT_EYE = 1
KPT_RIGHT_EYE = 2
KPT_LEFT_EAR = 3
KPT_RIGHT_EAR = 4
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_ELBOW = 7
KPT_RIGHT_ELBOW = 8
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
KPT_LEFT_HIP = 11
KPT_RIGHT_HIP = 12
KPT_LEFT_KNEE = 13
KPT_RIGHT_KNEE = 14
KPT_LEFT_ANKLE = 15
KPT_RIGHT_ANKLE = 16

# Body-only keypoints used for geometry and rendering (excludes facial landmarks 0-4)
BODY_KEYPOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Skeleton connections between body keypoints
SKELETON_CONNECTIONS = [
    (5, 7), (7, 9),    # Left arm: shoulder -> elbow -> wrist
    (6, 8), (8, 10),   # Right arm: shoulder -> elbow -> wrist
    (5, 6),            # Shoulders
    (5, 11), (6, 12),  # Torso: shoulders -> hips
    (11, 12),          # Hips
    (11, 13), (13, 15),# Left leg: hip -> knee -> ankle
    (12, 14), (14, 16) # Right leg: hip -> knee -> ankle
]

# Posture color scheme for HUD visualization (BGR)
POSTURE_COLORS = {
    "STANDING": (0, 255, 136),      # Teal / Green
    "SITTING": (255, 200, 0),       # Cyan / Light Blue
    "BENDING": (0, 215, 255),       # Amber / Gold
    "LYING_DOWN": (60, 60, 255),    # Red-Orange / Coral
    "POSSIBLE_FALL": (0, 0, 255),   # Bright Red
    "UNKNOWN": (140, 140, 140),     # Muted Gray
}


def load_pose_model(base_dir=None, model_name=None):
    """
    Load lightweight YOLO pose model (YOLOv8n-pose).
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_name is None:
        model_name = getattr(SentinelConfig, "POSTURE_MODEL_PATH", "yolov8n-pose.pt")

    candidate_paths = [
        os.path.join(base_dir, model_name),
        os.path.join(base_dir, "models", model_name),
        model_name
    ]

    chosen_path = None
    for p in candidate_paths:
        if os.path.isfile(p):
            chosen_path = p
            break

    if chosen_path is None:
        chosen_path = model_name  # let ultralytics handle or search default

    try:
        from ultralytics import YOLO
        model = YOLO(chosen_path)
        return model
    except Exception as e:
        print(f"[SENTINEL] Warning: Failed to load pose model: {e}")
        return None


def calculate_joint_angle(p1, p2, p3):
    """
    Calculate the interior angle at joint p2 formed by vectors p2->p1 and p2->p3 (in degrees).
    p1, p2, p3: (x, y) coordinates.
    Returns: angle in [0.0, 180.0].
    """
    v1x = p1[0] - p2[0]
    v1y = p1[1] - p2[1]
    v2x = p3[0] - p2[0]
    v2y = p3[1] - p2[1]

    mag1 = math.hypot(v1x, v1y)
    mag2 = math.hypot(v2x, v2y)
    if mag1 < 1e-4 or mag2 < 1e-4:
        return 180.0

    dot = (v1x * v2x + v1y * v2y) / (mag1 * mag2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def calculate_line_inclination(p1, p2):
    """
    Calculate the inclination angle of segment p1->p2 with the horizontal axis (in degrees).
    0 deg = horizontal, 90 deg = vertical.
    """
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    return math.degrees(math.atan2(dy, max(dx, 1e-4)))


def _extract_point(kpts, idx, min_conf=0.35):
    """Safely extract (x, y, conf) for a keypoint index. Returns None if invalid or low confidence."""
    if kpts is None or len(kpts) <= idx:
        return None
    pt = kpts[idx]
    if len(pt) >= 3:
        x, y, c = float(pt[0]), float(pt[1]), float(pt[2])
    elif len(pt) == 2:
        x, y, c = float(pt[0]), float(pt[1]), 1.0
    else:
        return None
    if c < min_conf:
        return None
    return (x, y, c)


def _midpoint(pt1, pt2):
    """Return average coordinates of two points if available."""
    if pt1 and pt2:
        return ((pt1[0] + pt2[0]) / 2.0, (pt1[1] + pt2[1]) / 2.0), (pt1[2] + pt2[2]) / 2.0
    if pt1:
        return (pt1[0], pt1[1]), pt1[2]
    if pt2:
        return (pt2[0], pt2[1]), pt2[2]
    return None, 0.0


def classify_posture(keypoints, bbox=None, min_kpt_conf=0.35):
    """
    Deterministic posture classification from 2D body keypoints and optional bounding box.
    
    Required posture classes:
    - STANDING
    - SITTING
    - BENDING
    - LYING_DOWN
    - UNKNOWN
    
    Returns:
        (posture_label: str, pose_confidence: float)
    """
    if keypoints is None or len(keypoints) < 17:
        return "UNKNOWN", 0.0

    # Extract body keypoints (excluding facial landmarks)
    ls = _extract_point(keypoints, KPT_LEFT_SHOULDER, min_kpt_conf)
    rs = _extract_point(keypoints, KPT_RIGHT_SHOULDER, min_kpt_conf)
    lh = _extract_point(keypoints, KPT_LEFT_HIP, min_kpt_conf)
    rh = _extract_point(keypoints, KPT_RIGHT_HIP, min_kpt_conf)
    lk = _extract_point(keypoints, KPT_LEFT_KNEE, min_kpt_conf)
    rk = _extract_point(keypoints, KPT_RIGHT_KNEE, min_kpt_conf)
    la = _extract_point(keypoints, KPT_LEFT_ANKLE, min_kpt_conf)
    ra = _extract_point(keypoints, KPT_RIGHT_ANKLE, min_kpt_conf)

    # Compute overall body keypoint confidence
    body_confs = []
    for idx in BODY_KEYPOINTS:
        if idx < len(keypoints) and len(keypoints[idx]) >= 3:
            body_confs.append(float(keypoints[idx][2]))
        elif idx < len(keypoints):
            body_confs.append(1.0)
    avg_conf = float(sum(body_confs) / len(body_confs)) if body_confs else 0.0

    # Torso anchors are essential: at least one shoulder and one hip must be visible
    mid_shoulder, s_conf = _midpoint(ls, rs)
    mid_hip, h_conf = _midpoint(lh, rh)

    if mid_shoulder is None or mid_hip is None:
        return "UNKNOWN", round(avg_conf, 2)

    # Torso geometry
    # Image coordinates: y increases downward. In an upright posture, y_shoulder < y_hip
    sx, sy = mid_shoulder
    hx, hy = mid_hip
    dx_torso = hx - sx
    dy_torso = hy - sy # positive when shoulder is above hip
    torso_length = math.hypot(dx_torso, dy_torso)
    torso_inclination = calculate_line_inclination(mid_shoulder, mid_hip) # 0 = flat, 90 = vertical

    # Bounding box geometry if provided
    bbox_aspect_ratio = 1.0 # width / height
    bbox_hwr = 1.0          # height / width
    if bbox is not None and len(bbox) == 4:
        bw = max(1.0, float(bbox[2] - bbox[0]))
        bh = max(1.0, float(bbox[3] - bbox[1]))
        bbox_aspect_ratio = bw / bh
        bbox_hwr = bh / bw

    mid_knee, _ = _midpoint(lk, rk)
    mid_ankle, _ = _midpoint(la, ra)

    # Check if legs provide vertical support below hips
    has_vertical_legs = False
    if mid_ankle is not None and (mid_ankle[1] - hy) > 0.5 * max(torso_length, 20.0):
        has_vertical_legs = True
    elif mid_knee is not None and (mid_knee[1] - hy) > 0.4 * max(torso_length, 20.0):
        has_vertical_legs = True

    # ─────────────────────────────────────────────────────────────────────────
    # 1. LYING_DOWN / POSSIBLE_FALL
    # Characteristic:
    # - Torso is predominantly horizontal (inclination < 35 deg) without upright leg support
    # - OR aspect ratio is distinctly horizontal (bbox width > height, aspect ratio > 1.25)
    #   combined with non-upright torso (inclination < 50 deg) and no vertical leg support
    # - All major joints are near the same vertical span along the floor/ground
    # ─────────────────────────────────────────────────────────────────────────
    is_lying = False
    if not has_vertical_legs:
        if torso_inclination < 35.0:
            is_lying = True
        elif bbox_aspect_ratio > 1.25 and torso_inclination < 50.0:
            is_lying = True
        elif mid_knee is not None and mid_ankle is not None:
            # Check if entire body spans horizontally along similar y level
            y_span = max(sy, hy, mid_knee[1], mid_ankle[1]) - min(sy, hy, mid_knee[1], mid_ankle[1])
            x_span = max(sx, hx, mid_knee[0], mid_ankle[0]) - min(sx, hx, mid_knee[0], mid_ankle[0])
            if x_span > 1.25 * max(y_span, 1.0) and torso_inclination < 45.0:
                is_lying = True

    if is_lying:
        pose_conf = round(max(avg_conf, (s_conf + h_conf) / 2.0), 2)
        return "LYING_DOWN", pose_conf

    # ─────────────────────────────────────────────────────────────────────────
    # 2. BENDING
    # Characteristic:
    # - Torso is significantly tilted forward/downward (inclination <= 58 deg)
    # - Hips are elevated above ankles/knees (person is supported on legs, not lying)
    # - Hip flexion angle (shoulder-hip-knee) is sharp (< 135 deg) while legs are standing
    # ─────────────────────────────────────────────────────────────────────────
    is_bending = False

    # Check hip flexion angle if knee is visible
    hip_angles = []
    if ls and lh and lk:
        hip_angles.append(calculate_joint_angle(ls[:2], lh[:2], lk[:2]))
    if rs and rh and rk:
        hip_angles.append(calculate_joint_angle(rs[:2], rh[:2], rk[:2]))

    # Check knee flexion
    knee_angles = []
    if lh and lk and la:
        knee_angles.append(calculate_joint_angle(lh[:2], lk[:2], la[:2]))
    if rh and rk and ra:
        knee_angles.append(calculate_joint_angle(rh[:2], rk[:2], ra[:2]))

    avg_hip_angle = sum(hip_angles) / len(hip_angles) if hip_angles else None
    avg_knee_angle = sum(knee_angles) / len(knee_angles) if knee_angles else None

    # Condition A: Torso is inclined <= 58 degrees with vertical legs
    if torso_inclination <= 58.0 and has_vertical_legs:
        is_bending = True
    elif 25.0 <= torso_inclination <= 58.0:
        if bbox is not None and bbox_hwr >= 0.9:
            is_bending = True

    # Condition B: Hip angle is bent sharply while legs are extended
    if avg_hip_angle is not None and avg_hip_angle < 135.0:
        if avg_knee_angle is None or avg_knee_angle > 130.0:
            if torso_inclination < 65.0:
                is_bending = True

    if is_bending:
        pose_conf = round(max(avg_conf, (s_conf + h_conf) / 2.0), 2)
        return "BENDING", pose_conf

    # ─────────────────────────────────────────────────────────────────────────
    # 3. SITTING
    # Characteristic:
    # - Torso is largely upright (inclination >= 55 deg, shoulder above hip)
    # - Thighs (hip to knee) are horizontal or knee angle is bent (~70 to 130 deg)
    # - Vertical displacement between hip and knee is small compared to torso length
    # ─────────────────────────────────────────────────────────────────────────
    is_sitting = False

    # Check thigh inclination angles
    thigh_inclinations = []
    if lh and lk:
        thigh_inclinations.append(calculate_line_inclination(lh[:2], lk[:2]))
    if rh and rk:
        thigh_inclinations.append(calculate_line_inclination(rh[:2], rk[:2]))

    avg_thigh_inc = sum(thigh_inclinations) / len(thigh_inclinations) if thigh_inclinations else None

    if torso_inclination >= 55.0 and sy < hy:
        # Case 1: Knee flexion angle in seated range
        if avg_knee_angle is not None and 65.0 <= avg_knee_angle <= 135.0:
            is_sitting = True
        # Case 2: Thigh is significantly horizontal
        elif avg_thigh_inc is not None and avg_thigh_inc < 52.0:
            is_sitting = True
        # Case 3: Hip flexion angle bent in seated posture
        elif avg_hip_angle is not None and 70.0 <= avg_hip_angle <= 135.0:
            is_sitting = True
        # Case 4: Knee is at a similar vertical height to hip while torso is upright
        elif mid_knee is not None and abs(mid_knee[1] - hy) < 0.55 * torso_length:
            is_sitting = True

    if is_sitting:
        pose_conf = round(max(avg_conf, (s_conf + h_conf) / 2.0), 2)
        return "SITTING", pose_conf

    # ─────────────────────────────────────────────────────────────────────────
    # 4. STANDING
    # Characteristic:
    # - Torso is upright (inclination >= 58 deg, shoulder above hip)
    # - Legs are extended vertically (hips above knees, knees above ankles)
    # - Knee angle > 140 deg (straight legs)
    # - Bounding box is typically vertical (hwr >= 1.25)
    # ─────────────────────────────────────────────────────────────────────────
    is_standing = False
    if torso_inclination >= 58.0 and sy < hy:
        if avg_knee_angle is not None and avg_knee_angle >= 140.0:
            is_standing = True
        elif avg_thigh_inc is not None and avg_thigh_inc >= 55.0:
            is_standing = True
        elif mid_ankle is not None and mid_knee is not None:
            if mid_ankle[1] > mid_knee[1] > hy:
                is_standing = True
        elif bbox_hwr >= 1.25:
            is_standing = True
        else:
            # Upright torso without contradictory leg evidence
            is_standing = True

    if is_standing:
        pose_conf = round(max(avg_conf, (s_conf + h_conf) / 2.0), 2)
        return "STANDING", pose_conf

    return "UNKNOWN", round(avg_conf, 2)


def estimate_postures(frame, tracked_entities, pose_model, conf_thresh=0.35, current_time=None):
    """
    Execute posture estimation on the current frame, associate keypoints and
    postures with active PERSON tracks, and manage temporal fall confirmation.
    
    Args:
        frame: BGR image frame
        tracked_entities: dictionary of active TrackState objects
        pose_model: YOLO pose model instance
        conf_thresh: minimum confidence threshold
        current_time: optional timestamp
        
    Returns:
        list of new POSSIBLE_FALL alert event dictionaries
    """
    if current_time is None:
        current_time = time.time()

    new_fall_alerts = []

    # Filter active human tracks
    person_tracks = {
        k: v for k, v in tracked_entities.items()
        if is_person(v["class_name"]) and v.get("missed_frames", 0) <= 5
    }

    if not person_tracks or pose_model is None or frame is None or frame.size == 0:
        return new_fall_alerts

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        results = pose_model(frame, conf=conf_thresh, device=device, verbose=False)
    except Exception as e:
        print(f"[SENTINEL] Pose inference warning: {e}")
        return new_fall_alerts

    # Parse detected poses
    pose_candidates = []
    for r in results:
        if not hasattr(r, "keypoints") or r.keypoints is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        kpts_data = r.keypoints.data.cpu().numpy()
        for i in range(len(kpts_data)):
            p_box = boxes[i] if i < len(boxes) else None
            p_kpts = kpts_data[i]
            pose_candidates.append({
                "bbox": p_box,
                "keypoints": p_kpts
            })

    fall_persist_frames = getattr(SentinelConfig, "POSTURE_FALL_CONFIRMATION_FRAMES", 25)
    fall_cooldown = getattr(SentinelConfig, "POSTURE_FALL_COOLDOWN", 15.0)

    # Associate each person track with the best-matching pose detection
    for t_key, track in person_tracks.items():
        t_box = track["bbox"]
        best_cand = None
        best_iou = 0.0

        for cand in pose_candidates:
            if cand["bbox"] is not None:
                iou = calculate_iou(t_box, cand["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_cand = cand

        # Match threshold IoU >= 0.20
        if best_cand is not None and best_iou >= 0.20:
            kpts = best_cand["keypoints"]
            posture, pose_conf = classify_posture(kpts, bbox=t_box, min_kpt_conf=conf_thresh)
            track["keypoints"] = kpts
            track["posture"] = posture
            track["pose_confidence"] = pose_conf
        else:
            # If no fresh pose detected, retain previous posture if recent, else UNKNOWN
            if "posture" not in track:
                track["posture"] = "UNKNOWN"
                track["pose_confidence"] = 0.0
            kpts = track.get("keypoints", None)

        current_posture = track.get("posture", "UNKNOWN")

        # ── Temporal Fall Confirmation ──
        if current_posture == "LYING_DOWN":
            track["consecutive_fall_frames"] = track.get("consecutive_fall_frames", 0) + 1
            if track["consecutive_fall_frames"] >= fall_persist_frames:
                last_fall = track.get("last_fall_alert_time", 0.0)
                if current_time - last_fall >= fall_cooldown:
                    track["last_fall_alert_time"] = current_time
                    track.setdefault("active_threats", set()).add("POSSIBLE_FALL")

                    new_fall_alerts.append({
                        "key": t_key,
                        "track_id": track["track_id"],
                        "class_name": track["class_name"],
                        "behavior": "POSSIBLE_FALL",
                        "event_name": "\u26A0 POSSIBLE FALL / PERSON DOWN",
                        "severity": "MEDIUM",
                        "score": 50,
                        "evidence": {
                            "posture": current_posture,
                            "pose_confidence": track.get("pose_confidence", 0.0),
                            "consecutive_fall_frames": track["consecutive_fall_frames"],
                        },
                        "description": f"PERSON #{track['track_id']} — \u26A0 POSSIBLE FALL DETECTED (Assistance Recommended)"
                    })
        else:
            track["consecutive_fall_frames"] = 0
            if "active_threats" in track:
                track["active_threats"].discard("POSSIBLE_FALL")

    return new_fall_alerts


def draw_pose_skeleton(frame, keypoints, posture="STANDING", min_conf=0.35):
    """
    Render privacy-preserving body skeleton (excluding facial keypoints 0-4).
    Lines and joints are drawn cleanly without obscuring bounding boxes.
    """
    if frame is None or keypoints is None or len(keypoints) < 17:
        return

    color = POSTURE_COLORS.get(posture, POSTURE_COLORS["UNKNOWN"])

    # Draw body bone connections
    for idx1, idx2 in SKELETON_CONNECTIONS:
        pt1 = _extract_point(keypoints, idx1, min_conf)
        pt2 = _extract_point(keypoints, idx2, min_conf)
        if pt1 is not None and pt2 is not None:
            p1 = (int(pt1[0]), int(pt1[1]))
            p2 = (int(pt2[0]), int(pt2[1]))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

    # Draw body joints
    for idx in BODY_KEYPOINTS:
        pt = _extract_point(keypoints, idx, min_conf)
        if pt is not None:
            px = int(pt[0])
            py = int(pt[1])
            cv2.circle(frame, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, color, 1, cv2.LINE_AA)

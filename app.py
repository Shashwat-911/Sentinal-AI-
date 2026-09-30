"""
SENTINEL AI — Public Safety Surveillance System
Full Tkinter desktop application for AI-based abandoned bag & loitering detection.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import time
import re
import subprocess
import math

import yaml
import cv2
import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLE & CONFIGURATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BG       = "#0a0a0a"
PANEL    = "#111111"
CARD     = "#1a1a1a"
ACCENT   = "#00ff88"
DANGER   = "#ff3333"
WARNING  = "#ffaa00"
TEXT     = "#e0e0e0"
MUTED    = "#666666"

FONT_HEADING = ("Consolas", 16, "bold")
FONT_BODY    = ("Consolas", 11)
FONT_SMALL   = ("Consolas", 9)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class SentinelConfig:
    # Tunable behavior thresholds
    LOITER_THRESHOLD = 20.0             # seconds
    LOITER_SPATIAL_RATIO = 0.15         # fraction of frame diagonal for spatial spread
    STATIONARY_THRESHOLD = 15.0         # seconds before a bag is considered unattended
    STATIONARY_DISPLACEMENT_RATIO = 0.02  # fraction of frame diagonal for stationary check
    SPEED_RATIO_THRESHOLD = 2.5         # ratio compared to historical speed
    SPEED_ABSOLUTE_MIN_RATIO = 0.10     # fraction of frame diagonal per second (minimum speed floor)
    PROXIMITY_RATIO = 0.30              # fraction of frame diagonal for person-object proximity
    ALERT_COOLDOWN = 10.0               # seconds between alerts for same entity + behavior
    TRACK_TIMEOUT = 3.0                 # seconds to keep missing tracks
    
    # Tunable tracker parameters (Phase 2)
    TRACK_MATCH_THRESHOLD = 100.0       # max pixel distance for centroid matching
    MAX_MISSED_FRAMES = 30              # frames to retain missing track (~1.0s at 30fps)
    MAX_TRACK_HISTORY = 60              # max points in trajectory history
    BBOX_SMOOTHING_ALPHA = 0.75         # EMA factor for bounding box smoothing (0.75 curr + 0.25 prev)
    VELOCITY_SMOOTHING_ALPHA = 0.30     # EMA factor for image-plane velocity (0.30 curr + 0.70 prev)
    
    # Luggage False-Positive Filter & Confirmation parameters
    DEBUG_DETECTION = False             # Set True to log/visualize raw vs suppressed detections
    BAG_CONFIRMATION_FRAMES = 3         # Frames a bag candidate must persist to be confirmed
    MAX_BAG_FRAME_AREA_RATIO = 0.70     # Discard boxes covering > 70% of frame (whole-scene artifact)
    PERSON_BAG_CONTAINMENT_THRESH = 0.85 # Max fraction of bag allowed inside person body
    PERSON_HEAD_ZONE_RATIO = 0.30       # Top fraction of person box considered head/face
    
    # Pacing / repeated movement thresholds (normalized to frame diagonal)
    PACING_MIN_MOVEMENT_RATIO = 0.02    # minimum movement per segment (was 0.006)
    PACING_MAX_NET_DISP_RATIO = 0.30    # maximum net displacement
    PACING_MIN_PATH_RATIO = 0.60        # minimum cumulative path length
    PACING_COMPARISON_WINDOW = 1.0      # seconds to look back for direction comparison
    
    # Restricted zone normalized polygon coordinates (0.0 to 1.0 relative to width/height)
    RESTRICTED_ZONE_ENABLED = True
    RESTRICTED_ZONE_COORDS = [
        (0.55, 0.15),
        (0.95, 0.15),
        (0.95, 0.70),
        (0.55, 0.70)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITY & DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def load_dual_models(base_dir):
    """
    Load two models:
    1. bag_model: best.pt from models/train/weights/best.pt or models/best.pt
    2. person_model: yolov9s.pt from project root (COCO pretrained)
    Returns (bag_model, person_model, status_string)
    """
    import torch
    from ultralytics import YOLO

    # Limit CPU openmp thread loops to avoid 100% CPU overheating
    if hasattr(torch, "set_num_threads"):
        try:
            torch.set_num_threads(min(4, os.cpu_count() or 4))
        except Exception:
            pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

    bag_path = os.path.join(base_dir, "models", "train", "weights", "best.pt")
    if not os.path.isfile(bag_path):
        bag_path = os.path.join(base_dir, "models", "best.pt")

    person_path = os.path.join(base_dir, "yolov9s.pt")

    errors = []
    bag_model = None
    person_model = None

    if os.path.isfile(bag_path):
        try:
            bag_model = YOLO(bag_path)
        except Exception as e:
            errors.append(f"Bag model error: {e}")
    else:
        errors.append(f"Bag model not found at {bag_path}")

    if os.path.isfile(person_path):
        try:
            person_model = YOLO(person_path)
        except Exception as e:
            errors.append(f"Person model error: {e}")
    else:
        errors.append(f"Person model not found at {person_path}")

    status = f"Both models loaded ({device.upper()}: {dev_name})" if not errors else " | ".join(errors)
    return bag_model, person_model, status


# ════════════════════════════════════════════════════════════════════════════════
# OBJECT NORMALIZATION UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

# Semantic object groups for cross-model consistency
# Model class names -> internal semantic group
OBJECT_GROUP_MAP = {
    "luggage": "BAG_OBJECT",
    "backpack": "BAG_OBJECT",
    "handbag": "BAG_OBJECT",
    "suitcase": "BAG_OBJECT",
    "person": "PERSON",
}

def get_object_group(class_name):
    """Map a model class name to its semantic object group."""
    return OBJECT_GROUP_MAP.get(class_name, "OTHER")

def is_bag_object(class_name):
    """Check if a class name belongs to the BAG_OBJECT group."""
    return get_object_group(class_name) == "BAG_OBJECT"

def is_person(class_name):
    """Check if a class name belongs to the PERSON group."""
    return get_object_group(class_name) == "PERSON"


# ════════════════════════════════════════════════════════════════════════════════
# SHARED UTILITY & DETECTION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════


def detect_bags(frame, model, conf, device=None):
    """Run bag model inference. Returns list of (x1, y1, x2, y2, confidence, class_name)."""
    if model is None or frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return []
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model(frame, conf=conf, device=device, verbose=False)
    detections = []
    for r in results:
        names = r.names
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, "luggage")
            detections.append((int(x1), int(y1), int(x2), int(y2), float(box.conf[0]), cls_name))
    return detections


def detect_persons(frame, model, conf=0.4, device=None):
    """Run person model inference. Returns list of (x1, y1, x2, y2, confidence, class_name).
    Detects class 0 (person), 24 (backpack), 26 (handbag), 28 (suitcase) from COCO model."""
    if model is None or frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return []
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We load standard COCO categories: person (0), backpack (24), handbag (26), suitcase (28)
    results = model(frame, conf=conf, classes=[0, 24, 26, 28], device=device, verbose=False)
    detections = []
    for r in results:
        names = r.names
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, "person")
            detections.append((int(x1), int(y1), int(x2), int(y2), float(box.conf[0]), cls_name))
    return detections


def calculate_iou(box1, box2):
    """Calculate Intersection-over-Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def iou_deduplicate(detections, iou_thresh=0.5, cross_class_iou_thresh=0.3):
    """
    Remove duplicate boxes belonging to similar semantic classes.
    Uses object_group normalization for cross-model consistency.
    Also suppresses bag-class detections that heavily overlap a person detection
    (cross-model false positive suppression).
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
    kept = []
    
    for det in sorted_dets:
        discard = False
        det_group = get_object_group(det[5])
        for k in kept:
            k_group = get_object_group(k[5])
            
            # 1. Same semantic group deduplication (e.g., BAG_OBJECT vs BAG_OBJECT)
            # This handles cross-model duplicates like luggage (best.pt) vs suitcase (yolov9s)
            if det_group == "BAG_OBJECT" and k_group == "BAG_OBJECT":
                if calculate_iou(det, k) > iou_thresh:
                    discard = True
                    break
            # 2. Cross-class suppression: BAG_OBJECT detection overlapping a PERSON
            # If a bag detection overlaps a person, it's likely a false positive
            # (person body part misclassified as bag). Use stricter criteria:
            # - High IoU (> cross_class_iou_thresh)
            # - Low confidence bag detection (< 0.5)
            elif (det_group == "BAG_OBJECT" and k_group == "PERSON") or \
                 (det_group == "PERSON" and k_group == "BAG_OBJECT"):
                if calculate_iou(det, k) > cross_class_iou_thresh:
                    bag_det = det if det_group == "BAG_OBJECT" else k
                    bag_conf = bag_det[4]
                    # Only suppress low-confidence bag detections overlapping person
                    if bag_conf < 0.5:
                        if det_group == "BAG_OBJECT":
                            discard = True
                            break
                        else:
                            # det is person, k is bag - remove the bag from kept
                            kept.remove(k)
        if not discard:
            kept.append(det)
            
    return kept


class TrackStatus:
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    TEMPORARILY_MISSED = "TEMPORARILY_MISSED"
    REMOVED = "REMOVED"


class TrackState(dict):
    """
    Structured track state representation for multi-object tracking.
    Maintains persistent track_id, temporal age, detection counts,
    bounded trajectory history, and smoothed image-plane velocity.
    Supports both dictionary indexing (backward compatible) and attribute access.
    """
    def __init__(self, track_id, class_name, bbox, confidence, centroid=None, **kwargs):
        super().__init__()
        self["track_id"] = int(track_id)
        self["class_name"] = str(class_name)
        self["bbox"] = tuple(int(v) for v in bbox)
        self["raw_bbox"] = tuple(int(v) for v in kwargs.get("raw_bbox", bbox))
        self["smoothed_bbox"] = tuple(int(v) for v in kwargs.get("smoothed_bbox", bbox))

        if centroid is None:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
        else:
            cx, cy = centroid
        self["centroid"] = (float(cx), float(cy))
        self["smoothed_centroid"] = (float(cx), float(cy))
        self["confidence"] = float(confidence)

        now = kwargs.get("current_time", time.time())
        self["first_seen"] = float(kwargs.get("first_seen", now))
        self["last_seen"] = float(kwargs.get("last_seen", now))
        self["age"] = int(kwargs.get("age", 1))
        self["missed_frames"] = int(kwargs.get("missed_frames", 0))
        self["detection_count"] = int(kwargs.get("detection_count", 1))
        self["active"] = bool(kwargs.get("active", True))
        self["state"] = str(kwargs.get("state", TrackStatus.NEW))

        # Trajectory history bounded to MAX_TRACK_HISTORY
        init_history = kwargs.get("history", [(float(cx), float(cy), now)])
        max_hist = getattr(SentinelConfig, "MAX_TRACK_HISTORY", 60)
        self["history"] = list(init_history)[-max_hist:]

        # Image-plane movement velocity (pixels per second)
        self["velocity"] = float(kwargs.get("velocity", 0.0))
        self["smoothed_velocity"] = float(kwargs.get("smoothed_velocity", 0.0))
        self["speed"] = self["smoothed_velocity"]  # alias for behavioral engine

        # Behavioral and threat properties
        self["active_threats"] = set(kwargs.get("active_threats", set()))
        self["alert_dismissed"] = bool(kwargs.get("alert_dismissed", False))
        self["last_alert_time"] = float(kwargs.get("last_alert_time", 0.0))
        self["risk_score"] = int(kwargs.get("risk_score", 0))
        self["severity"] = str(kwargs.get("severity", "LOW"))
        self["stationary_start"] = kwargs.get("stationary_start", None)
        self["last_move_time"] = float(kwargs.get("last_move_time", now))
        self["nearest_person_id"] = kwargs.get("nearest_person_id", None)
        self["last_proximity_time"] = float(kwargs.get("last_proximity_time", now))
        self["direction_reversals"] = list(kwargs.get("direction_reversals", []))
        self["consecutive_running_frames"] = int(kwargs.get("consecutive_running_frames", 0))
        self["consecutive_restricted_frames"] = int(kwargs.get("consecutive_restricted_frames", 0))

        for k, v in kwargs.items():
            if k not in self:
                self[k] = v

    @property
    def track_id(self): return self["track_id"]
    @property
    def class_name(self): return self["class_name"]
    @property
    def bbox(self): return self["bbox"]
    @property
    def raw_bbox(self): return self["raw_bbox"]
    @property
    def smoothed_bbox(self): return self["smoothed_bbox"]
    @property
    def centroid(self): return self["centroid"]
    @property
    def smoothed_centroid(self): return self["smoothed_centroid"]
    @property
    def confidence(self): return self["confidence"]
    @property
    def age(self): return self["age"]
    @property
    def missed_frames(self): return self["missed_frames"]
    @property
    def detection_count(self): return self["detection_count"]
    @property
    def active(self): return self["active"]
    @property
    def state(self): return self["state"]
    @property
    def history(self): return self["history"]
    @property
    def velocity(self): return self["velocity"]
    @property
    def smoothed_velocity(self): return self["smoothed_velocity"]


def match_entities(detections, tracked_entities, threshold=None, max_missing_sec=None):
    """
    Class-aware multi-object centroid tracker with competitive distance assignment,
    EMA bounding box smoothing, temporal age tracking, and image-plane velocity.

    detections: list of (x1, y1, x2, y2, confidence, class_name)
    tracked_entities: dict mapping semantic key (e.g. 'BAG_OBJECT_1') to TrackState or dict
    threshold: maximum pixel distance for spatial association (defaults to SentinelConfig.TRACK_MATCH_THRESHOLD)
    max_missing_sec: max seconds to keep temporarily missing tracks (defaults to SentinelConfig.TRACK_TIMEOUT)
    """
    if threshold is None:
        threshold = getattr(SentinelConfig, "TRACK_MATCH_THRESHOLD", 100.0)
    if max_missing_sec is None:
        max_missing_sec = getattr(SentinelConfig, "TRACK_TIMEOUT", 3.0)
    max_missed_frames = getattr(SentinelConfig, "MAX_MISSED_FRAMES", 30)
    alpha = getattr(SentinelConfig, "BBOX_SMOOTHING_ALPHA", 0.75)
    beta = getattr(SentinelConfig, "VELOCITY_SMOOTHING_ALPHA", 0.30)
    max_history = getattr(SentinelConfig, "MAX_TRACK_HISTORY", 60)

    current_time = time.time()
    new_tracked = {}
    matched_keys = set()
    matched_det_indices = set()

    # Organize detections by semantic group to enforce strict class-aware tracking
    dets_by_group = {}
    for idx, det in enumerate(detections):
        cls_name = det[5] if len(det) > 5 else "luggage"
        group = get_object_group(cls_name)
        dets_by_group.setdefault(group, []).append((idx, det))

    # Organize existing tracks by semantic group
    tracks_by_group = {}
    for key, info in tracked_entities.items():
        cls_name = info.get("class_name", "luggage")
        group = get_object_group(cls_name)
        tracks_by_group.setdefault(group, []).append((key, info))

    all_groups = set(dets_by_group.keys()) | set(tracks_by_group.keys())

    for group in all_groups:
        group_dets = dets_by_group.get(group, [])
        group_tracks = tracks_by_group.get(group, [])

        # Build candidate pairs with Euclidean distance
        candidate_pairs = []
        for det_idx, det in group_dets:
            x1, y1, x2, y2, conf, cls_name = det[:6]
            dcx = (x1 + x2) / 2.0
            dcy = (y1 + y2) / 2.0
            for t_key, t_info in group_tracks:
                tbx = t_info["bbox"]
                tcx = (tbx[0] + tbx[2]) / 2.0
                tcy = (tbx[1] + tbx[3]) / 2.0
                dist = math.hypot(dcx - tcx, dcy - tcy)
                if dist <= threshold:
                    candidate_pairs.append((dist, det_idx, det, t_key, t_info))

        # Sort candidate pairs by distance (competitive closest-first assignment)
        candidate_pairs.sort(key=lambda p: p[0])

        group_matched_dets = set()
        group_matched_tracks = set()

        for dist, det_idx, det, t_key, t_info in candidate_pairs:
            if det_idx in group_matched_dets or t_key in group_matched_tracks:
                continue

            group_matched_dets.add(det_idx)
            group_matched_tracks.add(t_key)
            matched_det_indices.add(det_idx)
            matched_keys.add(t_key)

            x1, y1, x2, y2, conf, cls_name = det[:6]
            dcx = (x1 + x2) / 2.0
            dcy = (y1 + y2) / 2.0

            # Bounding box EMA smoothing
            old_bbox = t_info["bbox"]
            smooth_x1 = int(alpha * x1 + (1.0 - alpha) * old_bbox[0])
            smooth_y1 = int(alpha * y1 + (1.0 - alpha) * old_bbox[1])
            smooth_x2 = int(alpha * x2 + (1.0 - alpha) * old_bbox[2])
            smooth_y2 = int(alpha * y2 + (1.0 - alpha) * old_bbox[3])
            smooth_bbox = (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
            smooth_cx = (smooth_x1 + smooth_x2) / 2.0
            smooth_cy = (smooth_y1 + smooth_y2) / 2.0

            # Trajectory history
            history = list(t_info.get("history", []))
            history.append((dcx, dcy, current_time))
            if len(history) > max_history:
                history.pop(0)

            # Image-plane velocity (pixels / second)
            prev_smoothed_vel = t_info.get("smoothed_velocity", t_info.get("speed", 0.0))
            inst_vel = 0.0
            if len(history) >= 2:
                prev_x, prev_y, prev_t = history[-2]
                dt = current_time - prev_t
                if dt > 0.001:
                    inst_vel = math.hypot(dcx - prev_x, dcy - prev_y) / dt
            smoothed_vel = beta * inst_vel + (1.0 - beta) * prev_smoothed_vel

            updated_track = TrackState(
                track_id=t_info["track_id"],
                class_name=cls_name,
                bbox=smooth_bbox,
                confidence=conf,
                centroid=(dcx, dcy),
                raw_bbox=(x1, y1, x2, y2),
                smoothed_bbox=smooth_bbox,
                smoothed_centroid=(smooth_cx, smooth_cy),
                first_seen=t_info["first_seen"],
                last_seen=current_time,
                age=t_info.get("age", 1) + 1,
                missed_frames=0,
                detection_count=t_info.get("detection_count", 1) + 1,
                active=True,
                state=TrackStatus.ACTIVE,
                history=history,
                velocity=inst_vel,
                smoothed_velocity=smoothed_vel,
                # Preserve behavior fields
                active_threats=t_info.get("active_threats", set()),
                alert_dismissed=t_info.get("alert_dismissed", False),
                last_alert_time=t_info.get("last_alert_time", 0.0),
                risk_score=t_info.get("risk_score", 0),
                severity=t_info.get("severity", "LOW"),
                stationary_start=t_info.get("stationary_start", None),
                last_move_time=t_info.get("last_move_time", current_time),
                nearest_person_id=t_info.get("nearest_person_id", None),
                last_proximity_time=t_info.get("last_proximity_time", current_time),
                direction_reversals=t_info.get("direction_reversals", []),
                consecutive_running_frames=t_info.get("consecutive_running_frames", 0),
                consecutive_restricted_frames=t_info.get("consecutive_restricted_frames", 0),
            )
            new_tracked[t_key] = updated_track

        # Spawn new tracks for unmatched detections in this group
        for det_idx, det in group_dets:
            if det_idx in matched_det_indices:
                continue

            x1, y1, x2, y2, conf, cls_name = det[:6]
            dcx = (x1 + x2) / 2.0
            dcy = (y1 + y2) / 2.0

            existing_ids = [
                info["track_id"]
                for info in tracked_entities.values()
                if get_object_group(info["class_name"]) == group
            ]
            new_ids = [
                info["track_id"]
                for info in new_tracked.values()
                if get_object_group(info["class_name"]) == group
            ]
            all_ids = existing_ids + new_ids
            new_id = (max(all_ids) + 1) if all_ids else 1
            key = f"{group}_{new_id}"

            new_track = TrackState(
                track_id=new_id,
                class_name=cls_name,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                centroid=(dcx, dcy),
                raw_bbox=(x1, y1, x2, y2),
                smoothed_bbox=(x1, y1, x2, y2),
                smoothed_centroid=(dcx, dcy),
                first_seen=current_time,
                last_seen=current_time,
                age=1,
                missed_frames=0,
                detection_count=1,
                active=True,
                state=TrackStatus.NEW,
                history=[(dcx, dcy, current_time)],
                velocity=0.0,
                smoothed_velocity=0.0,
            )
            new_tracked[key] = new_track
            matched_det_indices.add(det_idx)

    # Handle unmatched tracks (temporarily missing detections)
    for key, info in tracked_entities.items():
        if key in matched_keys:
            continue

        new_missed = info.get("missed_frames", 0) + 1
        elapsed_missing = current_time - info["last_seen"]

        # Keep track if within both frame count and temporal limits
        if new_missed <= max_missed_frames and elapsed_missing < max_missing_sec:
            decayed_vel = info.get("smoothed_velocity", 0.0) * 0.8
            kept_track = TrackState(
                track_id=info["track_id"],
                class_name=info["class_name"],
                bbox=info["bbox"],
                confidence=info["confidence"],
                centroid=info["centroid"],
                raw_bbox=info.get("raw_bbox", info["bbox"]),
                smoothed_bbox=info.get("smoothed_bbox", info["bbox"]),
                smoothed_centroid=info.get("smoothed_centroid", info["centroid"]),
                first_seen=info["first_seen"],
                last_seen=info["last_seen"],
                age=info.get("age", 1) + 1,
                missed_frames=new_missed,
                detection_count=info.get("detection_count", 1),
                active=False,
                state=TrackStatus.TEMPORARILY_MISSED,
                history=info.get("history", []),
                velocity=0.0,
                smoothed_velocity=decayed_vel,
                active_threats=info.get("active_threats", set()),
                alert_dismissed=info.get("alert_dismissed", False),
                last_alert_time=info.get("last_alert_time", 0.0),
                risk_score=info.get("risk_score", 0),
                severity=info.get("severity", "LOW"),
                stationary_start=info.get("stationary_start", None),
                last_move_time=info.get("last_move_time", current_time),
                nearest_person_id=info.get("nearest_person_id", None),
                last_proximity_time=info.get("last_proximity_time", current_time),
                direction_reversals=info.get("direction_reversals", []),
                consecutive_running_frames=info.get("consecutive_running_frames", 0),
                consecutive_restricted_frames=info.get("consecutive_restricted_frames", 0),
            )
            new_tracked[key] = kept_track

    return new_tracked


# Backward-compatibility wrappers for tests and app logic
def match_bag_centroids(detections, tracked_bags, threshold=80, max_missing_sec=3.0):
    """Adapt bag detections list and match using match_entities tracker."""
    formatted_dets = []
    for d in detections:
        if len(d) == 5:
            formatted_dets.append((d[0], d[1], d[2], d[3], d[4], "luggage"))
        else:
            formatted_dets.append(d)
    
    # Map tracked_bags to unified format
    unified = {}
    for bid, info in tracked_bags.items():
        key = f"luggage_{bid}"
        unified[key] = {
            "track_id": bid,
            "class_name": "luggage",
            "confidence": info.get("conf", 0.8),
            "bbox": info["bbox"],
            "centroid": ((info["bbox"][0]+info["bbox"][2])/2.0, (info["bbox"][1]+info["bbox"][3])/2.0),
            "first_seen": info["first_seen"],
            "last_seen": info["last_seen"],
            "active_threats": {"UNATTENDED_OBJECT"} if info.get("is_alerted") else set(),
            "alert_dismissed": info.get("alert_dismissed", False),
            "risk_score": info.get("risk_score", 0),
            "stationary_start": info["first_seen"] if info.get("is_alerted") else None,
            "last_proximity_time": info.get("owner_returned_at") or info["first_seen"],
            "return_msg_frames": info.get("return_msg_frames", 0)
        }
        
    res = match_entities(formatted_dets, unified, threshold, max_missing_sec)
    
    # Map back to tracked_bags format
    output = {}
    for key, val in res.items():
        bid = val["track_id"]
        output[bid] = {
            "bbox": val["bbox"],
            "conf": val["confidence"],
            "first_seen": val["first_seen"],
            "last_seen": val["last_seen"],
            "is_alerted": "UNATTENDED_OBJECT" in val["active_threats"],
            "alert_dismissed": val["alert_dismissed"],
            "risk_score": val["risk_score"],
            "owner_returned_at": val.get("last_proximity_time"),
            "return_msg_frames": val.get("return_msg_frames", 0)
        }
    return output


def match_person_centroids(detections, tracked_persons, threshold=100, max_missing_sec=3.0):
    """Adapt person detections list and match using match_entities tracker."""
    formatted_dets = []
    for d in detections:
        if len(d) == 5:
            formatted_dets.append((d[0], d[1], d[2], d[3], d[4], "person"))
        else:
            formatted_dets.append(d)
            
    unified = {}
    for pid, info in tracked_persons.items():
        key = f"person_{pid}"
        unified[key] = {
            "track_id": pid,
            "class_name": "person",
            "confidence": info.get("conf", 0.8),
            "bbox": info["bbox"],
            "centroid": ((info["bbox"][0]+info["bbox"][2])/2.0, (info["bbox"][1]+info["bbox"][3])/2.0),
            "first_seen": info["first_seen"],
            "last_seen": info["last_seen"],
            "active_threats": {"LOITERING"} if info.get("is_loitering") else set(),
            "alert_dismissed": info.get("loiter_dismissed", False),
            "dwell_seconds": info.get("dwell_seconds", 0)
        }
        
    res = match_entities(formatted_dets, unified, threshold, max_missing_sec)
    
    output = {}
    for key, val in res.items():
        pid = val["track_id"]
        output[pid] = {
            "bbox": val["bbox"],
            "conf": val["confidence"],
            "first_seen": val["first_seen"],
            "last_seen": val["last_seen"],
            "dwell_seconds": val["last_seen"] - val["first_seen"],
            "is_loitering": "LOITERING" in val["active_threats"],
            "loiter_dismissed": val["alert_dismissed"]
        }
    return output


def check_owner_return(tracked_bags, tracked_persons, return_radius=150):
    """Check proximity owner return. Thin wrapper returning reset bag IDs."""
    # This is handled directly in our behavior analyzer engine now.
    return set()


def calculate_entity_risk(track):
    """Calculate overall risk score (0-100), severity label and BGR color."""
    threats = track.get("active_threats", set())
    if not threats or track.get("alert_dismissed", False):
        return 0, "NORMAL", (180, 180, 0)
        
    max_score = 0
    if "LOITERING" in threats:
        max_score = max(max_score, 40)
    if "REPEATED_MOVEMENT" in threats:
        max_score = max(max_score, 50)
    if "SUDDEN_RUNNING" in threats:
        max_score = max(max_score, 70)
    if "RESTRICTED_ZONE" in threats:
        max_score = max(max_score, 85)
    if "UNATTENDED_OBJECT" in threats:
        elapsed = time.time() - (track.get("stationary_start") or time.time())
        ratio = elapsed / SentinelConfig.STATIONARY_THRESHOLD
        obj_score = min(int(ratio * 100), 100)
        max_score = max(max_score, obj_score)

    if max_score < 34:
        return max_score, "LOW", (0, 200, 100)       # green BGR
    elif max_score < 67:
        return max_score, "MEDIUM", (0, 165, 255)     # orange BGR
    elif max_score < 100:
        return max_score, "HIGH", (0, 60, 255)        # red-orange BGR
    else:
        return max_score, "CRITICAL", (0, 0, 255)     # pure red BGR


def analyze_behaviors(tracked_entities, w, h):
    """
    Central suspicious behavior rule evaluation engine.
    Modifies tracked_entities in-place.
    """
    current_time = time.time()
    
    # Pre-compute frame diagonal for normalized thresholds
    frame_diag = math.sqrt(w * w + h * h)
    
    # Normalized thresholds (pixels)
    loiter_spatial_thresh = SentinelConfig.LOITER_SPATIAL_RATIO * frame_diag
    stationary_disp_thresh = SentinelConfig.STATIONARY_DISPLACEMENT_RATIO * frame_diag
    proximity_thresh = SentinelConfig.PROXIMITY_RATIO * frame_diag
    speed_abs_min = SentinelConfig.SPEED_ABSOLUTE_MIN_RATIO * frame_diag
    pacing_min_mov = SentinelConfig.PACING_MIN_MOVEMENT_RATIO * frame_diag
    pacing_max_net_disp = SentinelConfig.PACING_MAX_NET_DISP_RATIO * frame_diag
    pacing_min_path = SentinelConfig.PACING_MIN_PATH_RATIO * frame_diag
    pacing_compare_window = SentinelConfig.PACING_COMPARISON_WINDOW
    
    persons = {key: info for key, info in tracked_entities.items() if is_person(info["class_name"])}
    objects = {key: info for key, info in tracked_entities.items() if is_bag_object(info["class_name"])}
    
    # 1. Proximity Association and Unattended Luggage Engine
    for obj_key, obj_info in objects.items():
        ocx, ocy = obj_info["centroid"]
        
        # Find distance to closest person
        min_dist = float('inf')
        nearest_person_key = None
        for p_key, p_info in persons.items():
            pcx, pcy = p_info["centroid"]
            dist = math.sqrt((ocx - pcx)**2 + (ocy - pcy)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_person_key = p_key

        if min_dist <= proximity_thresh:
            # Nearest person is nearby — update proximity association
            obj_info["nearest_person_id"] = nearest_person_key
            obj_info["last_proximity_time"] = current_time
            if "UNATTENDED_OBJECT" in obj_info["active_threats"]:
                obj_info["active_threats"].discard("UNATTENDED_OBJECT")
                obj_info["stationary_start"] = None
        else:
            # Stationary Check: verify if the object hasn't moved significantly
            history = obj_info["history"]
            if len(history) >= 10:
                recent_pts = [pt[:2] for pt in history[-10:]]
                xs = [c[0] for c in recent_pts]
                ys = [c[1] for c in recent_pts]
                displacement = math.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)
                if displacement < stationary_disp_thresh:
                    if obj_info["stationary_start"] is None:
                        obj_info["stationary_start"] = current_time
                else:
                    obj_info["stationary_start"] = None
                    obj_info["active_threats"].discard("UNATTENDED_OBJECT")
            else:
                if obj_info["stationary_start"] is None:
                    obj_info["stationary_start"] = current_time

            # Flag unattended threat if stationary duration exceeds threshold
            if obj_info["stationary_start"] is not None:
                elapsed_unattended = current_time - obj_info["last_proximity_time"]
                if elapsed_unattended >= SentinelConfig.STATIONARY_THRESHOLD:
                    obj_info["active_threats"].add("UNATTENDED_OBJECT")

    # 2. Person Movement Behavior Rules
    for p_key, p_info in persons.items():
        pcx, pcy = p_info["centroid"]
        history = p_info["history"]

        # A. Restricted Zone Entry Check
        inside_zone = False
        if SentinelConfig.RESTRICTED_ZONE_ENABLED:
            poly_pts = np.array([
                [int(pt[0] * w), int(pt[1] * h)] for pt in SentinelConfig.RESTRICTED_ZONE_COORDS
            ], dtype=np.int32)
            dist_to_poly = cv2.pointPolygonTest(poly_pts, (pcx, pcy), False)
            if dist_to_poly >= 0:
                inside_zone = True

        if inside_zone:
            p_info["consecutive_restricted_frames"] += 1
            if p_info["consecutive_restricted_frames"] >= 8:  # Temporal persistence (8 frames)
                p_info["active_threats"].add("RESTRICTED_ZONE")
        else:
            p_info["consecutive_restricted_frames"] = 0
            p_info["active_threats"].discard("RESTRICTED_ZONE")

        # B. Prolonged Loitering Check
        recent_pts = [pt for pt in history if current_time - pt[2] <= SentinelConfig.LOITER_THRESHOLD]
        total_time = current_time - p_info["first_seen"]
        
        if total_time >= SentinelConfig.LOITER_THRESHOLD and len(recent_pts) >= 5:
            xs = [pt[0] for pt in recent_pts]
            ys = [pt[1] for pt in recent_pts]
            max_spread = math.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)
            if max_spread < loiter_spatial_thresh:  # Remained within spatial threshold
                p_info["active_threats"].add("LOITERING")
            else:
                p_info["active_threats"].discard("LOITERING")

        # C. Sudden Running Check
        speed = p_info["speed"]
        past_pts = [pt for pt in history if 2.0 <= current_time - pt[2] <= 7.0]
        if len(past_pts) >= 10:
            past_speeds = []
            for i in range(1, len(past_pts)):
                dx = past_pts[i][0] - past_pts[i-1][0]
                dy = past_pts[i][1] - past_pts[i-1][1]
                dt = past_pts[i][2] - past_pts[i-1][2]
                if dt > 0:
                    past_speeds.append(math.sqrt(dx**2 + dy**2) / dt)
            if past_speeds:
                avg_past_speed = sum(past_speeds) / len(past_speeds)
                # Rapid if absolute speed exceeds normalized floor AND is 2.5x normal speed
                if speed > speed_abs_min and speed >= SentinelConfig.SPEED_RATIO_THRESHOLD * max(avg_past_speed, speed_abs_min * 0.125):
                    p_info["consecutive_running_frames"] += 1
                    if p_info["consecutive_running_frames"] >= 8:
                        p_info["active_threats"].add("SUDDEN_RUNNING")
                else:
                    p_info["consecutive_running_frames"] = 0
                    p_info["active_threats"].discard("SUDDEN_RUNNING")
        else:
            # Fallback to absolute speed if history is short
            if speed > speed_abs_min * 1.5:
                p_info["consecutive_running_frames"] += 1
                if p_info["consecutive_running_frames"] >= 15:
                    p_info["active_threats"].add("SUDDEN_RUNNING")
            else:
                p_info["consecutive_running_frames"] = 0

        # D. Repeated Back-and-Forth Movement (Pacing) — 2D direction analysis
        if len(history) >= 20:
            reversals = p_info.get("direction_reversals", [])
            reversals = [r for r in reversals if current_time - r <= 15.0]
            
            # Use time-based comparison window instead of fixed frame indices
            # Find points approximately pacing_compare_window seconds apart
            if len(history) >= 5:
                # Look for a point approximately pacing_compare_window seconds ago
                target_time_current = current_time
                target_time_prev = current_time - pacing_compare_window
                target_time_earlier = current_time - 2 * pacing_compare_window
                
                # Find closest history points to these target times
                idx_current = len(history) - 1
                idx_prev = None
                idx_earlier = None
                
                for i in range(len(history) - 1, -1, -1):
                    if idx_prev is None and history[i][2] <= target_time_prev:
                        idx_prev = i
                    if idx_earlier is None and history[i][2] <= target_time_earlier:
                        idx_earlier = i
                        break
                
                # Fallback to fixed indices if time-based lookup fails
                if idx_prev is None:
                    idx_prev = max(0, len(history) - 3)
                if idx_earlier is None:
                    idx_earlier = max(0, len(history) - 5)
                
                dx_current = history[idx_current][0] - history[idx_prev][0]
                dy_current = history[idx_current][1] - history[idx_prev][1]
                dx_prev = history[idx_prev][0] - history[idx_earlier][0]
                dy_prev = history[idx_prev][1] - history[idx_earlier][1]
                
                mag_current = math.hypot(dx_current, dy_current)
                mag_prev = math.hypot(dx_prev, dy_prev)
                
                # Check 2D directional reversal (dot product negative = >90 degree turn)
                if mag_current > pacing_min_mov and mag_prev > pacing_min_mov:
                    dot_product = dx_current * dx_prev + dy_current * dy_prev
                    if dot_product < 0:  # Direction reversed
                        if not reversals or (current_time - reversals[-1] > 1.5):
                            reversals.append(current_time)
            p_info["direction_reversals"] = reversals

            if len(reversals) >= 3:
                pts_15s = [pt for pt in history if current_time - pt[2] <= 15.0]
                if len(pts_15s) >= 10:
                    total_dist = 0.0
                    for i in range(1, len(pts_15s)):
                        total_dist += math.sqrt((pts_15s[i][0] - pts_15s[i-1][0])**2 + (pts_15s[i][1] - pts_15s[i-1][1])**2)
                    net_disp = math.sqrt((pts_15s[-1][0] - pts_15s[0][0])**2 + (pts_15s[-1][1] - pts_15s[0][1])**2)
                    
                    if net_disp < pacing_max_net_disp and total_dist > pacing_min_path:
                        p_info["active_threats"].add("REPEATED_MOVEMENT")
                    else:
                        p_info["active_threats"].discard("REPEATED_MOVEMENT")


def draw_overlays(frame, tracked_entities, frame_count=0):
    """
    Draw bounding boxes, threat overlays, and Restricted Zone.
    Returns (annotated_frame, list_of_new_alerts).
    """
    h, w = frame.shape[:2]
    current_time = time.time()
    new_alerts = []

    # Run temporal behavior rules
    analyze_behaviors(tracked_entities, w, h)

    # 1. Draw Restricted Zone Polygon
    if SentinelConfig.RESTRICTED_ZONE_ENABLED:
        poly_pts = np.array([
            [int(pt[0] * w), int(pt[1] * h)] for pt in SentinelConfig.RESTRICTED_ZONE_COORDS
        ], dtype=np.int32)
        
        # Check if zone is currently violated by any active restricted zone threat
        zone_violated = any("RESTRICTED_ZONE" in info["active_threats"] for info in tracked_entities.values())
        zone_color = (0, 0, 255) if zone_violated else (255, 255, 0) # Red if violated, Cyan/Yellow if secure
        
        cv2.polylines(frame, [poly_pts], True, zone_color, 2)
        cv2.putText(frame, "RESTRICTED SURVEILLANCE ZONE", (poly_pts[0][0] + 5, poly_pts[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone_color, 1)

    # 2. Draw Bounding Boxes and Status Badges
    for key, info in tracked_entities.items():
        if info.get("alert_dismissed", False):
            continue
        # Avoid rendering ghost boxes if target has been missed for several frames
        if info.get("missed_frames", 0) > 5:
            continue

        x1, y1, x2, y2 = info["bbox"]
        class_name = info["class_name"]
        track_id = info["track_id"]
        conf = int(info["confidence"] * 100)
        
        risk_score, severity, color = calculate_entity_risk(info)
        info["risk_score"] = risk_score
        info["severity"] = severity

        # Change color for flashing alerts
        if risk_score >= 85: # HIGH or CRITICAL
            if (frame_count // 8) % 2 == 0:
                color = (0, 0, 255) # flash red
            else:
                color = (255, 255, 255) # flash white

        # Main Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if risk_score < 70 else 3)

        # Draw HUD label
        label = f"{class_name.upper()} #{track_id} | {conf}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Draw suspicious behaviors badge if active
        threats = info.get("active_threats", set())
        if threats:
            threat_str = " | ".join(list(threats))
            (ttw, tth), _ = cv2.getTextSize(threat_str, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(frame, (x1, y2), (x1 + ttw + 4, y2 + tth + 6), (0, 0, 0), -1)
            cv2.putText(frame, threat_str, (x1 + 2, y2 + tth + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Fire alerts with cooldown control
            for t in threats:
                last_alert = info.get("last_alert_time", 0.0)
                if current_time - last_alert >= SentinelConfig.ALERT_COOLDOWN:
                    info["last_alert_time"] = current_time
                    new_alerts.append({
                        "key": key,
                        "track_id": track_id,
                        "class_name": class_name,
                        "behavior": t,
                        "severity": severity,
                        "score": risk_score,
                        "description": f"{class_name.upper()} #{track_id} - {t} alert triggered"
                    })

    # 3. Draw Flashing Alarm Banner
    active_threat_count = sum(len(info["active_threats"]) for info in tracked_entities.values() if not info.get("alert_dismissed"))
    if active_threat_count > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (10, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        
        threat_labels = []
        for key, info in tracked_entities.items():
            if info["active_threats"] and not info.get("alert_dismissed"):
                threat_labels.append(f"{info['class_name'].upper()}#{info['track_id']} ({','.join(info['active_threats'])})")
        
        banner = "⚠ ALERT -- " + " | ".join(threat_labels)[:80] + " -- SECURITY ACTION REQUIRED"
        cv2.putText(frame, banner, (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 255), 2)

    return frame, new_alerts


def frame_to_tk(frame, canvas_w, canvas_h):
    """Convert cv2 BGR frame to ImageTk.PhotoImage sized to fit canvas."""
    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return None
    scale = min(canvas_w / w, canvas_h / h)
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class SentinelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SENTINEL AI \u2014 Public Safety Surveillance")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # State
        self.status_var = tk.StringVar(value="IDLE")
        self.training_process = None
        self.video_running = False
        self.camera_running = False
        self.current_training_epoch = 0
        self.train_losses = []
        self.map_values = []
        self.loss_epochs = []
        self.map_epochs = []
        self._last_frame = None
        self._alert_sound_active = False
        self._alert_sound_thread = None

        self._setup_styles()
        self._build_header()
        self._build_notebook()
        self._build_status_bar()
        self._build_dataset_tab()
        self._build_train_tab()
        self._build_video_tab()
        self._build_camera_tab()
        self._update_clock()

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def _on_close(self):
        self.video_running = False
        self.camera_running = False
        if self.training_process:
            try:
                self.training_process.terminate()
            except Exception:
                pass
        self.root.destroy()

    # ── Styles ───────────────────────────────────────────────────────────────

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("default")

        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=0)
        style.configure("TNotebook.Tab", background=CARD, foreground=MUTED,
                         font=FONT_BODY, padding=[22, 10])
        style.map("TNotebook.Tab",
                  background=[("selected", PANEL)],
                  foreground=[("selected", ACCENT)])

        style.configure("Horizontal.TProgressbar",
                         background=ACCENT, troughcolor=CARD, borderwidth=0, thickness=6)

    # ── Header ───────────────────────────────────────────────────────────────

    def _build_header(self):
        header = tk.Frame(self.root, bg=PANEL, height=60)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="\u2B21 SENTINEL AI", font=FONT_HEADING,
                 fg=ACCENT, bg=PANEL).pack(side="left", padx=20)

        tk.Label(header, text="PUBLIC SAFETY SURVEILLANCE", font=FONT_SMALL,
                 fg=MUTED, bg=PANEL).pack(side="left", padx=(0, 20))

        self.clock_label = tk.Label(header, text="", font=FONT_BODY, fg=MUTED, bg=PANEL)
        self.clock_label.pack(side="right", padx=20)

    def _update_clock(self):
        self.clock_label.config(text=time.strftime("%Y-%m-%d  %H:%M:%S"))
        self.root.after(1000, self._update_clock)

    # ── Notebook ─────────────────────────────────────────────────────────────

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.tab_dataset = tk.Frame(self.notebook, bg=BG)
        self.tab_train   = tk.Frame(self.notebook, bg=BG)
        self.tab_video   = tk.Frame(self.notebook, bg=BG)
        self.tab_camera  = tk.Frame(self.notebook, bg=BG)
        self.tab_stats   = tk.Frame(self.notebook, bg=BG)

        self.notebook.add(self.tab_dataset, text="  [ DATASET ]  ")
        self.notebook.add(self.tab_train,   text="  [ TRAIN ]  ")
        self.notebook.add(self.tab_video,   text="  [ VIDEO DEMO ]  ")
        self.notebook.add(self.tab_camera,  text="  [ LIVE CAMERA ]  ")
        self.notebook.add(self.tab_stats,   text="  [ STATS ]  ")
        self._build_stats_tab()

    # ── Status Bar ───────────────────────────────────────────────────────────

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=PANEL, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        tk.Label(bar, text="\u25CF", font=("Consolas", 8), fg=ACCENT,
                 bg=PANEL).pack(side="left", padx=(15, 4))

        self.status_label = tk.Label(bar, textvariable=self.status_var,
                                      font=FONT_SMALL, fg=ACCENT, bg=PANEL)
        self.status_label.pack(side="left")

    def _set_status(self, text, color=ACCENT):
        self.status_var.set(text)
        self.status_label.config(fg=color)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — DATASET
    # ═════════════════════════════════════════════════════════════════════════

    def _build_dataset_tab(self):
        # ── Left panel ──
        left = tk.Frame(self.tab_dataset, bg=PANEL, width=400)
        left.pack(side="left", fill="y", padx=(10, 5), pady=10)
        left.pack_propagate(False)

        tk.Label(left, text="DATASET VERSIONS", font=FONT_HEADING,
                 fg=ACCENT, bg=PANEL).pack(padx=15, pady=(15, 10), anchor="w")

        self.dataset_list = tk.Text(left, bg=CARD, fg=TEXT, font=FONT_BODY,
                                     relief="flat", height=20, width=40,
                                     insertbackground=TEXT, padx=12, pady=10,
                                     highlightthickness=1, highlightbackground=MUTED)
        self.dataset_list.pack(padx=15, pady=5, fill="both", expand=True)
        self._populate_dataset_list()
        self.dataset_list.config(state="disabled")

        self.verify_progress = ttk.Progressbar(left, mode="indeterminate",
                                                 style="Horizontal.TProgressbar")
        self.verify_progress.pack(padx=15, pady=8, fill="x")

        tk.Button(left, text="\u25B6 RUN VERIFICATION", font=FONT_BODY,
                  bg=ACCENT, fg=BG, relief="flat", cursor="hand2",
                  activebackground="#00cc6a", activeforeground=BG,
                  command=self._run_verification).pack(padx=15, pady=(0, 15),
                                                        fill="x", ipady=8)

        # ── Right panel ──
        right = tk.Frame(self.tab_dataset, bg=PANEL)
        right.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

        tk.Label(right, text="VERIFICATION RESULTS", font=FONT_HEADING,
                 fg=ACCENT, bg=PANEL).pack(padx=15, pady=(15, 10), anchor="w")

        self.verify_output = tk.Text(right, bg=CARD, fg=TEXT, font=FONT_BODY,
                                      relief="flat", insertbackground=TEXT,
                                      padx=12, pady=10,
                                      highlightthickness=1, highlightbackground=MUTED)
        self.verify_output.pack(padx=15, pady=(5, 15), fill="both", expand=True)
        self.verify_output.tag_config("green",   foreground=ACCENT)
        self.verify_output.tag_config("red",     foreground=DANGER)
        self.verify_output.tag_config("header",  foreground=ACCENT, font=FONT_HEADING)
        self.verify_output.tag_config("warning", foreground=WARNING)
        self.verify_output.config(state="disabled")

    def _populate_dataset_list(self):
        dataset_dir = os.path.join(BASE_DIR, "dataset")
        self.dataset_list.config(state="normal")
        self.dataset_list.delete("1.0", "end")

        if os.path.isdir(dataset_dir):
            folders = sorted(d for d in os.listdir(dataset_dir)
                             if os.path.isdir(os.path.join(dataset_dir, d)))
            if folders:
                for folder in folders:
                    self.dataset_list.insert("end", f"  \U0001F4C1 {folder}\n")
            else:
                self.dataset_list.insert("end", "  No dataset folders found.\n")
        else:
            self.dataset_list.insert("end", "  dataset/ directory not found.\n")

        self.dataset_list.config(state="disabled")

    # ── Verification ─────────────────────────────────────────────────────────

    def _run_verification(self):
        self.verify_output.config(state="normal")
        self.verify_output.delete("1.0", "end")
        self.verify_output.config(state="disabled")
        self.verify_progress.start(12)
        self._set_status("VERIFYING", WARNING)
        threading.Thread(target=self._verification_worker, daemon=True).start()

    def _verification_worker(self):
        try:
            yaml_path = os.path.join(BASE_DIR, "data.yaml")
            if not os.path.isfile(yaml_path):
                self.root.after(0, self._vlog, "\u274C data.yaml not found!\n", "red")
                self.root.after(0, self._verification_done, False)
                return

            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            self.root.after(0, self._vlog,
                            "\u2501\u2501\u2501 DATA.YAML LOADED \u2501\u2501\u2501\n", "header")
            self.root.after(0, self._vlog,
                            f"  Classes: {data.get('names', 'N/A')}\n", "green")
            self.root.after(0, self._vlog,
                            f"  NC:      {data.get('nc', 'N/A')}\n\n", "green")

            issues = False

            for split in ("train", "val", "test"):
                split_path = data.get(split, "")
                if not split_path:
                    self.root.after(0, self._vlog,
                                    f"\u26A0 {split} path not defined in data.yaml\n", "red")
                    issues = True
                    continue

                if not os.path.isabs(split_path):
                    split_path = os.path.normpath(os.path.join(BASE_DIR, split_path))

                images_dir = split_path
                labels_dir = split_path.replace("images", "labels")

                self.root.after(0, self._vlog,
                                f"\u2501\u2501\u2501 {split.upper()} SPLIT \u2501\u2501\u2501\n", "header")

                if not os.path.isdir(images_dir):
                    self.root.after(0, self._vlog,
                                    f"  \u274C Images dir not found:\n  {images_dir}\n", "red")
                    issues = True
                    continue

                imgs = [f for f in os.listdir(images_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                self.root.after(0, self._vlog,
                                f"  \U0001F4F7 Images: {len(imgs)}\n", "green")

                if os.path.isdir(labels_dir):
                    lbls = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
                    self.root.after(0, self._vlog,
                                    f"  \U0001F3F7\uFE0F  Labels: {len(lbls)}\n", "green")

                    zero = [f for f in lbls
                            if os.path.getsize(os.path.join(labels_dir, f)) == 0]
                    if zero:
                        issues = True
                        self.root.after(0, self._vlog,
                                        f"  \u26A0 Zero-byte labels: {len(zero)}\n", "red")
                        for zf in zero[:5]:
                            self.root.after(0, self._vlog, f"    \u2192 {zf}\n", "red")
                        if len(zero) > 5:
                            self.root.after(0, self._vlog,
                                            f"    ... and {len(zero)-5} more\n", "red")

                    no_lbl = [img for img in imgs
                              if not os.path.isfile(
                                  os.path.join(labels_dir,
                                               os.path.splitext(img)[0] + ".txt"))]
                    if no_lbl:
                        issues = True
                        self.root.after(0, self._vlog,
                                        f"  \u26A0 Images without labels: {len(no_lbl)}\n", "red")
                        for nl in no_lbl[:5]:
                            self.root.after(0, self._vlog, f"    \u2192 {nl}\n", "red")
                        if len(no_lbl) > 5:
                            self.root.after(0, self._vlog,
                                            f"    ... and {len(no_lbl)-5} more\n", "red")
                else:
                    self.root.after(0, self._vlog,
                                    f"  \u274C Labels dir not found:\n  {labels_dir}\n", "red")
                    issues = True

                self.root.after(0, self._vlog, "\n", None)
                time.sleep(0.3)

            self.root.after(0, self._verification_done, not issues)

        except Exception as e:
            print(f"Verification error: {e}")
            self.root.after(0, self._vlog, f"\u274C Error: {e}\n", "red")
            self.root.after(0, self._verification_done, False)

    def _vlog(self, text, tag):
        self.verify_output.config(state="normal")
        if tag:
            self.verify_output.insert("end", text, tag)
        else:
            self.verify_output.insert("end", text)
        self.verify_output.see("end")
        self.verify_output.config(state="disabled")

    def _verification_done(self, success):
        self.verify_progress.stop()
        if success:
            self._vlog("\n\u2705 DATASET READY \u2014 proceed to training\n", "green")
        else:
            self._vlog("\n\u26A0 ISSUES FOUND \u2014 review flagged files\n", "red")
        self._set_status("IDLE", ACCENT)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — TRAIN
    # ═════════════════════════════════════════════════════════════════════════

    def _build_train_tab(self):
        # ── Left controls ──
        left = tk.Frame(self.tab_train, bg=PANEL, width=300)
        left.pack(side="left", fill="y", padx=(10, 5), pady=10)
        left.pack_propagate(False)

        tk.Label(left, text="TRAINING CONFIG", font=FONT_HEADING,
                 fg=ACCENT, bg=PANEL).pack(padx=15, pady=(15, 15), anchor="w")

        def _field(parent, label_text, default):
            tk.Label(parent, text=label_text, font=FONT_SMALL,
                     fg=MUTED, bg=PANEL).pack(padx=15, anchor="w")
            var = tk.StringVar(value=default)
            tk.Entry(parent, textvariable=var, font=FONT_BODY,
                     bg=CARD, fg=TEXT, insertbackground=TEXT,
                     relief="flat", highlightthickness=1,
                     highlightbackground=MUTED).pack(padx=15, pady=(2, 10),
                                                      fill="x", ipady=4)
            return var

        # Model dropdown
        tk.Label(left, text="MODEL", font=FONT_SMALL, fg=MUTED,
                 bg=PANEL).pack(padx=15, anchor="w")
        available_models = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(BASE_DIR)
            if f.endswith(".pt") and os.path.isfile(os.path.join(BASE_DIR, f))
        )
        default_model = available_models[0] if available_models else "yolov9s"
        self.model_var = tk.StringVar(value=default_model)
        self.model_size_var = self.model_var
        ttk.Combobox(left, textvariable=self.model_var,
                      values=available_models if available_models else ["yolov9s"],
                      state="readonly", font=FONT_BODY
                      ).pack(padx=15, pady=(2, 10), fill="x")

        self.epochs_var  = _field(left, "EPOCHS",         "50")
        self.imgsz_var   = _field(left, "IMAGE SIZE",     "640")
        self.imgsize_var = self.imgsz_var
        self.batch_var   = _field(left, "BATCH SIZE",     "4")
        self.savedir_var = _field(left, "SAVE DIRECTORY", "models/")

        self.btn_train = tk.Button(left, text="\u25B6 START TRAINING", font=FONT_BODY,
                                    bg=ACCENT, fg=BG, relief="flat", cursor="hand2",
                                    activebackground="#00cc6a", activeforeground=BG,
                                    command=self._start_training)
        self.btn_train.pack(padx=15, pady=(8, 4), fill="x", ipady=8)

        self.btn_stop_train = tk.Button(left, text="\u25A0 STOP", font=FONT_BODY,
                                         bg=DANGER, fg=TEXT, relief="flat", cursor="hand2",
                                         activebackground="#cc2929",
                                         command=self._stop_training, state="disabled")
        self.btn_stop_train.pack(padx=15, pady=(0, 8), fill="x", ipady=8)

        self.btn_test = tk.Button(left, text="\u25B6 TEST MODEL", font=FONT_BODY,
                                   bg=CARD, fg=MUTED, relief="flat", cursor="hand2",
                                   command=self._test_model, state="disabled")
        self.btn_test.pack(padx=15, pady=(0, 15), fill="x", ipady=8)

        # ── Right panel ──
        right = tk.Frame(self.tab_train, bg=PANEL)
        right.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

        # Charts
        chart_frame = tk.Frame(right, bg=CARD, height=270)
        chart_frame.pack(padx=10, pady=(10, 5), fill="x")
        chart_frame.pack_propagate(False)

        self.fig = Figure(figsize=(8, 2.8), dpi=100, facecolor=CARD)
        self.fig.subplots_adjust(left=0.07, right=0.96, top=0.87, bottom=0.2, wspace=0.3)
        self.ax_loss = self.fig.add_subplot(121)
        self.ax_map  = self.fig.add_subplot(122)
        self._style_axis(self.ax_loss, "Train Loss", "Epoch", "Loss")
        self._style_axis(self.ax_map,  "mAP@50",     "Epoch", "mAP")

        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.chart_canvas.draw()

        # Training log
        tk.Label(right, text="TRAINING LOG", font=FONT_SMALL, fg=MUTED,
                 bg=PANEL).pack(padx=15, pady=(10, 2), anchor="w")

        log_frame = tk.Frame(right, bg=CARD)
        log_frame.pack(padx=10, pady=(0, 10), fill="both", expand=True)

        scrollbar = tk.Scrollbar(log_frame, bg=CARD, troughcolor=CARD)
        scrollbar.pack(side="right", fill="y")

        self.train_log = tk.Text(log_frame, bg=CARD, fg=TEXT, font=FONT_SMALL,
                                  relief="flat", insertbackground=TEXT,
                                  padx=10, pady=8, yscrollcommand=scrollbar.set,
                                  highlightthickness=0)
        self.train_log.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.train_log.yview)
        self.train_log.config(state="disabled")
        self.excel_log_path = None

    def _style_axis(self, ax, title, xlabel, ylabel):
        ax.set_facecolor(CARD)
        ax.set_title(title, color=ACCENT, fontsize=10, fontfamily="Consolas")
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8, fontfamily="Consolas")
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8, fontfamily="Consolas")
        ax.tick_params(colors=MUTED, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
            spine.set_linewidth(0.5)
        ax.grid(True, alpha=0.15, color=MUTED)

    # ── Training control ─────────────────────────────────────────────────────

    def _start_training(self):
        try:
            epochs_val = int(self.epochs_var.get())
            imgsz_val  = int(self.imgsz_var.get())
            batch_val  = int(self.batch_var.get())
            if epochs_val <= 0 or imgsz_val <= 0 or batch_val <= 0:
                raise ValueError("Values must be positive")
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Epochs, Image Size, and Batch Size must be positive integers.")
            return

        self.train_losses.clear()
        self.map_values.clear()
        self.loss_epochs.clear()
        self.map_epochs.clear()
        self.current_training_epoch = 0

        self.btn_train.config(state="disabled")
        self.btn_stop_train.config(state="normal")
        self.btn_test.config(state="disabled", bg=CARD, fg=MUTED)
        self._set_status("TRAINING", WARNING)

        self.train_log.config(state="normal")
        self.train_log.delete("1.0", "end")
        self.train_log.config(state="disabled")

        threading.Thread(target=self._run_training_thread, daemon=True).start()

    def _run_training_thread(self):
        model_size = self.model_size_var.get().strip()
        if not model_size.endswith(".pt"):
            model_size += ".pt"

        epochs = self.epochs_var.get().strip()
        imgsz = self.imgsize_var.get().strip()
        batch = self.batch_var.get().strip()
        project_dir = self.savedir_var.get().strip().replace("\\", "/")
        data_yaml = os.path.abspath(os.path.join(BASE_DIR, "data.yaml")).replace("\\", "/")

        run_name = f"run_{int(time.time())}"
        save_dir = os.path.join(BASE_DIR, self.savedir_var.get().strip())
        self.root.after(0, self._init_excel_log, save_dir, run_name)

        script = (
            "import sys, os; "
            "sys.stdout.reconfigure(encoding='utf-8', line_buffering=True); "
            "from ultralytics import YOLO; "
            f"m = YOLO(r'{model_size}'); "
            f"m.train(data=r'{data_yaml}', epochs={epochs}, imgsz={imgsz}, "
            f"batch={batch}, workers=0, project=r'{project_dir}', name='{run_name}', exist_ok=True, verbose=True)"
        )
        cmd = [sys.executable, "-u", "-c", script]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1,
                cwd=BASE_DIR, env=env
            )
            self.training_process = proc

            epoch_loss_regex = re.compile(r'^\s*(\d+)/(\d+)\s+([\d\.]+G?|0G)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
            map_regex = re.compile(r'^\s*all\s+\d+\s+\d+\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)')

            last_map_val = 0.0
            if proc.stdout:
                for line in iter(proc.stdout.readline, ''):
                    if not line:
                        break

                    self.root.after(0, self._tlog, line.rstrip('\r\n'))

                    loss_match = epoch_loss_regex.search(line)
                    if loss_match:
                        ep = int(loss_match.group(1))
                        box_loss = float(loss_match.group(4))
                        self.current_training_epoch = ep
                        self.root.after(0, self._update_loss_chart, ep, box_loss)
                        self.root.after(0, self._append_excel_row, ep, box_loss, last_map_val)

                    map_match = map_regex.search(line)
                    if map_match:
                        last_map_val = float(map_match.group(1))
                        self.root.after(0, self._update_map_chart, self.current_training_epoch, last_map_val)

            proc.wait()
            rc = proc.returncode

            if rc == 0:
                self.root.after(0, self._training_complete)
            else:
                self.root.after(0, self._tlog, f"\nProcess exited with code {rc}")
                self.root.after(0, self._training_reset)
                self.root.after(0, self._set_status, "TRAINING FAILED", DANGER)

        except Exception as e:
            print(f"Training error: {e}")
            self.root.after(0, self._set_status, f"ERROR: {e}", DANGER)
            self.root.after(0, self._training_reset)

    _training_worker = _run_training_thread

    def _tlog(self, text):
        self.train_log.config(state="normal")
        self.train_log.insert("end", text + "\n")
        self.train_log.see("end")
        self.train_log.config(state="disabled")

    def _update_loss_chart(self, epoch, loss):
        if self.loss_epochs and epoch == self.loss_epochs[-1]:
            self.train_losses[-1] = loss
        else:
            self.loss_epochs.append(epoch)
            self.train_losses.append(loss)

        self.ax_loss.clear()
        self._style_axis(self.ax_loss, "Train Loss", "Epoch", "Loss")
        self.ax_loss.plot(self.loss_epochs, self.train_losses,
                          color=ACCENT, linewidth=1.5, marker="o", markersize=3)
        if len(self.train_losses) > 1:
            self.ax_loss.fill_between(self.loss_epochs, self.train_losses,
                                       alpha=0.08, color=ACCENT)
        self.chart_canvas.draw_idle()

    def _update_map_chart(self, epoch, mAP):
        if self.map_epochs and epoch == self.map_epochs[-1]:
            self.map_values[-1] = mAP
        else:
            self.map_epochs.append(epoch)
            self.map_values.append(mAP)

        self.ax_map.clear()
        self._style_axis(self.ax_map, "mAP@50", "Epoch", "mAP")
        self.ax_map.plot(self.map_epochs, self.map_values,
                         color=WARNING, linewidth=1.5, marker="o", markersize=3)
        if len(self.map_values) > 1:
            self.ax_map.fill_between(self.map_epochs, self.map_values,
                                      alpha=0.08, color=WARNING)
        self.chart_canvas.draw_idle()

    def _init_excel_log(self, save_dir, run_name):
        """Create a new Excel workbook for this training run."""
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment
            from openpyxl.utils import get_column_letter

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Training Log"

            # Header row styling
            headers = ["Epoch", "Box Loss", "mAP@50", "Timestamp"]
            header_fill = PatternFill("solid", fgColor="0A0A0A")
            header_font = Font(name="Consolas", bold=True, color="00FF88", size=11)

            for col, h in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col, value=h)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")
                ws.column_dimensions[get_column_letter(col)].width = 18

            # Metadata sheet
            ws_meta = wb.create_sheet("Run Info")
            ws_meta["A1"] = "SENTINEL AI — Training Run"
            ws_meta["A2"] = f"Run Name: {run_name}"
            ws_meta["A3"] = f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            ws_meta["A4"] = f"Model: {self.model_var.get()}"
            ws_meta["A5"] = f"Epochs: {self.epochs_var.get()}"
            ws_meta["A6"] = f"Image Size: {self.imgsz_var.get()}"
            ws_meta["A7"] = f"Batch Size: {self.batch_var.get()}"

            os.makedirs(save_dir, exist_ok=True)
            self.excel_log_path = os.path.join(save_dir, f"{run_name}_training_log.xlsx")
            wb.save(self.excel_log_path)
            self._tlog(f"[EXCEL] Log started: {self.excel_log_path}")
        except Exception as e:
            self._tlog(f"[EXCEL] Could not create log: {e}")
            self.excel_log_path = None

    def _append_excel_row(self, epoch, loss, mAP):
        """Append one epoch row to the Excel log."""
        if not self.excel_log_path or not os.path.isfile(self.excel_log_path):
            return
        try:
            import openpyxl
            from openpyxl.styles import PatternFill, Font

            wb = openpyxl.load_workbook(self.excel_log_path)
            ws = wb["Training Log"]
            row = ws.max_row + 1

            # Alternating row colors
            fill_color = "1A1A1A" if row % 2 == 0 else "111111"
            fill = PatternFill("solid", fgColor=fill_color)
            font = Font(name="Consolas", color="E0E0E0", size=10)

            values = [epoch, round(loss, 6), round(mAP, 6),
                      time.strftime('%H:%M:%S')]
            for col, val in enumerate(values, 1):
                cell = ws.cell(row=row, column=col, value=val)
                cell.fill = fill
                cell.font = font

            wb.save(self.excel_log_path)
        except Exception as e:
            print(f"[EXCEL] Row append failed: {e}")

    def _training_complete(self):
        self._training_reset()
        self.btn_test.config(state="normal", bg=ACCENT, fg=BG)
        self._set_status("IDLE", ACCENT)
        self._export_training_graphs()
        messagebox.showinfo("Training Complete",
                            f"✅ Training Complete\n"
                            f"Model saved to {self.savedir_var.get()}\n"
                            f"Excel log: {os.path.basename(self.excel_log_path) if self.excel_log_path else 'N/A'}\n"
                            f"Graphs saved to models/ folder")

    def _export_training_graphs(self):
        """Save loss and mAP charts as PNG files to the models/ folder."""
        try:
            import matplotlib.pyplot as plt

            graphs_dir = os.path.join(BASE_DIR, "models")
            os.makedirs(graphs_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")

            # Loss graph
            if self.loss_epochs and self.train_losses:
                fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0a0a0a")
                ax.set_facecolor("#1a1a1a")
                ax.plot(self.loss_epochs, self.train_losses,
                        color="#00ff88", linewidth=2, marker="o", markersize=4)
                ax.fill_between(self.loss_epochs, self.train_losses,
                                alpha=0.1, color="#00ff88")
                ax.set_title("SENTINEL AI — Training Loss", color="#00ff88",
                             fontsize=14, fontfamily="monospace")
                ax.set_xlabel("Epoch", color="#666666", fontfamily="monospace")
                ax.set_ylabel("Box Loss", color="#666666", fontfamily="monospace")
                ax.tick_params(colors="#666666")
                for spine in ax.spines.values():
                    spine.set_color("#666666")
                ax.grid(True, alpha=0.15, color="#666666")
                loss_path = os.path.join(graphs_dir, f"loss_curve_{ts}.png")
                fig.savefig(loss_path, dpi=150, bbox_inches="tight",
                            facecolor="#0a0a0a")
                plt.close(fig)
                self._tlog(f"[GRAPH] Loss curve saved: {loss_path}")

            # mAP graph
            if self.map_epochs and self.map_values:
                fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0a0a0a")
                ax.set_facecolor("#1a1a1a")
                ax.plot(self.map_epochs, self.map_values,
                        color="#ffaa00", linewidth=2, marker="o", markersize=4)
                ax.fill_between(self.map_epochs, self.map_values,
                                alpha=0.1, color="#ffaa00")
                ax.set_title("SENTINEL AI — mAP@50", color="#ffaa00",
                             fontsize=14, fontfamily="monospace")
                ax.set_xlabel("Epoch", color="#666666", fontfamily="monospace")
                ax.set_ylabel("mAP@50", color="#666666", fontfamily="monospace")
                ax.tick_params(colors="#666666")
                for spine in ax.spines.values():
                    spine.set_color("#666666")
                ax.grid(True, alpha=0.15, color="#666666")
                map_path = os.path.join(graphs_dir, f"map_curve_{ts}.png")
                fig.savefig(map_path, dpi=150, bbox_inches="tight",
                            facecolor="#0a0a0a")
                plt.close(fig)
                self._tlog(f"[GRAPH] mAP curve saved: {map_path}")

        except Exception as e:
            self._tlog(f"[GRAPH] Export failed: {e}")

    def _training_reset(self):
        self.btn_train.config(state="normal")
        self.btn_stop_train.config(state="disabled")
        self.training_process = None

    def _stop_training(self):
        if self.training_process:
            try:
                self.training_process.terminate()
            except Exception:
                pass
            self._set_status("TRAINING STOPPED", DANGER)
            self._training_reset()

    def _test_model(self):
        filepath = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filepath:
            return

        bag_model, person_model, status = load_dual_models(BASE_DIR)
        model = bag_model if bag_model is not None else person_model
        if model is None:
            messagebox.showerror("Error", status)
            return

        results = model(filepath, verbose=False)
        result_img = results[0].plot()

        win = tk.Toplevel(self.root)
        win.title("Inference Result")
        win.configure(bg=BG)

        rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(900 / w, 650 / h, 1.0)
        resized = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        photo = ImageTk.PhotoImage(Image.fromarray(resized))

        lbl = tk.Label(win, image=photo, bg=BG)
        lbl.image = photo
        lbl.pack(padx=10, pady=10)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — VIDEO DEMO
    # ═════════════════════════════════════════════════════════════════════════

    def _build_video_tab(self):
        # ── Controls bar ──
        ctrl = tk.Frame(self.tab_video, bg=PANEL, height=55)
        ctrl.pack(fill="x", padx=10, pady=(10, 5))
        ctrl.pack_propagate(False)

        tk.Button(ctrl, text="\U0001F4C1 UPLOAD VIDEO", font=FONT_BODY,
                  bg=CARD, fg=TEXT, relief="flat", cursor="hand2",
                  activebackground=PANEL, command=self._upload_video
                  ).pack(side="left", padx=(10, 5), pady=8)

        self.video_fname = tk.Label(ctrl, text="No file selected", font=FONT_SMALL,
                                     fg=MUTED, bg=PANEL)
        self.video_fname.pack(side="left", padx=5)

        self.btn_vid_play = tk.Button(ctrl, text="\u25B6 PLAY", font=FONT_BODY,
                                       bg=ACCENT, fg=BG, relief="flat", cursor="hand2",
                                       activebackground="#00cc6a",
                                       command=self._play_video, state="disabled")
        self.btn_vid_play.pack(side="left", padx=5, pady=8)

        self.btn_vid_stop = tk.Button(ctrl, text="\u25A0 STOP", font=FONT_BODY,
                                       bg=DANGER, fg=TEXT, relief="flat", cursor="hand2",
                                       activebackground="#cc2929",
                                       command=self._stop_video, state="disabled")
        self.btn_vid_stop.pack(side="left", padx=5, pady=8)

        tk.Label(ctrl, text="CONFIDENCE", font=FONT_SMALL, fg=MUTED,
                 bg=PANEL).pack(side="left", padx=(20, 2))
        self.vid_conf = tk.DoubleVar(value=0.25)
        tk.Scale(ctrl, from_=0.1, to=0.9, resolution=0.05, orient="horizontal",
                 variable=self.vid_conf, bg=PANEL, fg=ACCENT, troughcolor=CARD,
                 highlightthickness=0, font=FONT_SMALL, length=110, sliderlength=14
                 ).pack(side="left", padx=2)

        # ── Main Layout (Split Frame) ──
        main_layout = tk.Frame(self.tab_video, bg=BG)
        main_layout.pack(fill="both", expand=True, padx=10, pady=5)

        # Left Column (Video canvas + charts + alert log)
        left_col = tk.Frame(main_layout, bg=BG)
        left_col.pack(side="left", fill="both", expand=True)

        self.vid_canvas = tk.Canvas(left_col, bg="black", highlightthickness=0)
        self.vid_canvas.pack(pady=5, fill="both", expand=True)

        hist_frame = tk.Frame(left_col, bg=CARD, height=105)
        hist_frame.pack(pady=(0, 5), fill="x")
        hist_frame.pack_propagate(False)

        tk.Label(hist_frame, text="CONFIDENCE DISTRIBUTION", font=FONT_SMALL,
                 fg=MUTED, bg=CARD).pack(anchor="w", padx=8, pady=(4, 0))

        self.vid_hist_fig = Figure(figsize=(8, 0.75), dpi=100, facecolor=CARD)
        self.vid_hist_fig.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.25)
        self.vid_hist_ax = self.vid_hist_fig.add_subplot(111)
        self.vid_hist_ax.set_facecolor(BG)
        self.vid_hist_ax.tick_params(colors=MUTED, labelsize=7)
        for spine in self.vid_hist_ax.spines.values():
            spine.set_color(MUTED)
            spine.set_linewidth(0.4)

        self.vid_hist_canvas = FigureCanvasTkAgg(self.vid_hist_fig, master=hist_frame)
        self.vid_hist_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.vid_hist_canvas.draw()
        self.vid_conf_history = []   # rolling confidence values

        tk.Label(left_col, text="ALERT LOG", font=FONT_SMALL, fg=MUTED,
                 bg=BG).pack(anchor="w")
        self.vid_alert_log = tk.Text(left_col, bg=CARD, fg=DANGER, font=FONT_SMALL,
                                      relief="flat", height=5, padx=10, pady=5,
                                      highlightthickness=0)
        self.vid_alert_log.pack(fill="x")
        self.vid_alert_log.config(state="disabled")

        # Right Column (Cyber-HUD Sidebar Behavior Panel)
        right_col = tk.Frame(main_layout, bg=PANEL, width=320)
        right_col.pack(side="right", fill="both", expand=False, padx=(10, 0))
        right_col.pack_propagate(False)

        tk.Label(right_col, text="TACTICAL BEHAVIOR HUD", font=FONT_HEADING,
                 fg=ACCENT, bg=PANEL).pack(pady=(15, 10))

        # Active Targets Display
        tk.Label(right_col, text="ACTIVE TARGETS", font=FONT_SMALL, fg=MUTED,
                 bg=PANEL).pack(anchor="w", padx=15, pady=(10, 2))
        self.vid_targets_log = tk.Text(right_col, bg=CARD, fg=TEXT, font=FONT_BODY,
                                       relief="flat", height=10, padx=10, pady=5,
                                       highlightthickness=0)
        self.vid_targets_log.pack(fill="x", padx=10, pady=2)
        self.vid_targets_log.config(state="disabled")

        # Global Risk Meter
        tk.Label(right_col, text="GLOBAL RISK SEVERITY", font=FONT_SMALL, fg=MUTED,
                 bg=PANEL).pack(anchor="w", padx=15, pady=(10, 2))
        self.vid_risk_lbl = tk.Label(right_col, text="NORMAL", font=("Consolas", 20, "bold"),
                                     fg=ACCENT, bg=CARD, pady=12)
        self.vid_risk_lbl.pack(fill="x", padx=10, pady=2)

        # Recent Behavior Events
        tk.Label(right_col, text="RECENT BEHAVIOR EVENTS", font=FONT_SMALL, fg=MUTED,
                 bg=PANEL).pack(anchor="w", padx=15, pady=(10, 2))
        self.vid_events_log = tk.Text(right_col, bg=CARD, fg=WARNING, font=FONT_SMALL,
                                      relief="flat", height=6, padx=10, pady=5,
                                      highlightthickness=0)
        self.vid_events_log.pack(fill="both", expand=True, padx=10, pady=(2, 15))
        self.vid_events_log.config(state="disabled")

        self.video_path = None
        self.vid_tracked_entities = {}
        self.vid_fcount = 0

    def _upload_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mpg"), ("All Files", "*.*")]
        )
        if path:
            self.video_path = path
            self.video_fname.config(text=os.path.basename(path), fg=TEXT)
            self.btn_vid_play.config(state="normal")

    def _play_video(self):
        if not self.video_path:
            return

        bag_model, person_model, status = load_dual_models(BASE_DIR)
        if bag_model is None and person_model is None:
            self._canvas_msg(self.vid_canvas, f"Models failed to load: {status}")
            return

        self.video_running = True
        self.vid_tracked_entities = {}
        self.vid_fcount = 0
        self.btn_vid_play.config(state="disabled")
        self.btn_vid_stop.config(state="normal")
        self._set_status("DETECTING", WARNING)

        threading.Thread(target=self._video_worker, args=(bag_model, person_model), daemon=True).start()

    def _video_worker(self, bag_model, person_model):
        cap = None
        try:
            if not self.video_path or not os.path.isfile(self.video_path):
                self.root.after(0, self._canvas_msg, self.vid_canvas, "Video file not found or invalid.")
                self.root.after(0, self._set_status, "FILE NOT FOUND", DANGER)
                return

            cap = cv2.VideoCapture(self.video_path)
            if not cap or not cap.isOpened():
                self.root.after(0, self._canvas_msg, self.vid_canvas, f"Cannot open video:\n{os.path.basename(self.video_path)}")
                self.root.after(0, self._set_status, "CANNOT OPEN VIDEO", DANGER)
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            delay = 1.0 / max(fps, 1)

            while self.video_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    break

                self.vid_fcount += 1
                
                # Model inference
                bag_dets = detect_bags(frame, bag_model, self.vid_conf.get())
                person_dets = detect_persons(frame, person_model)
                
                # Combine & Deduplicate overlapping boxes
                merged_dets = bag_dets + person_dets
                merged_dets = iou_deduplicate(merged_dets)

                # Tracking match
                self.vid_tracked_entities = match_entities(merged_dets, self.vid_tracked_entities)
                
                # Behavior analysis and overlay drawing
                annotated, new_alerts = draw_overlays(frame, self.vid_tracked_entities, self.vid_fcount)
                self._last_frame = annotated.copy()

                # Update live HUD statistics on every frame
                self.root.after(0, self._update_hud_sidebar, "vid", self.vid_tracked_entities)

                self.root.after(0, self._update_conf_histogram,
                                bag_dets, person_dets,
                                self.vid_hist_ax, self.vid_hist_canvas,
                                self.vid_conf_history)

                # Process newly triggered suspicious behavior alerts
                for alert in new_alerts:
                    # Critical unattended bag triggers alarm & snapshot
                    is_critical_unattended = (alert["behavior"] == "UNATTENDED_OBJECT" and alert["severity"] == "CRITICAL")
                    is_high_threat = (alert["severity"] in ("HIGH", "CRITICAL"))
                    
                    if is_critical_unattended or is_high_threat:
                        snapshot = annotated.copy()
                        self._save_incident_snapshot(snapshot, alert["behavior"], alert["track_id"], alert["severity"])
                        self.root.after(0, self._play_alert_sound)

                    self.root.after(0, self._fire_behavior_alert, "vid", alert)

                self.root.after(0, self._show_frame, annotated, self.vid_canvas)
                time.sleep(delay)

        except Exception as e:
            print(f"Video error: {e}")
            self.root.after(0, self._set_status, f"ERROR: {e}", DANGER)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.root.after(0, self._video_stopped)

    def _stop_video(self):
        self.video_running = False
        self._stop_alert_sound()

    def _video_stopped(self):
        self.video_running = False
        self.btn_vid_play.config(state="normal")
        self.btn_vid_stop.config(state="disabled")
        self._set_status("IDLE", ACCENT)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4 — LIVE CAMERA
    # ═════════════════════════════════════════════════════════════════════════

    def _build_camera_tab(self):
        # ── Controls bar ──
        ctrl = tk.Frame(self.tab_camera, bg=PANEL, height=55)
        ctrl.pack(fill="x", padx=10, pady=(10, 5))
        ctrl.pack_propagate(False)

        self.btn_cam_start = tk.Button(ctrl, text="\u25B6 START CAMERA", font=FONT_BODY,
                                        bg=ACCENT, fg=BG, relief="flat", cursor="hand2",
                                        activebackground="#00cc6a",
                                        command=self._start_camera)
        self.btn_cam_start.pack(side="left", padx=(10, 5), pady=8)

        self.btn_cam_stop = tk.Button(ctrl, text="\u25A0 STOP CAMERA", font=FONT_BODY,
                                       bg=DANGER, fg=TEXT, relief="flat", cursor="hand2",
                                       activebackground="#cc2929",
                                       command=self._stop_camera, state="disabled")
        self.btn_cam_stop.pack(side="left", padx=5, pady=8)

        tk.Label(ctrl, text="CAMERA", font=FONT_SMALL, fg=MUTED,
                 bg=PANEL).pack(side="left", padx=(20, 5))
        self.cam_idx_var = tk.StringVar(value="0")
        ttk.Combobox(ctrl, textvariable=self.cam_idx_var,
                      values=["0", "1", "2"], state="readonly",
                      font=FONT_BODY, width=3).pack(side="left", padx=2)

        tk.Label(ctrl, text="CONFIDENCE", font=FONT_SMALL, fg=MUTED,
                 bg=PANEL).pack(side="left", padx=(20, 2))
        self.cam_conf = tk.DoubleVar(value=0.25)
        tk.Scale(ctrl, from_=0.1, to=0.9, resolution=0.05, orient="horizontal",
                 variable=self.cam_conf, bg=PANEL, fg=ACCENT, troughcolor=CARD,
                 highlightthickness=0, font=FONT_SMALL, length=110, sliderlength=14
                 ).pack(side="left", padx=2)

        # ── Main Layout (Split Frame) ──
        main_layout = tk.Frame(self.tab_camera, bg=BG)
        main_layout.pack(fill="both", expand=True, padx=10, pady=5)

        # Left Column
        left_col = tk.Frame(main_layout, bg=BG)
        left_col.pack(side="left", fill="both", expand=True)

        self.cam_canvas = tk.Canvas(left_col, bg="black", highlightthickness=0)
        self.cam_canvas.pack(pady=5, fill="both", expand=True)

        hist_frame = tk.Frame(left_col, bg=CARD, height=105)
        hist_frame.pack(pady=(0, 5), fill="x")
        hist_frame.pack_propagate(False)

        tk.Label(hist_frame, text="CONFIDENCE DISTRIBUTION", font=FONT_SMALL,
                 fg=MUTED, bg=CARD).pack(anchor="w", padx=8, pady=(4, 0))

        self.cam_hist_fig = Figure(figsize=(8, 0.75), dpi=100, facecolor=CARD)
        self.cam_hist_fig.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.25)
        self.cam_hist_ax = self.cam_hist_fig.add_subplot(111)
        self.cam_hist_ax.set_facecolor(BG)
        self.cam_hist_ax.tick_params(colors=MUTED, labelsize=7)
        for spine in self.cam_hist_ax.spines.values():
            spine.set_color(MUTED)
            spine.set_linewidth(0.4)

        self.cam_hist_canvas = FigureCanvasTkAgg(self.cam_hist_fig, master=hist_frame)
        self.cam_hist_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.cam_hist_canvas.draw()
        self.cam_conf_history = []

        tk.Label(left_col, text="ALERT LOG", font=FONT_SMALL, fg=MUTED,
                 bg=BG).pack(anchor="w")
        self.cam_alert_log = tk.Text(left_col, bg=CARD, fg=DANGER, font=FONT_SMALL,
                                      relief="flat", height=5, padx=10, pady=5,
                                      highlightthickness=0)
        self.cam_alert_log.pack(fill="x")
        self.cam_alert_log.config(state="disabled")

        # Right Column (Cyber-HUD Sidebar Behavior Panel)
        right_col = tk.Frame(main_layout, bg=PANEL, width=320)
        right_col.pack(side="right", fill="both", expand=False, padx=(10, 0))
        right_col.pack_propagate(False)

        tk.Label(right_col, text="TACTICAL BEHAVIOR HUD", font=FONT_HEADING,
                 fg=ACCENT, bg=PANEL).pack(pady=(15, 10))

        tk.Label(right_col, text="ACTIVE TARGETS", font=FONT_SMALL, fg=MUTED,
                 bg=PANEL).pack(anchor="w", padx=15, pady=(10, 2))
        self.cam_targets_log = tk.Text(right_col, bg=CARD, fg=TEXT, font=FONT_BODY,
                                       relief="flat", height=10, padx=10, pady=5,
                                       highlightthickness=0)
        self.cam_targets_log.pack(fill="x", padx=10, pady=2)
        self.cam_targets_log.config(state="disabled")

        tk.Label(right_col, text="GLOBAL RISK SEVERITY", font=FONT_SMALL, fg=MUTED,
                 bg=PANEL).pack(anchor="w", padx=15, pady=(10, 2))
        self.cam_risk_lbl = tk.Label(right_col, text="NORMAL", font=("Consolas", 20, "bold"),
                                     fg=ACCENT, bg=CARD, pady=12)
        self.cam_risk_lbl.pack(fill="x", padx=10, pady=2)

        tk.Label(right_col, text="RECENT BEHAVIOR EVENTS", font=FONT_SMALL, fg=MUTED,
                 bg=PANEL).pack(anchor="w", padx=15, pady=(10, 2))
        self.cam_events_log = tk.Text(right_col, bg=CARD, fg=WARNING, font=FONT_SMALL,
                                      relief="flat", height=6, padx=10, pady=5,
                                      highlightthickness=0)
        self.cam_events_log.pack(fill="both", expand=True, padx=10, pady=(2, 15))
        self.cam_events_log.config(state="disabled")

        self.cam_tracked_entities = {}
        self.cam_fcount = 0

    def _start_camera(self):
        bag_model, person_model, status = load_dual_models(BASE_DIR)
        if bag_model is None and person_model is None:
            self.root.after(50, lambda: self._canvas_msg(
                self.cam_canvas, f"Models failed to load: {status}"))
            return

        cam_idx_str = self.cam_idx_var.get()
        try:
            cam_idx = int(cam_idx_str)
        except (ValueError, TypeError):
            cam_idx = 0

        self.camera_running = True
        self.cam_tracked_entities = {}
        self.cam_fcount = 0
        self.btn_cam_start.config(state="disabled")
        self.btn_cam_stop.config(state="normal")
        self._set_status("CONNECTING TO CAMERA...", WARNING)

        threading.Thread(target=self._camera_worker,
                         args=(bag_model, person_model, cam_idx), daemon=True).start()

    def _camera_worker(self, bag_model, person_model, cam_idx):
        cap = None
        try:
            cap = cv2.VideoCapture(cam_idx)
            if not cap or not cap.isOpened():
                self.root.after(0, self._canvas_msg, self.cam_canvas,
                                f"\u26A0 No camera detected at index {cam_idx}\nPlease check webcam connection or select another index.")
                self.root.after(0, self._set_status, f"CAMERA NOT FOUND (idx {cam_idx})", DANGER)
                return

            # Test-read first frame
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                self.root.after(0, self._canvas_msg, self.cam_canvas,
                                f"\u26A0 Failed to read feed from camera index {cam_idx}")
                self.root.after(0, self._set_status, f"CAMERA READ FAILED (idx {cam_idx})", DANGER)
                return

            self.root.after(0, self._set_status, "CAMERA ACTIVE", ACCENT)

            failed_reads = 0
            while self.camera_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    failed_reads += 1
                    if failed_reads > 30:
                        self.root.after(0, self._canvas_msg, self.cam_canvas,
                                        f"\u26A0 Camera feed lost or disconnected (index {cam_idx})")
                        self.root.after(0, self._set_status, "CAMERA DISCONNECTED", DANGER)
                        break
                    time.sleep(0.03)
                    continue

                failed_reads = 0
                self.cam_fcount += 1
                
                bag_dets = detect_bags(frame, bag_model, self.cam_conf.get())
                person_dets = detect_persons(frame, person_model)
                
                merged_dets = bag_dets + person_dets
                merged_dets = iou_deduplicate(merged_dets)

                self.cam_tracked_entities = match_entities(merged_dets, self.cam_tracked_entities)
                annotated, new_alerts = draw_overlays(frame, self.cam_tracked_entities, self.cam_fcount)
                self._last_frame = annotated.copy()

                self.root.after(0, self._update_hud_sidebar, "cam", self.cam_tracked_entities)

                self.root.after(0, self._update_conf_histogram,
                                bag_dets, person_dets,
                                self.cam_hist_ax, self.cam_hist_canvas,
                                self.cam_conf_history)

                for alert in new_alerts:
                    is_critical_unattended = (alert["behavior"] == "UNATTENDED_OBJECT" and alert["severity"] == "CRITICAL")
                    is_high_threat = (alert["severity"] in ("HIGH", "CRITICAL"))
                    
                    if is_critical_unattended or is_high_threat:
                        snapshot = annotated.copy()
                        self._save_incident_snapshot(snapshot, alert["behavior"], alert["track_id"], alert["severity"])
                        self.root.after(0, self._play_alert_sound)

                    self.root.after(0, self._fire_behavior_alert, "cam", alert)

                self.root.after(0, self._show_frame, annotated, self.cam_canvas)
                time.sleep(0.03)  # ~30 fps cap

        except Exception as e:
            print(f"Camera error: {e}")
            self.root.after(0, self._set_status, f"ERROR: {e}", DANGER)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.root.after(0, self._camera_stopped)

    def _stop_camera(self):
        self.camera_running = False
        self._stop_alert_sound()

    def _camera_stopped(self):
        self.camera_running = False
        self.btn_cam_start.config(state="normal")
        self.btn_cam_stop.config(state="disabled")
        self._set_status("IDLE", ACCENT)

    # ═════════════════════════════════════════════════════════════════════════
    # SHARED DISPLAY HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    def _show_frame(self, frame, canvas):
        """Render a cv2 frame onto a Tk canvas, centred and aspect-ratio correct."""
        try:
            canvas.update_idletasks()
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 10 or ch < 10:
                return

            photo = frame_to_tk(frame, cw, ch)
            if photo is None:
                return
            canvas.delete("all")
            canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
            canvas._photo = photo  # prevent GC
        except (tk.TclError, Exception):
            pass

    def _canvas_msg(self, canvas, text):
        """Show a centred text message on a canvas."""
        try:
            canvas.update_idletasks()
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            canvas.delete("all")
            canvas.create_text(max(cw // 2, 100), max(ch // 2, 50),
                               text=text, fill=DANGER, font=FONT_HEADING,
                               width=max(cw - 40, 200))
        except (tk.TclError, Exception):
            pass

    def _update_conf_histogram(self, bag_dets, person_dets, hist_ax, hist_canvas, conf_history):
        """Update the confidence histogram with latest detections."""
        try:
            new_confs = [d[4] for d in bag_dets] + [d[4] for d in person_dets]
            conf_history.extend(new_confs)
            if len(conf_history) > 150:
                del conf_history[:-150]

            hist_ax.clear()
            hist_ax.set_facecolor(BG)
            for spine in hist_ax.spines.values():
                spine.set_color(MUTED)
                spine.set_linewidth(0.4)
            hist_ax.tick_params(colors=MUTED, labelsize=7)

            if conf_history:
                bins = [i/10 for i in range(1, 11)]
                counts = [0] * 9
                for c in conf_history:
                    idx = min(int((c - 0.1) / 0.1), 8)
                    if 0 <= idx < 9:
                        counts[idx] += 1
                bar_colors = []
                for i, b in enumerate(bins[:-1]):
                    mid = b + 0.05
                    if mid < 0.5:
                        bar_colors.append("#ffaa00")
                    elif mid < 0.75:
                        bar_colors.append("#00aaff")
                    else:
                        bar_colors.append("#00ff88")
                hist_ax.bar([b + 0.05 for b in bins[:-1]], counts,
                            width=0.08, color=bar_colors, alpha=0.85)
                hist_ax.set_xlim(0.1, 1.0)
                hist_ax.set_xlabel("Confidence", color=MUTED,
                                   fontsize=7, fontfamily="monospace")

            hist_canvas.draw_idle()
        except Exception as e:
            print(f"Histogram error: {e}")

    def _update_hud_sidebar(self, stream_type, tracked_entities):
        """Update the active target list and global threat severity gauge in the right Cyber-HUD sidebar panel."""
        targets_text = ""
        max_risk_score = 0
        global_severity = "NORMAL"
        severity_color = ACCENT

        # Retrieve text widget and risk label depending on tab stream type
        targets_log = getattr(self, f"{stream_type}_targets_log", None)
        risk_lbl = getattr(self, f"{stream_type}_risk_lbl", None)

        if not targets_log or not risk_lbl:
            return

        # Build lines for each tracked entity
        for key, info in sorted(tracked_entities.items()):
            if info.get("alert_dismissed", False):
                continue
            
            cls_name = info["class_name"]
            track_id = info["track_id"]
            threats = info.get("active_threats", set())
            
            status = "NORMAL"
            if threats:
                status = ", ".join(list(threats))

            targets_text += f"{cls_name.upper()} #{track_id:<3} - {status}\n"
            
            # Keep track of global maximum risk severity
            risk_score, severity, color = calculate_entity_risk(info)
            if risk_score > max_risk_score:
                max_risk_score = risk_score
                global_severity = severity
                severity_color = color

        # Update HUD active targets panel
        targets_log.config(state="normal")
        targets_log.delete("1.0", "end")
        targets_log.insert("1.0", targets_text)
        targets_log.config(state="disabled")

        # Convert severity color to Hex for Tkinter safely
        if isinstance(severity_color, str) and severity_color.startswith("#"):
            hex_color = severity_color
        elif isinstance(severity_color, (tuple, list)) and len(severity_color) == 3:
            b, g, r = severity_color
            hex_color = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        else:
            hex_color = ACCENT

        # Update HUD risk gauge panel
        risk_lbl.config(text=f"{global_severity} ({max_risk_score}%)", fg=hex_color)

    def _fire_behavior_alert(self, stream_type, alert):
        """
        Log alerts in log widgets, update statistics counters, and pop up Acknowledge window
        for HIGH/CRITICAL threat events.
        """
        ts = time.strftime("%H:%M:%S")
        behavior = alert["behavior"]
        severity = alert["severity"]
        track_id = alert["track_id"]
        class_name = alert["class_name"]

        alert_msg = f"[{ts}] ⚠ {class_name.upper()} #{track_id} - SUSPICIOUS {behavior} ({severity})\n"

        # Log into primary alert log (bottom text box)
        alert_log = getattr(self, f"{stream_type}_alert_log", None)
        if alert_log:
            self._write_log(alert_log, alert_msg)

        # Log into tactical HUD events log (sidebar)
        events_log = getattr(self, f"{stream_type}_events_log", None)
        if events_log:
            self._write_log(events_log, f"[{ts}] {class_name.upper()} #{track_id} {behavior}\n")

        # Update stats
        self._update_alert_stats(behavior.lower(), f"{class_name.upper()}#{track_id}", severity)

        # Trigger popup acknowledgement window for HIGH and CRITICAL alerts
        if severity in ("HIGH", "CRITICAL"):
            self.root.after(0, self._show_dismiss_popup,
                            f"SUSPICIOUS {behavior} DETECTED\nTarget: {class_name.upper()} #{track_id}\nSeverity: {severity}",
                            class_name, track_id)

    def _write_log(self, log_widget, msg):
        log_widget.config(state="normal")
        log_widget.insert("end", msg)
        log_widget.see("end")
        log_widget.config(state="disabled")

    def _play_alert_sound(self):
        """Start the shared alert sound loop if not already running."""
        if self._alert_sound_active:
            return  # already looping
        self._alert_sound_active = True

        def sound_loop():
            while self._alert_sound_active:
                try:
                    if HAS_WINSOUND:
                        winsound.Beep(2000, 150)
                        time.sleep(0.04)
                        winsound.Beep(2500, 150)
                        time.sleep(0.04)
                        winsound.Beep(3000, 250)
                    else:
                        print('\a', end='', flush=True)
                except Exception:
                    pass
                for _ in range(30):
                    if not self._alert_sound_active:
                        return
                    time.sleep(0.1)

        self._alert_sound_thread = threading.Thread(target=sound_loop, daemon=True)
        self._alert_sound_thread.start()

    def _stop_alert_sound(self):
        """Stop the shared alert sound loop."""
        self._alert_sound_active = False
        self._alert_sound_thread = None

    def _show_dismiss_popup(self, message, alert_type, target_id):
        popup = tk.Toplevel(self.root)
        popup.title("⚠ SENTINEL ALERT")
        popup.geometry("500x220")
        popup.configure(bg=CARD)
        popup.grab_set()
        popup.resizable(False, False)

        tk.Label(popup, text="⚠ ALERT TRIGGERED", font=("Consolas", 14, "bold"),
                 fg=DANGER, bg=CARD).pack(pady=(20, 5))
        tk.Label(popup, text=message, font=("Consolas", 12),
                 fg=TEXT, bg=CARD, justify="center").pack(pady=5)
        tk.Label(popup, text="Assign an officer to investigate immediately.",
                 font=("Consolas", 10), fg=MUTED, bg=CARD).pack(pady=5)

        def dismiss():
            self._stop_alert_sound()
            
            # Dismiss alert inside unified tracked entities dictionaries
            vid_ents = getattr(self, "vid_tracked_entities", {})
            cam_ents = getattr(self, "cam_tracked_entities", {})
            
            for ents in [vid_ents, cam_ents]:
                for key, info in ents.items():
                    if info["class_name"] == alert_type and info["track_id"] == target_id:
                        info["alert_dismissed"] = True
                        info["active_threats"].clear()
                        
            popup.destroy()

        tk.Button(popup, text="✓ ACKNOWLEDGED — DISMISS ALERT",
                  font=("Consolas", 11, "bold"), bg=ACCENT, fg=BG,
                  relief="flat", cursor="hand2", padx=20, pady=8,
                  command=dismiss).pack(pady=15)

    def _save_incident_snapshot(self, frame, behavior, track_id, severity):
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return
        incidents_dir = os.path.join(BASE_DIR, "incidents")
        os.makedirs(incidents_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"incident_{behavior}_{track_id}_{severity}_{ts}.jpg"
        filepath = os.path.join(incidents_dir, filename)
        try:
            snapshot = frame.copy()
            label = f"{behavior} | {severity} | ID: {track_id} | {time.strftime('%Y-%m-%d %H:%M:%S')}"
            cv2.putText(snapshot, label, (10, snapshot.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(snapshot, "SENTINEL AI — INCIDENT RECORD", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            success = cv2.imwrite(filepath, snapshot)
            if not success:
                print(f"[WARN] cv2.imwrite failed for {filepath}")
        except Exception as e:
            print(f"[ERROR] Snapshot save failed: {e}")

    def _build_stats_tab(self):
        self.stats_vars = {
            "total_bags": tk.IntVar(value=0),
            "total_persons": tk.IntVar(value=0),
            "total_alerts": tk.IntVar(value=0),       # Unattended Objects
            "total_loiters": tk.IntVar(value=0),      # Loitering
            "total_running": tk.IntVar(value=0),      # Rapid Movement
            "total_pacing": tk.IntVar(value=0),       # Repeated Movement
            "total_zone_entries": tk.IntVar(value=0), # Restricted Zone Violations
            "session_start": time.time(),
        }
        self.alert_history = []  # log of alert events

        header = tk.Frame(self.tab_stats, bg=PANEL, height=50)
        header.pack(fill="x", padx=10, pady=(10, 5))
        header.pack_propagate(False)
        tk.Label(header, text="SESSION STATISTICS", font=FONT_HEADING,
                 fg=ACCENT, bg=PANEL).pack(side="left", padx=15, pady=10)

        btn_frame = tk.Frame(self.tab_stats, bg=BG)
        btn_frame.pack(fill="x", padx=10, pady=5)
        tk.Button(btn_frame, text="⟳ REFRESH", font=FONT_BODY, bg=CARD, fg=ACCENT,
                  relief="flat", cursor="hand2",
                  command=self._refresh_stats).pack(side="left", padx=5)
        tk.Button(btn_frame, text="⬇ EXPORT PDF REPORT", font=FONT_BODY,
                  bg=CARD, fg=WARNING, relief="flat", cursor="hand2",
                  command=self._export_pdf).pack(side="left", padx=5)

        cards = tk.Frame(self.tab_stats, bg=BG)
        cards.pack(fill="x", padx=10, pady=10)
        
        stat_items = [
            ("BAGS DETECTED", "total_bags", ACCENT),
            ("PERSONS TRACKED", "total_persons", "#44aaff"),
            ("UNATTENDED BAGS", "total_alerts", DANGER),
            ("LOITERING ALERTS", "total_loiters", WARNING),
            ("RAPID MOVEMENT", "total_running", WARNING),
            ("REPEATED PACING", "total_pacing", WARNING),
            ("ZONE VIOLATIONS", "total_zone_entries", DANGER),
        ]
        
        # Grid layout with 4 columns
        for i, (label, key, color) in enumerate(stat_items):
            row = i // 4
            col = i % 4
            card = tk.Frame(cards, bg=CARD, padx=20, pady=12)
            card.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
            tk.Label(card, textvariable=self.stats_vars[key], font=("Consolas", 24, "bold"),
                     fg=color, bg=CARD).pack()
            tk.Label(card, text=label, font=FONT_SMALL, fg=MUTED, bg=CARD).pack()
            
        for c in range(4):
            cards.grid_columnconfigure(c, weight=1)

        tk.Label(self.tab_stats, text="ALERT HISTORY", font=FONT_BODY,
                 fg=MUTED, bg=BG).pack(padx=15, anchor="w", pady=(10, 2))
        self.stats_log = tk.Text(self.tab_stats, bg=CARD, fg=TEXT, font=FONT_SMALL,
                                  relief="flat", padx=10, pady=5, highlightthickness=0)
        self.stats_log.pack(padx=10, pady=(0, 10), fill="both", expand=True)
        self.stats_log.config(state="disabled")

    def _refresh_stats(self):
        vid_ents = getattr(self, "vid_tracked_entities", {})
        cam_ents = getattr(self, "cam_tracked_entities", {})
        
        bags = sum(1 for e in vid_ents.values() if e["class_name"] != "person") + \
               sum(1 for e in cam_ents.values() if e["class_name"] != "person")
        persons = sum(1 for e in vid_ents.values() if e["class_name"] == "person") + \
                  sum(1 for e in cam_ents.values() if e["class_name"] == "person")
                  
        self.stats_vars["total_bags"].set(bags)
        self.stats_vars["total_persons"].set(persons)

    def _on_owner_returned(self, bag_id):
        self._stop_alert_sound()
        ts = time.strftime("%H:%M:%S")
        msg = f"[{ts}] ✓ BAG #{bag_id} — OWNER RETURNED, alert cancelled\n"
        for log in [getattr(self, 'vid_alert_log', None),
                    getattr(self, 'cam_alert_log', None)]:
            if log:
                self._write_log(log, msg)

    def _update_alert_stats(self, alert_type, alert_id, severity=""):
        # Increment behavior counts
        if "unattended_object" in alert_type:
            self.stats_vars["total_alerts"].set(self.stats_vars["total_alerts"].get() + 1)
        elif "loiter" in alert_type:
            self.stats_vars["total_loiters"].set(self.stats_vars["total_loiters"].get() + 1)
        elif "running" in alert_type:
            self.stats_vars["total_running"].set(self.stats_vars["total_running"].get() + 1)
        elif "pacing" in alert_type or "movement" in alert_type:
            self.stats_vars["total_pacing"].set(self.stats_vars["total_pacing"].get() + 1)
        elif "restricted" in alert_type or "zone" in alert_type:
            self.stats_vars["total_zone_entries"].set(self.stats_vars["total_zone_entries"].get() + 1)

        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {alert_type.upper()} on {alert_id} | SEVERITY: {severity}\n"
        self.alert_history.append(entry)
        if hasattr(self, 'stats_log'):
            self.stats_log.config(state="normal")
            self.stats_log.insert("end", entry)
            self.stats_log.see("end")
            self.stats_log.config(state="disabled")

    def _export_pdf(self):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            import glob

            save_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")],
                initialfile=f"sentinel_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            if not save_path:
                return

            doc = SimpleDocTemplate(save_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph("SENTINEL AI — Threat Detection Session Report", styles["Title"]))
            story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
            story.append(Spacer(1, 20))

            duration = int(time.time() - self.stats_vars["session_start"])
            summary_data = [
                ["Metric / Behavior", "Value"],
                ["Session Duration", f"{duration // 60}m {duration % 60}s"],
                ["Bags Tracked", str(self.stats_vars["total_bags"].get())],
                ["Persons Tracked", str(self.stats_vars["total_persons"].get())],
                ["Unattended Objects Fired", str(self.stats_vars["total_alerts"].get())],
                ["Loitering Incidents", str(self.stats_vars["total_loiters"].get())],
                ["Rapid Movement Detections", str(self.stats_vars["total_running"].get())],
                ["Repeated Pacing Incidents", str(self.stats_vars["total_pacing"].get())],
                ["Restricted Zone Crossings", str(self.stats_vars["total_zone_entries"].get())],
            ]
            t = Table(summary_data, colWidths=[250, 200])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.black),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            story.append(t)
            story.append(Spacer(1, 20))

            story.append(Paragraph("Suspicious Behavior Alert Log", styles["Heading2"]))
            for entry in self.alert_history:
                story.append(Paragraph(entry.strip(), styles["Normal"]))

            incidents_dir = os.path.join(BASE_DIR, "incidents")
            snapshots = glob.glob(os.path.join(incidents_dir, "*.jpg"))
            if snapshots:
                story.append(Spacer(1, 20))
                story.append(Paragraph(f"Incident Snapshots: {len(snapshots)} saved to incidents/ folder", styles["Normal"]))

            doc.build(story)
            messagebox.showinfo("Export Complete", f"Report saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = SentinelApp(root)
    root.mainloop()
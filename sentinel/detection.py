"""
SENTINEL AI — Object Detection & False Positive Filtering Module
Provides strict dual-model inference isolation:
- Luggage model emits exclusively 'luggage' detections.
- Person model emits exclusively 'person' detections (COCO class 0).
- Applies geometric containment, head-zone suppression, and diagnostic logging.
"""

import os
import sys
import math
import numpy as np
from sentinel.config import SentinelConfig

# ═══════════════════════════════════════════════════════════════════════════════
# OBJECT GROUP DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

OBJECT_GROUP_MAP = {
    "luggage": "BAG_OBJECT",
    "backpack": "BAG_OBJECT",
    "handbag": "BAG_OBJECT",
    "suitcase": "BAG_OBJECT",
    "person": "PERSON",
}

def get_object_group(class_name):
    """Map a model class name to its semantic object group."""
    return OBJECT_GROUP_MAP.get(str(class_name).lower(), "OTHER")

def is_bag_object(class_name):
    """Check if a class name belongs to the BAG_OBJECT group."""
    return get_object_group(class_name) == "BAG_OBJECT"

def is_person(class_name):
    """Check if a class name belongs to the PERSON group."""
    return get_object_group(class_name) == "PERSON"


# ═══════════════════════════════════════════════════════════════════════════════
# DUAL MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_dual_models(base_dir):
    """
    Load dual inference models:
    1. bag_model: fine-tuned weights (models/train/weights/best.pt or models/best.pt)
    2. person_model: yolov9s.pt from root (COCO pretrained)
    Returns (bag_model, person_model, status_string)
    """
    import torch
    from ultralytics import YOLO

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


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION PIPELINES
# ═══════════════════════════════════════════════════════════════════════════════

def detect_bags(frame, model, conf=None, device=None):
    """
    Run luggage model inference.
    Requirements:
    - Strictly accepts class 0 ('luggage') from the fine-tuned model.
    - Discards invalid frame-covering artifacts (> MAX_BAG_FRAME_AREA_RATIO).
    - Logs diagnostic information in development mode.
    Returns list of tuples: (x1, y1, x2, y2, confidence, class_name)
    """
    if model is None or frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return []

    if conf is None:
        conf = SentinelConfig.BAG_CONF_DEFAULT

    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    h, w = frame.shape[:2]
    frame_area = float(w * h)

    results = model(frame, conf=conf, device=device, verbose=False)
    detections = []

    for r in results:
        names = r.names
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            raw_conf = float(box.conf[0])
            cls_name = names.get(cls_id, "luggage")

            # Verify class is luggage
            if cls_name.lower() not in ("luggage", "bag", "backpack", "suitcase", "handbag"):
                if SentinelConfig.DEBUG_DETECTION:
                    print(f"[DIAGNOSTIC] REJECTED non-luggage class from bag model: id={cls_id}, name={cls_name}")
                continue

            # Standardize class name to 'luggage'
            cls_name = "luggage"
            bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
            bw = max(0, bx2 - bx1)
            bh = max(0, by2 - by1)
            box_area = float(bw * bh)

            # Filter whole-frame artifacts
            if frame_area > 0 and (box_area / frame_area) > SentinelConfig.MAX_BAG_FRAME_AREA_RATIO:
                if SentinelConfig.DEBUG_DETECTION:
                    print(f"[DIAGNOSTIC] REJECTED whole-scene artifact: area_ratio={box_area/frame_area:.2f} > {SentinelConfig.MAX_BAG_FRAME_AREA_RATIO}")
                continue

            det = (bx1, by1, bx2, by2, raw_conf, cls_name)
            detections.append(det)

            if SentinelConfig.DEBUG_DETECTION:
                print(f"[DIAGNOSTIC] MODEL=best.pt | CLASS_ID={cls_id} | CLASS_NAME={cls_name} | "
                      f"CONF={raw_conf:.2f} | BBOX=({bx1},{by1},{bx2},{by2})")

    return detections


def detect_persons(frame, model, conf=None, device=None):
    """
    Run person model inference.
    Requirements:
    - Exclusively detects humans (COCO class 0).
    - NEVER detects or outputs luggage/backpacks from this detector.
    - Logs diagnostic information in development mode.
    Returns list of tuples: (x1, y1, x2, y2, confidence, class_name)
    """
    if model is None or frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return []

    if conf is None:
        conf = SentinelConfig.PERSON_CONF_DEFAULT

    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Strictly query class 0 (person) ONLY
    results = model(frame, conf=conf, classes=[0], device=device, verbose=False)
    detections = []

    for r in results:
        names = r.names
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            raw_conf = float(box.conf[0])
            cls_name = names.get(cls_id, "person")

            if cls_id != 0 or cls_name.lower() != "person":
                continue

            px1, py1, px2, py2 = int(x1), int(y1), int(x2), int(y2)
            det = (px1, py1, px2, py2, raw_conf, "person")
            detections.append(det)

            if SentinelConfig.DEBUG_DETECTION:
                print(f"[DIAGNOSTIC] MODEL=yolov9s.pt | CLASS_ID={cls_id} | CLASS_NAME=person | "
                      f"CONF={raw_conf:.2f} | BBOX=({px1},{py1},{px2},{py2})")

    return detections


# ═══════════════════════════════════════════════════════════════════════════════
# FALSE POSITIVE SUPPRESSION & GEOMETRIC CONTAINMENT
# ═══════════════════════════════════════════════════════════════════════════════

def filter_bag_person_false_positives(bag_dets, person_dets):
    """
    Eliminates false luggage detections that are actually parts of a person's body.
    Checks:
    1. Bag containment ratio inside person bounding box: Area(Bag ∩ Person) / Area(Bag).
    2. Overlap with person's head/upper-torso zone (top PERSON_HEAD_ZONE_RATIO of person bbox).
    """
    if not bag_dets or not person_dets:
        return bag_dets

    clean_bags = []
    containment_thresh = SentinelConfig.PERSON_BAG_CONTAINMENT_THRESH
    head_ratio = SentinelConfig.PERSON_HEAD_ZONE_RATIO

    for b in bag_dets:
        bx1, by1, bx2, by2, bconf, bname = b[:6]
        b_area = max(1.0, float((bx2 - bx1) * (by2 - by1)))
        suppress = False
        suppress_reason = ""

        for p in person_dets:
            px1, py1, px2, py2 = p[:4]
            p_height = max(1, py2 - py1)

            # Intersection between bag and person
            ix1 = max(bx1, px1)
            iy1 = max(by1, py1)
            ix2 = min(bx2, px2)
            iy2 = min(by2, py2)

            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter_area = float(inter_w * inter_h)

            containment = inter_area / b_area

            # Head / upper torso zone
            head_zone_bottom = py1 + int(p_height * head_ratio)
            head_iy1 = max(by1, py1)
            head_iy2 = min(by2, head_zone_bottom)
            head_inter_h = max(0, head_iy2 - head_iy1)
            head_inter_area = float(inter_w * head_inter_h)
            head_overlap = head_inter_area / b_area

            # If bag is substantially inside person's head/chest area -> person body false positive
            if head_overlap >= 0.40:
                suppress = True
                suppress_reason = f"Head/Torso overlap ({head_overlap:.2f} >= 0.40)"
                break

            # If bag is heavily contained inside person bbox (> 70%) -> body false positive
            if containment >= containment_thresh:
                suppress = True
                suppress_reason = f"Body containment ({containment:.2f} >= {containment_thresh})"
                break

        if suppress:
            if SentinelConfig.DEBUG_DETECTION:
                print(f"[DIAGNOSTIC] SUPPRESSED false luggage detection: BBOX=({bx1},{by1},{bx2},{by2}), "
                      f"CONF={bconf:.2f}, REASON={suppress_reason}")
        else:
            clean_bags.append(b)

    return clean_bags


def calculate_iou(box1, box2):
    """Calculate standard Intersection-over-Union (IoU) of two boxes."""
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
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def iou_deduplicate(detections, iou_thresh=0.5, cross_class_iou_thresh=0.3):
    """
    Remove duplicate boxes belonging to similar semantic classes.
    Uses object_group normalization for cross-model consistency.
    """
    if len(detections) <= 1:
        return detections

    sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
    kept = []

    for det in sorted_dets:
        discard = False
        det_group = get_object_group(det[5])
        for k in kept:
            k_group = get_object_group(k[5])

            # Deduplicate boxes within the same semantic group
            if det_group == k_group:
                if calculate_iou(det, k) > iou_thresh:
                    discard = True
                    break

        if not discard:
            kept.append(det)

    return kept

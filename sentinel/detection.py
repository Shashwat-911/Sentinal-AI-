"""
SENTINEL AI - Person-only detection utilities.

The runtime detector is the COCO YOLO person model only. It requests class 0
from YOLO and discards every non-person class before tracking or display.
"""

import os


from sentinel.config import SentinelConfig


OBJECT_GROUP_MAP = {
    "person": "PERSON",
}


def get_object_group(class_name):
    """Map a model class name to its semantic object group."""
    return OBJECT_GROUP_MAP.get(str(class_name).lower(), "OTHER")


def is_person(class_name):
    """Check if a class name belongs to the PERSON group."""
    return get_object_group(class_name) == "PERSON"


def load_person_model(base_dir):
    """
    Load the YOLO person detector.
    Returns (person_model, status_string). The detector is always queried with
    classes=[0], so non-person classes are never emitted to the tracker.
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
    person_path = os.path.join(base_dir, "yolov9s.pt")

    if not os.path.isfile(person_path):
        return None, f"Person model not found at {person_path}"

    try:
        person_model = YOLO(person_path)
    except Exception as e:
        return None, f"Person model error: {e}"

    return person_model, f"Person model loaded ({device.upper()}: {dev_name})"


def detect_persons(frame, model, conf=None, device=None):
    """
    Run person-only inference.
    Returns list of tuples: (x1, y1, x2, y2, confidence, "person").
    """
    if model is None or frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return []

    if conf is None:
        conf = SentinelConfig.PERSON_CONF_DEFAULT

    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = model(frame, conf=conf, classes=[0], device=device, verbose=False)
    detections = []

    for r in results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, "person")
            if cls_id != 0 or str(cls_name).lower() != "person":
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            raw_conf = float(box.conf[0])
            px1, py1, px2, py2 = int(x1), int(y1), int(x2), int(y2)
            detections.append((px1, py1, px2, py2, raw_conf, "person"))

            if SentinelConfig.DEBUG_DETECTION:
                print(
                    f"[DIAGNOSTIC] MODEL=yolov9s.pt | CLASS_ID=0 | CLASS_NAME=person | "
                    f"CONF={raw_conf:.2f} | BBOX=({px1},{py1},{px2},{py2})"
                )

    return detections


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
    """Remove duplicate person boxes."""
    if len(detections) <= 1:
        return detections

    sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
    kept = []
    for det in sorted_dets:
        if not is_person(det[5]):
            continue
        if any(calculate_iou(det, existing) > iou_thresh for existing in kept):
            continue
        kept.append(det)
    return kept

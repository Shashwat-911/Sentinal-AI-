"""
SENTINEL AI — Multi-Object Tracker Module
Maintains isolated, class-aware multi-object tracking for persons and luggage items.
Enforces:
- Strict isolation: A person detection NEVER becomes a luggage track, and vice-versa.
- Persistent tracking IDs (PERSON-1, PERSON-2, LUGGAGE-1, LUGGAGE-2).
- EMA bounding box smoothing and image-plane velocity tracking.
- Bounded trajectory histories with grid-cell visit tracking for behavioral analysis.
"""

import time
import math
from sentinel.config import SentinelConfig
from sentinel.detection import get_object_group, is_person, is_bag_object


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
    Supports both dictionary indexing and attribute access.
    """
    def __init__(self, track_id, class_name, bbox, confidence, centroid=None, **kwargs):
        super().__init__()
        self["track_id"] = int(track_id)
        self["class_name"] = str(class_name)
        self["object_type"] = "PERSON" if is_person(class_name) else "LUGGAGE"
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
        max_hist = getattr(SentinelConfig, "MAX_TRACK_HISTORY", 90)
        self["history"] = list(init_history)[-max_hist:]

        # Image-plane movement velocity
        self["velocity"] = float(kwargs.get("velocity", 0.0))
        self["smoothed_velocity"] = float(kwargs.get("smoothed_velocity", 0.0))
        self["speed"] = self["smoothed_velocity"]

        # Behavioral and threat properties
        self["active_threats"] = set(kwargs.get("active_threats", set()))
        self["alert_dismissed"] = bool(kwargs.get("alert_dismissed", False))
        self["last_alert_time"] = float(kwargs.get("last_alert_time", 0.0))
        self["last_alert_times"] = dict(kwargs.get("last_alert_times", {}))
        self["risk_score"] = int(kwargs.get("risk_score", 0))
        self["severity"] = str(kwargs.get("severity", "LOW"))
        self["stationary_start"] = kwargs.get("stationary_start", None)
        self["last_move_time"] = float(kwargs.get("last_move_time", now))
        self["nearest_person_id"] = kwargs.get("nearest_person_id", None)
        self["last_proximity_time"] = float(kwargs.get("last_proximity_time", now))
        self["direction_reversals"] = list(kwargs.get("direction_reversals", []))
        self["consecutive_running_frames"] = int(kwargs.get("consecutive_running_frames", 0))
        self["consecutive_restricted_frames"] = int(kwargs.get("consecutive_restricted_frames", 0))

        # Phase 4 & Suspicious Behavior Analysis Properties
        self["meander_ratio"] = float(kwargs.get("meander_ratio", 1.0))
        self["revisit_count"] = int(kwargs.get("revisit_count", 0))
        self["visited_cells"] = list(kwargs.get("visited_cells", []))
        self["last_grid_cell"] = kwargs.get("last_grid_cell", None)
        self["owner_returned"] = bool(kwargs.get("owner_returned", False))
        self["owner_return_time"] = kwargs.get("owner_return_time", None)
        self["suspicion_score"] = int(kwargs.get("suspicion_score", 0))
        self["suspicion_level"] = str(kwargs.get("suspicion_level", "NORMAL"))
        self["behavior_state"] = str(kwargs.get("behavior_state", "NORMAL"))
        self["suspicion_evidence"] = dict(kwargs.get("suspicion_evidence", {}))
        self["sustained_suspicious_frames"] = int(kwargs.get("sustained_suspicious_frames", 0))
        self["cooldown_frames"] = int(kwargs.get("cooldown_frames", 0))
        self["path_length"] = float(kwargs.get("path_length", 0.0))
        self["net_displacement"] = float(kwargs.get("net_displacement", 0.0))
        self["in_restricted_zone"] = bool(kwargs.get("in_restricted_zone", False))

        for k, v in kwargs.items():
            if k not in self:
                self[k] = v

    @property
    def track_id(self): return self["track_id"]
    @property
    def class_name(self): return self["class_name"]
    @property
    def object_type(self): return self["object_type"]
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


def match_entities(detections, tracked_entities, threshold=None, max_missing_sec=None, current_time=None):
    """
    Class-aware multi-object centroid tracker.
    Guarantees:
    - Persons are strictly matched against person tracks.
    - Luggage items are strictly matched against luggage tracks.
    - Person tracks and Luggage tracks operate in independent namespaces.
    - Supports explicit current_time for deterministic video-time evaluation.
    """
    if threshold is None:
        threshold = getattr(SentinelConfig, "TRACK_MATCH_THRESHOLD", 100.0)
    if max_missing_sec is None:
        max_missing_sec = getattr(SentinelConfig, "TRACK_TIMEOUT", 3.0)
    max_missed_frames = getattr(SentinelConfig, "MAX_MISSED_FRAMES", 30)
    alpha = getattr(SentinelConfig, "BBOX_SMOOTHING_ALPHA", 0.75)
    beta = getattr(SentinelConfig, "VELOCITY_SMOOTHING_ALPHA", 0.30)
    max_history = getattr(SentinelConfig, "MAX_TRACK_HISTORY", 600)

    if current_time is None:
        current_time = time.time()
    else:
        current_time = float(current_time)
    new_tracked = {}
    matched_keys = set()
    matched_det_indices = set()

    # Organize detections strictly by semantic group (PERSON or BAG_OBJECT)
    dets_by_group = {}
    for idx, det in enumerate(detections):
        cls_name = det[5] if len(det) > 5 else "luggage"
        group = get_object_group(cls_name)
        dets_by_group.setdefault(group, []).append((idx, det))

    # Organize existing tracks strictly by semantic group
    tracks_by_group = {}
    for key, info in tracked_entities.items():
        cls_name = info.get("class_name", "luggage")
        group = get_object_group(cls_name)
        tracks_by_group.setdefault(group, []).append((key, info))

    all_groups = set(dets_by_group.keys()) | set(tracks_by_group.keys())

    for group in all_groups:
        group_dets = dets_by_group.get(group, [])
        group_tracks = tracks_by_group.get(group, [])

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

        # Competitive closest-first assignment
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

            # Image-plane velocity
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
                # Retain behavior fields
                active_threats=t_info.get("active_threats", set()),
                alert_dismissed=t_info.get("alert_dismissed", False),
                last_alert_time=t_info.get("last_alert_time", 0.0),
                last_alert_times=t_info.get("last_alert_times", {}),
                risk_score=t_info.get("risk_score", 0),
                severity=t_info.get("severity", "LOW"),
                stationary_start=t_info.get("stationary_start", None),
                last_move_time=t_info.get("last_move_time", current_time),
                nearest_person_id=t_info.get("nearest_person_id", None),
                last_proximity_time=t_info.get("last_proximity_time", current_time),
                direction_reversals=t_info.get("direction_reversals", []),
                consecutive_running_frames=t_info.get("consecutive_running_frames", 0),
                consecutive_restricted_frames=t_info.get("consecutive_restricted_frames", 0),
                meander_ratio=t_info.get("meander_ratio", 1.0),
                revisit_count=t_info.get("revisit_count", 0),
                visited_cells=t_info.get("visited_cells", []),
                last_grid_cell=t_info.get("last_grid_cell", None),
                owner_returned=t_info.get("owner_returned", False),
                owner_return_time=t_info.get("owner_return_time", None),
                suspicion_score=t_info.get("suspicion_score", 0),
                suspicion_level=t_info.get("suspicion_level", "NORMAL"),
                behavior_state=t_info.get("behavior_state", "NORMAL"),
                suspicion_evidence=t_info.get("suspicion_evidence", {}),
                sustained_suspicious_frames=t_info.get("sustained_suspicious_frames", 0),
                cooldown_frames=t_info.get("cooldown_frames", 0),
                path_length=t_info.get("path_length", 0.0),
                net_displacement=t_info.get("net_displacement", 0.0),
                in_restricted_zone=t_info.get("in_restricted_zone", False)
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
                last_alert_times=info.get("last_alert_times", {}),
                risk_score=info.get("risk_score", 0),
                severity=info.get("severity", "LOW"),
                stationary_start=info.get("stationary_start", None),
                last_move_time=info.get("last_move_time", current_time),
                nearest_person_id=info.get("nearest_person_id", None),
                last_proximity_time=info.get("last_proximity_time", current_time),
                direction_reversals=info.get("direction_reversals", []),
                consecutive_running_frames=info.get("consecutive_running_frames", 0),
                consecutive_restricted_frames=info.get("consecutive_restricted_frames", 0),
                meander_ratio=info.get("meander_ratio", 1.0),
                revisit_count=info.get("revisit_count", 0),
                visited_cells=info.get("visited_cells", []),
                last_grid_cell=info.get("last_grid_cell", None),
                owner_returned=info.get("owner_returned", False),
                owner_return_time=info.get("owner_return_time", None),
                suspicion_score=info.get("suspicion_score", 0),
                suspicion_level=info.get("suspicion_level", "NORMAL"),
                behavior_state=info.get("behavior_state", "NORMAL"),
                suspicion_evidence=info.get("suspicion_evidence", {}),
                sustained_suspicious_frames=info.get("sustained_suspicious_frames", 0),
                cooldown_frames=info.get("cooldown_frames", 0),
                path_length=info.get("path_length", 0.0),
                net_displacement=info.get("net_displacement", 0.0),
                in_restricted_zone=info.get("in_restricted_zone", False)
            )
            new_tracked[key] = kept_track

    return new_tracked


# Backward compatibility wrappers
def match_bag_centroids(detections, tracked_bags, threshold=80, max_missing_sec=3.0):
    formatted_dets = []
    for d in detections:
        if len(d) == 5:
            formatted_dets.append((d[0], d[1], d[2], d[3], d[4], "luggage"))
        else:
            formatted_dets.append(d)

    unified = {}
    for bid, info in tracked_bags.items():
        key = f"BAG_OBJECT_{bid}"
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
    formatted_dets = []
    for d in detections:
        if len(d) == 5:
            formatted_dets.append((d[0], d[1], d[2], d[3], d[4], "person"))
        else:
            formatted_dets.append(d)

    unified = {}
    for pid, info in tracked_persons.items():
        key = f"PERSON_{pid}"
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

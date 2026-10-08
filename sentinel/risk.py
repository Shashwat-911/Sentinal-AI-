"""
SENTINEL AI — Risk Analysis & Owner Proximity Association Module
Preserves:
- 4-Tier unattended baggage risk escalation: LOW -> MEDIUM -> HIGH -> CRITICAL
- Proximity-based owner-return mechanism (Rowner = 150px)
- Immediate cancellation and timer reset when owner returns.
"""

import math
import time
from sentinel.config import SentinelConfig
from sentinel.detection import is_person, is_bag_object
from sentinel.behavior import evaluate_person_behavior


def calculate_entity_risk(track, current_time=None):
    """
    Calculate risk score (0-100), severity label, and HUD display color (BGR).
    Distinct paths for PERSON and LUGGAGE entities.
    Supports explicit current_time for deterministic video-time evaluation.
    """
    threats = track.get("active_threats", set())
    if not threats or track.get("alert_dismissed", False):
        return 0, "NORMAL", (180, 180, 0)

    max_score = 0
    if current_time is None:
        now = track.get("last_seen", time.time())
    else:
        now = float(current_time)

    # Luggage Risk Model: Strictly proportional to unattended dwell duration
    if "UNATTENDED_OBJECT" in threats:
        stationary_start = track.get("stationary_start") or now
        elapsed = max(0.0, float(now - stationary_start))
        ratio = max(0.0, min(1.0, elapsed / max(1.0, SentinelConfig.STATIONARY_THRESHOLD)))
        obj_score = int(ratio * 100)
        max_score = max(max_score, obj_score)

    # Person Behavior Threats (Separated, multi-signal explainable scoring)
    if is_person(track.get("class_name", "")):
        pers_score = track.get("suspicion_score", 0)
        max_score = max(max_score, pers_score)

        if "HIGH_CONFIDENCE_SUSPICIOUS" in threats or track.get("behavior_state") == "CRITICAL":
            max_score = max(max_score, 100)  # CRITICAL
        elif "SUSPICIOUS_LOITERING" in threats or track.get("behavior_state") == "ALERT":
            max_score = max(max_score, 75)   # HIGH
        elif "LOITERING" in threats:
            max_score = max(max_score, 50)   # Backward-compatible baseline
        elif "RESTRICTED_ZONE" in threats:
            max_score = max(max_score, 70)   # HIGH
        elif "SUDDEN_RUNNING" in threats:
            max_score = max(max_score, 65)   # HIGH
        elif "REPEATED_MOVEMENT" in threats:
            max_score = max(max_score, 45)   # MEDIUM
        elif "LONG_DWELL" in threats:
            max_score = max(max_score, 25)   # LOW
    else:
        # Fallback for non-person tracks
        if "LOITERING" in threats:
            max_score = max(max_score, 50)
        if "REPEATED_MOVEMENT" in threats:
            max_score = max(max_score, 45)
        if "SUDDEN_RUNNING" in threats:
            max_score = max(max_score, 65)
        if "RESTRICTED_ZONE" in threats:
            max_score = max(max_score, 70)

    # 4-Tier Risk Escalation
    if max_score < 34:
        return max_score, "LOW", (0, 200, 100)       # Green
    elif max_score < 67:
        return max_score, "MEDIUM", (0, 165, 255)    # Orange
    elif max_score < 100:
        return max_score, "HIGH", (0, 60, 255)       # Red-Orange
    else:
        return 100, "CRITICAL", (0, 0, 255)          # Pure Red


def check_owner_proximity_and_returns(tracked_entities, current_time=None):
    """
    Evaluates Euclidean distance between each bag and all tracked persons.
    Implements Rowner = 150px proximity return disarming.
    """
    if current_time is None:
        current_time = time.time()

    persons = {k: v for k, v in tracked_entities.items() if is_person(v["class_name"])}
    bags = {k: v for k, v in tracked_entities.items() if is_bag_object(v["class_name"])}

    owner_radius = SentinelConfig.OWNER_RADIUS_PIXELS

    for b_key, b_info in bags.items():
        bcx, bcy = b_info["centroid"]
        min_dist = float("inf")
        nearest_p_key = None

        for p_key, p_info in persons.items():
            pcx, pcy = p_info["centroid"]
            dist = math.hypot(bcx - pcx, bcy - pcy)
            if dist < min_dist:
                min_dist = dist
                nearest_p_key = p_key

        b_info["nearest_person_id"] = nearest_p_key

        # If a person is within owner radius Rowner
        if min_dist <= owner_radius:
            b_info["last_proximity_time"] = current_time

            # If the bag was previously unattended, cancel the alert and mark owner returned
            if "UNATTENDED_OBJECT" in b_info["active_threats"] or b_info.get("stationary_start") is not None:
                b_info["active_threats"].discard("UNATTENDED_OBJECT")
                b_info["stationary_start"] = None
                b_info["owner_returned"] = True
                b_info["owner_return_time"] = current_time
        else:
            # Person has left owner radius: initiate or continue unattended timer
            if b_info.get("stationary_start") is None:
                b_info["stationary_start"] = current_time
                b_info["owner_returned"] = False

            elapsed_unattended = current_time - b_info["stationary_start"]
            # Flag unattended threat as duration progresses
            if elapsed_unattended >= (SentinelConfig.STATIONARY_THRESHOLD * 0.33):
                b_info["active_threats"].add("UNATTENDED_OBJECT")


def analyze_behaviors(tracked_entities, w, h, current_time=None):
    """
    Central behavior analysis pipeline.
    Coordinates:
    1. Person trajectory and behavioral loitering analysis
    2. Proximity owner association and unattended luggage scoring
    """
    if current_time is None:
        current_time = time.time()

    # 1. Evaluate behavior for all currently-visible persons.
    # Occluded tracks keep prior behavioural state (do not reset, do not keep accumulating).
    for key, info in tracked_entities.items():
        if not is_person(info["class_name"]):
            continue
        if info.get("missed_frames", 0) > 0 or info.get("state") == "TEMPORARILY_MISSED":
            continue
        evaluate_person_behavior(info, w, h, current_time)

    # 2. Evaluate owner proximity and unattended luggage
    check_owner_proximity_and_returns(tracked_entities, current_time)

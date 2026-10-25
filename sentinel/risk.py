"""
SENTINEL AI - Person behavior risk analysis.
"""

import time
from sentinel.detection import is_person
from sentinel.behavior import evaluate_person_behavior


def calculate_entity_risk(track, current_time=None):
    """
    Calculate risk score (0-100), severity label, and HUD display color (BGR).
    Person-only risk scoring for dwell, movement, loitering, running, and safety events.
    Supports explicit current_time for deterministic video-time evaluation.
    """
    threats = track.get("active_threats", set())
    if not threats or track.get("alert_dismissed", False):
        return 0, "NORMAL", (180, 180, 0)

    max_score = 0
    if is_person(track.get("class_name", "")):
        pers_score = track.get("suspicion_score", 0)
        max_score = max(max_score, pers_score)

        if "HIGH_CONFIDENCE_SUSPICIOUS" in threats or track.get("behavior_state") == "CRITICAL":
            max_score = max(max_score, 100)  # CRITICAL
        elif "SUSPICIOUS_LOITERING" in threats or track.get("behavior_state") == "ALERT":
            max_score = max(max_score, 75)   # HIGH
        elif "LOITERING" in threats:
            max_score = max(max_score, 50)   # Backward-compatible baseline
        elif "SUDDEN_RUNNING" in threats:
            max_score = max(max_score, 65)   # HIGH
        elif "REPEATED_MOVEMENT" in threats:
            max_score = max(max_score, 45)   # MEDIUM
        elif "LONG_DWELL" in threats:
            max_score = max(max_score, 25)   # LOW
        elif "POSSIBLE_FALL" in threats:
            max_score = max(max_score, 50)   # MEDIUM safety event

    # 4-Tier Risk Escalation
    if max_score < 34:
        return max_score, "LOW", (0, 200, 100)       # Green
    elif max_score < 67:
        return max_score, "MEDIUM", (0, 165, 255)    # Orange
    elif max_score < 100:
        return max_score, "HIGH", (0, 60, 255)       # Red-Orange
    else:
        return 100, "CRITICAL", (0, 0, 255)          # Pure Red


def analyze_behaviors(tracked_entities, w, h, current_time=None):
    """
    Central behavior analysis pipeline.
    Coordinates person trajectory and behavioral loitering analysis.
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

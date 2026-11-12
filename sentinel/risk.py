"""
SENTINEL AI - Person behavior risk analysis.
"""

import time
from sentinel.detection import is_person
from sentinel.behavior import evaluate_person_behavior


def _score_to_tier(score):
    """Map numeric 0-100 risk score to (score, severity_label, BGR_color)."""
    if score < 34:
        return score, "LOW", (0, 200, 100)       # Green
    elif score < 67:
        return score, "MEDIUM", (0, 165, 255)    # Orange
    elif score < 100:
        return score, "HIGH", (0, 60, 255)       # Red-Orange
    else:
        return 100, "CRITICAL", (0, 0, 255)      # Pure Red


def calculate_entity_risk(track, current_time=None, ai_context=None):
    """
    Calculate risk score (0-100), severity label, and HUD display color (BGR).
    Person-only risk scoring for dwell, movement, loitering, running, and safety events.
    When ai_context is provided, applies AI-weighted fusion (80% AI / 20% Rules) so
    both HUD risk and context analysis align (never suppressing POSSIBLE_FALL).
    """
    if track.get("alert_dismissed", False):
        return 0, "NORMAL", (180, 180, 0)

    threats = track.get("active_threats", set())
    rule_score = 0
    if threats and is_person(track.get("class_name", "")):
        pers_score = track.get("suspicion_score", 0)
        rule_score = max(rule_score, pers_score)

        if "HIGH_CONFIDENCE_SUSPICIOUS" in threats or track.get("behavior_state") == "CRITICAL":
            rule_score = max(rule_score, 100)  # CRITICAL
        elif "SUSPICIOUS_LOITERING" in threats or track.get("behavior_state") == "ALERT":
            rule_score = max(rule_score, 75)   # HIGH
        elif "LOITERING" in threats:
            rule_score = max(rule_score, 50)   # Backward-compatible baseline
        elif "SUDDEN_RUNNING" in threats:
            rule_score = max(rule_score, 65)   # HIGH
        elif "REPEATED_MOVEMENT" in threats:
            rule_score = max(rule_score, 45)   # MEDIUM
        elif "LONG_DWELL" in threats:
            rule_score = max(rule_score, 25)   # LOW
        elif "POSSIBLE_FALL" in threats:
            rule_score = max(rule_score, 50)   # MEDIUM safety event

    ai_cls = (ai_context or {}).get("classification")
    if ai_cls and "POSSIBLE_FALL" not in threats and is_person(track.get("class_name", "")):
        if ai_cls == "not_suspicious":
            weighted_score = round(0.20 * rule_score)
            if weighted_score < 25:
                return weighted_score, "NORMAL", (0, 255, 136)
            return weighted_score, "LOW", (0, 200, 100)
        elif ai_cls == "suspicious":
            ai_risk = 100 if rule_score >= 75 else 85
            weighted_score = min(100, round(0.80 * ai_risk + 0.20 * rule_score))
            return _score_to_tier(weighted_score)
        elif ai_cls == "safety_concern":
            return 100, "CRITICAL", (0, 0, 255)

    if not threats:
        return 0, "NORMAL", (180, 180, 0)

    return _score_to_tier(rule_score)


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

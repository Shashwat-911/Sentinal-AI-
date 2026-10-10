"""
SENTINEL AI - Person-only behavior and tracking tests.
"""

import os
import sys
import time
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from sentinel.detection import get_object_group, is_person, iou_deduplicate
from sentinel.tracking import TrackState, match_entities
from sentinel.behavior import (
    compute_dwell_time, compute_trajectory_metrics,
    update_grid_revisits, evaluate_person_behavior
)
from sentinel.risk import calculate_entity_risk, analyze_behaviors


def test_1_person_only_semantics():
    print("\n--- Test 1: Person-only semantic mapping ---")
    assert get_object_group("person") == "PERSON"
    assert get_object_group("suitcase") == "OTHER"
    assert is_person("person") is True
    assert is_person("suitcase") is False

    t_person = TrackState(track_id=1, class_name="person", bbox=(0, 0, 10, 10), confidence=0.9)
    assert t_person.object_type == "PERSON"
    print("  PASS: Only person class maps to a tracked object type.")


def test_2_non_person_detections_ignored():
    print("\n--- Test 2: Non-person detections are ignored by tracker ---")
    tracked = match_entities([(10, 10, 50, 50, 0.9, "suitcase")], {})
    assert len(tracked) == 0
    print("  PASS: Non-person detection did not create a track.")


def test_3_person_tracking_persistence():
    print("\n--- Test 3: Person tracking persistence ---")
    tracked = {}
    for f in range(15):
        dets = [(100 + f * 2, 100, 150 + f * 2, 250, 0.88, "person")]
        tracked = match_entities(dets, tracked)
        assert len(tracked) == 1
        key = list(tracked.keys())[0]
        assert tracked[key]["track_id"] == 1
    print("  PASS: PERSON #1 remained persistent across 15 continuous frames.")


def test_4_dwell_time_calculation():
    print("\n--- Test 4: Dwell-time calculation ---")
    now = time.time()
    t = TrackState(track_id=1, class_name="person", bbox=(0, 0, 10, 10), confidence=0.9, first_seen=now - 50.0)
    dwell = compute_dwell_time(t, current_time=now)
    assert math.isclose(dwell, 50.0, abs_tol=0.1)
    print(f"  PASS: Dwell time accurately calculated: {dwell:.1f}s.")


def test_5_meander_ratio_calculation():
    print("\n--- Test 5: Meander-ratio calculation ---")
    direct = [(0, 0, 0.0), (100, 0, 1.0), (200, 0, 2.0)]
    _, _, direct_ratio = compute_trajectory_metrics(direct)
    assert 0.9 <= direct_ratio <= 1.1

    wander = [(0, 0, 0.0), (100, 0, 1.0), (0, 0, 2.0), (100, 0, 3.0), (0, 0, 4.0)]
    _, _, wander_ratio = compute_trajectory_metrics(wander)
    assert wander_ratio > 2.5
    print(f"  PASS: Meander ratio direct={direct_ratio:.2f}, wandering={wander_ratio:.2f}.")


def test_6_revisit_count_calculation():
    print("\n--- Test 6: Grid revisit-count calculation ---")
    now = time.time()
    t = TrackState(track_id=1, class_name="person", bbox=(0, 0, 10, 10), confidence=0.9, current_time=now)
    update_grid_revisits(t, 50, 50, 800, 600)
    update_grid_revisits(t, 300, 50, 800, 600)
    revisits = update_grid_revisits(t, 50, 50, 800, 600)
    assert revisits == 1
    print("  PASS: Revisit count accurately tracks cell departure and return.")


def test_7_suspicious_loitering_decision():
    print("\n--- Test 7: Suspicious-loitering decision ---")
    now = time.time()
    p = TrackState(track_id=1, class_name="person", bbox=(0, 0, 50, 150), confidence=0.9, first_seen=now - 60)
    p["history"] = []
    for i in range(80):
        x = 100 if i % 2 == 0 else 250
        p["history"].append((x, 300, now - 60 + i * 0.75))
    p["centroid"] = p["history"][-1][:2]

    for _ in range(20):
        evaluate_person_behavior(p, 800, 600, current_time=now)
    assert "SUSPICIOUS_LOITERING" in p["active_threats"]
    assert p["behavior_state"] in ("ALERT", "CRITICAL")
    print("  PASS: Dwell plus repeated movement produces suspicious loitering.")


def test_8_posture_fall_safety_event_separation():
    print("\n--- Test 8: Posture fall safety event separation ---")
    now = time.time()
    p = TrackState(track_id=1, class_name="person", bbox=(100, 100, 200, 250), confidence=0.9, first_seen=now - 60)
    p["posture"] = "LYING_DOWN"
    p["pose_confidence"] = 0.88
    p["active_threats"].add("POSSIBLE_FALL")
    analyze_behaviors({"PERSON_1": p}, 800, 600, current_time=now)
    score, severity, _ = calculate_entity_risk(p, current_time=now)
    assert "SUSPICIOUS_LOITERING" not in p["active_threats"]
    assert severity in ("LOW", "MEDIUM")
    assert score >= 25
    print("  PASS: Possible fall remains a safety/person-posture event.")


def test_9_multiple_people_tracking():
    print("\n--- Test 9: Multiple people tracking ---")
    tracked = {}
    dets = [
        (50, 100, 100, 250, 0.9, "person"),
        (300, 100, 350, 250, 0.85, "person"),
        (550, 100, 600, 250, 0.88, "person"),
    ]
    tracked = match_entities(dets, tracked)
    ids = sorted(t["track_id"] for t in tracked.values())
    assert ids == [1, 2, 3]
    print("  PASS: Multiple people tracked with stable, distinct IDs [1, 2, 3].")


def test_10_iou_deduplicate_persons_only():
    print("\n--- Test 10: IoU deduplication keeps persons only ---")
    dets = [
        (10, 10, 100, 200, 0.9, "person"),
        (12, 12, 102, 202, 0.7, "person"),
        (250, 20, 320, 180, 0.8, "person"),
        (0, 0, 50, 50, 0.99, "suitcase"),
    ]
    deduped = iou_deduplicate(dets)
    assert len(deduped) == 2
    assert all(d[5] == "person" for d in deduped)
    print("  PASS: Duplicate people collapsed and non-person detections dropped.")


def run_all_tests():
    print("=" * 70)
    print("SENTINEL AI - PERSON-ONLY BEHAVIOR & TRACKING TESTS")
    print("=" * 70)
    test_1_person_only_semantics()
    test_2_non_person_detections_ignored()
    test_3_person_tracking_persistence()
    test_4_dwell_time_calculation()
    test_5_meander_ratio_calculation()
    test_6_revisit_count_calculation()
    test_7_suspicious_loitering_decision()
    test_8_posture_fall_safety_event_separation()
    test_9_multiple_people_tracking()
    test_10_iou_deduplicate_persons_only()
    print("\n" + "=" * 70)
    print("ALL PERSON-ONLY BEHAVIOR & TRACKING TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

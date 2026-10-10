"""
SENTINEL AI - Person tracker validation suite.
"""

import time
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app import match_entities, TrackStatus, SentinelConfig, get_object_group


def test_1_single_detection_multiple_frames():
    print("\n--- Running Test 1: Single person persistence across frames ---")
    tracked = {}
    dets_f1 = [(100, 100, 150, 200, 0.90, "person")]
    tracked = match_entities(dets_f1, tracked)
    assert len(tracked) == 1
    track_key = list(tracked.keys())[0]
    initial_id = tracked[track_key]["track_id"]
    assert tracked[track_key]["state"] == TrackStatus.NEW

    for f in range(2, 11):
        x_offset = f * 2
        dets_f = [(100 + x_offset, 100, 150 + x_offset, 200, 0.92, "person")]
        tracked = match_entities(dets_f, tracked)
        assert len(tracked) == 1
        assert tracked[track_key]["track_id"] == initial_id
        assert tracked[track_key]["age"] == f
        assert tracked[track_key]["detection_count"] == f
        assert tracked[track_key]["missed_frames"] == 0
        assert tracked[track_key]["state"] == TrackStatus.ACTIVE

    print(f"  PASS: PERSON #{initial_id} remained persistent over 10 consecutive frames.")


def test_2_two_independent_people():
    print("\n--- Running Test 2: Two independent moving people ---")
    tracked = {}
    for frame in range(1, 15):
        dets = [
            (50 + frame * 3, 100, 90 + frame * 3, 220, 0.88, "person"),
            (400 - frame * 3, 100, 440 - frame * 3, 220, 0.85, "person")
        ]
        tracked = match_entities(dets, tracked)
        assert len(tracked) == 2

    ids = sorted(t["track_id"] for t in tracked.values())
    assert ids == [1, 2]
    print(f"  PASS: Two people tracked with stable IDs {ids}.")


def test_3_temporary_missed_detection():
    print("\n--- Running Test 3: Temporary person detection loss and recovery ---")
    tracked = {}
    dets = [(200, 200, 260, 320, 0.90, "person")]
    tracked = match_entities(dets, tracked)
    key = list(tracked.keys())[0]
    original_id = tracked[key]["track_id"]

    for f in range(1, 11):
        tracked = match_entities([], tracked)
        assert key in tracked
        assert tracked[key]["state"] == TrackStatus.TEMPORARILY_MISSED
        assert tracked[key]["active"] is False
        assert tracked[key]["missed_frames"] == f

    tracked = match_entities([(205, 200, 265, 320, 0.88, "person")], tracked)
    assert key in tracked
    assert tracked[key]["track_id"] == original_id
    assert tracked[key]["missed_frames"] == 0
    assert tracked[key]["active"] is True
    assert tracked[key]["state"] == TrackStatus.ACTIVE
    print(f"  PASS: PERSON #{original_id} survived occlusion and restored identity.")


def test_4_detection_missing_beyond_limit_evicted():
    print("\n--- Running Test 4: Track eviction beyond MAX_MISSED_FRAMES ---")
    tracked = match_entities([(100, 100, 150, 200, 0.85, "person")], {})
    max_miss = SentinelConfig.MAX_MISSED_FRAMES
    for _ in range(max_miss):
        tracked = match_entities([], tracked)
        assert len(tracked) == 1

    tracked = match_entities([], tracked)
    assert len(tracked) == 0
    print(f"  PASS: Track evicted after exceeding {max_miss} missed frames.")


def test_5_non_person_detections_do_not_track():
    print("\n--- Running Test 5: Non-person detections do not create tracks ---")
    tracked = match_entities([
        (100, 100, 160, 240, 0.90, "person"),
        (110, 160, 150, 210, 0.85, "suitcase"),
    ], {})
    assert len(tracked) == 1
    only_track = list(tracked.values())[0]
    assert only_track["class_name"] == "person"
    assert get_object_group("suitcase") == "OTHER"
    print("  PASS: Person track created and non-person detection ignored.")


def test_6_trajectory_history_bounded():
    print("\n--- Running Test 6: Bounded trajectory history ---")
    tracked = {}
    for f in range(100):
        tracked = match_entities([(100 + f, 100, 150 + f, 200, 0.90, "person")], tracked)

    track = list(tracked.values())[0]
    hist_len = len(track["history"])
    assert hist_len <= SentinelConfig.MAX_TRACK_HISTORY
    print(f"  PASS: History length ({hist_len}) is within cap ({SentinelConfig.MAX_TRACK_HISTORY}).")


def test_7_stationary_person_zero_velocity():
    print("\n--- Running Test 7: Stationary person velocity ---")
    tracked = {}
    for _ in range(10):
        time.sleep(0.01)
        tracked = match_entities([(300, 300, 350, 450, 0.95, "person")], tracked)

    track = list(tracked.values())[0]
    vel = track["smoothed_velocity"]
    assert vel < 1.0
    print(f"  PASS: Stationary person velocity = {vel:.4f} px/s.")


def test_8_moving_person_positive_velocity():
    print("\n--- Running Test 8: Moving person velocity measurement ---")
    tracked = {}
    for f in range(10):
        time.sleep(0.04)
        tracked = match_entities([(100 + f * 8, 200, 150 + f * 8, 320, 0.90, "person")], tracked)

    track = list(tracked.values())[0]
    vel = track["smoothed_velocity"]
    assert vel > 50.0
    print(f"  PASS: Moving person measured smoothed velocity = {vel:.2f} px/s.")


def test_9_crossing_paths_assignment():
    print("\n--- Running Test 9: Crossing person trajectories ---")
    tracked = {}
    for step in range(15):
        x1 = 100 + step * 12
        x2 = 300 - step * 12
        dets = [
            (x1, 200, x1 + 40, 320, 0.90, "person"),
            (x2, 200, x2 + 40, 320, 0.90, "person")
        ]
        tracked = match_entities(dets, tracked)
        assert len(tracked) == 2

    ids = sorted(t["track_id"] for t in tracked.values())
    assert ids == [1, 2]
    print(f"  PASS: Crossing people resolved with 1-to-1 assignment (IDs: {ids}).")


def run_all_tracker_tests():
    print("=" * 70)
    print("SENTINEL AI - Person Tracker Validation Suite")
    print("=" * 70)
    test_1_single_detection_multiple_frames()
    test_2_two_independent_people()
    test_3_temporary_missed_detection()
    test_4_detection_missing_beyond_limit_evicted()
    test_5_non_person_detections_do_not_track()
    test_6_trajectory_history_bounded()
    test_7_stationary_person_zero_velocity()
    test_8_moving_person_positive_velocity()
    test_9_crossing_paths_assignment()

    print("\n" + "=" * 70)
    print("ALL 9 PERSON TRACKER TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tracker_tests()

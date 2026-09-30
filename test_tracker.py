"""
Comprehensive Unit Tests for SENTINEL AI Multi-Object Tracker (Phase 2).
Tests track persistence, lifecycle, class-awareness, trajectory history,
image-plane velocity, and competitive assignment.
"""
import time
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app import (
    match_entities,
    TrackState,
    TrackStatus,
    SentinelConfig,
    get_object_group
)


def test_1_single_detection_multiple_frames():
    """TEST 1: One detection across multiple frames maintains the same track_id."""
    print("\n--- Running Test 1: Single detection persistence across frames ---")
    tracked = {}
    
    # Frame 1: Detection at (100, 100, 150, 200)
    dets_f1 = [(100, 100, 150, 200, 0.90, "person")]
    tracked = match_entities(dets_f1, tracked)
    assert len(tracked) == 1, f"Expected 1 track, got {len(tracked)}"
    track_key = list(tracked.keys())[0]
    initial_id = tracked[track_key]["track_id"]
    assert tracked[track_key]["state"] == TrackStatus.NEW
    assert tracked[track_key]["age"] == 1
    assert tracked[track_key]["detection_count"] == 1

    # Frames 2 to 10: slight movement
    for f in range(2, 11):
        x_offset = f * 2
        dets_f = [(100 + x_offset, 100, 150 + x_offset, 200, 0.92, "person")]
        tracked = match_entities(dets_f, tracked)
        assert len(tracked) == 1
        current_id = tracked[track_key]["track_id"]
        assert current_id == initial_id, f"Track ID changed from {initial_id} to {current_id}!"
        assert tracked[track_key]["age"] == f
        assert tracked[track_key]["detection_count"] == f
        assert tracked[track_key]["missed_frames"] == 0
        assert tracked[track_key]["state"] == TrackStatus.ACTIVE

    print(f"  PASS: Track #{initial_id} remained persistent over 10 consecutive frames.")


def test_2_two_independent_moving_objects():
    """TEST 2: Two objects moving independently maintain distinct, stable track IDs."""
    print("\n--- Running Test 2: Two independent moving objects ---")
    tracked = {}

    # Person A starts at x=50, Person B starts at x=400
    for frame in range(1, 15):
        dets = [
            (50 + frame * 3, 100, 90 + frame * 3, 220, 0.88, "person"),
            (400 - frame * 3, 100, 440 - frame * 3, 220, 0.85, "person")
        ]
        tracked = match_entities(dets, tracked)
        assert len(tracked) == 2, f"Expected 2 tracks at frame {frame}, got {len(tracked)}"

    ids = sorted([t["track_id"] for t in tracked.values()])
    assert ids == [1, 2], f"Expected IDs [1, 2], got {ids}"
    print(f"  PASS: Two independent targets tracked with stable IDs {ids}.")


def test_3_temporary_missed_detection():
    """TEST 3: Detection temporarily missing survives up to MAX_MISSED_FRAMES."""
    print("\n--- Running Test 3: Temporary detection loss and recovery ---")
    tracked = {}

    # Frame 1: Detection present
    dets = [(200, 200, 260, 320, 0.90, "person")]
    tracked = match_entities(dets, tracked)
    key = list(tracked.keys())[0]
    original_id = tracked[key]["track_id"]

    # Next 10 frames: Detection disappears (occlusion/missed detection)
    for f in range(1, 11):
        tracked = match_entities([], tracked)
        assert key in tracked, f"Track prematurely evicted at missed frame {f}!"
        assert tracked[key]["state"] == TrackStatus.TEMPORARILY_MISSED
        assert tracked[key]["active"] is False
        assert tracked[key]["missed_frames"] == f

    # Frame 12: Detection reappears near previous location
    dets_reappear = [(205, 200, 265, 320, 0.88, "person")]
    tracked = match_entities(dets_reappear, tracked)
    assert key in tracked, "Reappeared target not matched to existing track!"
    assert tracked[key]["track_id"] == original_id, "Track ID altered upon recovery!"
    assert tracked[key]["missed_frames"] == 0, "missed_frames should reset to 0"
    assert tracked[key]["active"] is True
    assert tracked[key]["state"] == TrackStatus.ACTIVE

    print(f"  PASS: Track #{original_id} survived 10 frames of occlusion and restored identity.")


def test_4_detection_missing_beyond_limit_evicted():
    """TEST 4: Detection missing beyond MAX_MISSED_FRAMES is evicted."""
    print("\n--- Running Test 4: Track eviction beyond MAX_MISSED_FRAMES ---")
    tracked = {}

    # Frame 1: Present
    dets = [(100, 100, 150, 200, 0.85, "person")]
    tracked = match_entities(dets, tracked)
    assert len(tracked) == 1

    # Miss frames up to MAX_MISSED_FRAMES
    max_miss = SentinelConfig.MAX_MISSED_FRAMES
    for _ in range(max_miss):
        tracked = match_entities([], tracked)
        assert len(tracked) == 1

    # One more frame beyond threshold -> eviction
    tracked = match_entities([], tracked)
    assert len(tracked) == 0, f"Expected 0 tracks after {max_miss + 1} missed frames, got {len(tracked)}"
    print(f"  PASS: Track successfully evicted after exceeding {max_miss} missed frames.")


def test_5_class_aware_tracking():
    """TEST 5: Person and luggage in the same location never share a track."""
    print("\n--- Running Test 5: Strict Class-Aware Tracking Isolation ---")
    tracked = {}

    # Person and luggage co-located at approximately (100, 100)
    dets = [
        (100, 100, 160, 240, 0.90, "person"),
        (110, 160, 150, 210, 0.85, "backpack")  # backpack is BAG_OBJECT
    ]
    tracked = match_entities(dets, tracked)
    assert len(tracked) == 2, f"Expected 2 tracks, got {len(tracked)}"

    keys = list(tracked.keys())
    groups = [get_object_group(tracked[k]["class_name"]) for k in keys]
    assert "PERSON" in groups and "BAG_OBJECT" in groups

    # Next frame: person steps away, luggage stays
    dets_step = [
        (180, 100, 240, 240, 0.90, "person"),
        (110, 160, 150, 210, 0.85, "backpack")
    ]
    tracked = match_entities(dets_step, tracked)
    assert len(tracked) == 2

    # Verify no luggage was assigned to person track or vice versa
    for k, info in tracked.items():
        if "PERSON" in k:
            assert info["class_name"] == "person"
            assert info["bbox"][0] > 150  # stepped to x=180
        elif "BAG_OBJECT" in k:
            assert info["class_name"] == "backpack"
            assert info["bbox"][0] < 150  # stayed at x=110

    print("  PASS: Person and luggage maintained strict semantic group separation without cross-match.")


def test_6_trajectory_history_bounded():
    """TEST 6: Trajectory history remains bounded by MAX_TRACK_HISTORY."""
    print("\n--- Running Test 6: Bounded Trajectory History (Memory Safety) ---")
    tracked = {}

    max_hist = SentinelConfig.MAX_TRACK_HISTORY
    # Feed 100 frames
    for f in range(100):
        dets = [(100 + f, 100, 150 + f, 200, 0.90, "person")]
        tracked = match_entities(dets, tracked)

    track = list(tracked.values())[0]
    hist_len = len(track["history"])
    assert hist_len <= max_hist, f"History size {hist_len} exceeds max {max_hist}!"
    print(f"  PASS: History length ({hist_len}) strictly capped at MAX_TRACK_HISTORY ({max_hist}).")


def test_7_stationary_object_zero_velocity():
    """TEST 7: Stationary object has approximately zero image-plane velocity."""
    print("\n--- Running Test 7: Stationary object velocity ---")
    tracked = {}

    # 10 frames at exact same pixel coordinates
    for f in range(10):
        time.sleep(0.01)  # small delta time
        dets = [(300, 300, 350, 350, 0.95, "suitcase")]
        tracked = match_entities(dets, tracked)

    track = list(tracked.values())[0]
    vel = track["smoothed_velocity"]
    assert vel < 1.0, f"Expected near-zero velocity for stationary bag, got {vel:.3f} px/s"
    print(f"  PASS: Stationary object velocity = {vel:.4f} px/s (~0).")


def test_8_moving_object_positive_velocity():
    """TEST 8: Moving object has positive and reasonably stable velocity."""
    print("\n--- Running Test 8: Moving object velocity measurement ---")
    tracked = {}

    # Move 10 pixels every 0.05s = ~200 px/s
    dt = 0.04
    for f in range(10):
        time.sleep(dt)
        dets = [(100 + f * 8, 200, 150 + f * 8, 300, 0.90, "person")]
        tracked = match_entities(dets, tracked)

    track = list(tracked.values())[0]
    vel = track["smoothed_velocity"]
    assert vel > 50.0, f"Expected positive velocity for moving object, got {vel:.2f} px/s"
    print(f"  PASS: Moving object measured smoothed image-plane velocity = {vel:.2f} px/s.")


def test_9_crossing_paths_assignment():
    """TEST 9: Two targets crossing paths evaluate competitive assignment."""
    print("\n--- Running Test 9: Crossing trajectories competitive assignment ---")
    tracked = {}

    # Target 1 moving left-to-right (from x=100 towards x=300)
    # Target 2 moving right-to-left (from x=300 towards x=100)
    for step in range(15):
        x1 = 100 + step * 12
        x2 = 300 - step * 12
        dets = [
            (x1, 200, x1 + 40, 300, 0.90, "person"),
            (x2, 200, x2 + 40, 300, 0.90, "person")
        ]
        tracked = match_entities(dets, tracked)
        assert len(tracked) == 2, f"Target count dropped during crossing at step {step}"

    ids = sorted([t["track_id"] for t in tracked.values()])
    assert ids == [1, 2], f"Expected IDs [1, 2], got {ids}"
    print(f"  PASS: Crossing targets resolved with 1-to-1 competitive assignment (IDs: {ids}).")


def run_all_tracker_tests():
    print("=" * 70)
    print("SENTINEL AI — Multi-Object Tracker Validation Suite (Phase 2)")
    print("=" * 70)
    
    test_1_single_detection_multiple_frames()
    test_2_two_independent_moving_objects()
    test_3_temporary_missed_detection()
    test_4_detection_missing_beyond_limit_evicted()
    test_5_class_aware_tracking()
    test_6_trajectory_history_bounded()
    test_7_stationary_object_zero_velocity()
    test_8_moving_object_positive_velocity()
    test_9_crossing_paths_assignment()

    print("\n" + "=" * 70)
    print("ALL 9 TRACKER TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tracker_tests()

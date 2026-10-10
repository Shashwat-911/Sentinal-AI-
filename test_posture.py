"""
Unit tests for deterministic human posture estimation and possible fall alerts.
Tests synthetic keypoint configurations for:
- STANDING
- SITTING
- BENDING
- LYING_DOWN / POSSIBLE_FALL
- MISSING KEYPOINTS
- LOW CONFIDENCE
- Temporal confirmation and alert cooldowns
"""

import sys
import os
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from sentinel.posture import (
    classify_posture, estimate_postures,
    KPT_LEFT_SHOULDER, KPT_RIGHT_SHOULDER,
    KPT_LEFT_HIP, KPT_RIGHT_HIP,
    KPT_LEFT_KNEE, KPT_RIGHT_KNEE,
    KPT_LEFT_ANKLE, KPT_RIGHT_ANKLE
)
from sentinel.tracking import TrackState


def _make_empty_keypoints(conf=1.0):
    """Create a 17-keypoint template filled with zeros."""
    return [[0.0, 0.0, conf] for _ in range(17)]


def test_standing_synthetic_pose():
    """Verify STANDING posture geometry: upright torso and extended vertical legs."""
    print("\n--- Test: Standing Synthetic Pose ---")
    kpts = _make_empty_keypoints(conf=0.95)
    # Shoulders
    kpts[KPT_LEFT_SHOULDER] = [180.0, 150.0, 0.95]
    kpts[KPT_RIGHT_SHOULDER] = [220.0, 150.0, 0.95]
    # Hips
    kpts[KPT_LEFT_HIP] = [185.0, 300.0, 0.95]
    kpts[KPT_RIGHT_HIP] = [215.0, 300.0, 0.95]
    # Knees
    kpts[KPT_LEFT_KNEE] = [185.0, 420.0, 0.95]
    kpts[KPT_RIGHT_KNEE] = [215.0, 420.0, 0.95]
    # Ankles
    kpts[KPT_LEFT_ANKLE] = [185.0, 540.0, 0.95]
    kpts[KPT_RIGHT_ANKLE] = [215.0, 540.0, 0.95]

    bbox = [160, 120, 240, 560] # vertical bounding box (H: 440, W: 80, HWR=5.5)
    posture, conf = classify_posture(kpts, bbox=bbox)
    assert posture == "STANDING", f"Expected STANDING, got {posture}"
    assert conf >= 0.8, f"Expected high confidence, got {conf}"
    print(f"  PASS: Classified as {posture} with conf={conf:.2f}")


def test_sitting_synthetic_pose():
    """Verify SITTING posture geometry: upright torso with horizontal thighs / bent knees."""
    print("\n--- Test: Sitting Synthetic Pose ---")
    kpts = _make_empty_keypoints(conf=0.92)
    # Shoulders (upright torso)
    kpts[KPT_LEFT_SHOULDER] = [200.0, 150.0, 0.92]
    kpts[KPT_RIGHT_SHOULDER] = [240.0, 150.0, 0.92]
    # Hips
    kpts[KPT_LEFT_HIP] = [200.0, 280.0, 0.92]
    kpts[KPT_RIGHT_HIP] = [240.0, 280.0, 0.92]
    # Knees positioned forward horizontally at similar y level
    kpts[KPT_LEFT_KNEE] = [320.0, 290.0, 0.92]
    kpts[KPT_RIGHT_KNEE] = [360.0, 290.0, 0.92]
    # Ankles down vertically from knees
    kpts[KPT_LEFT_ANKLE] = [320.0, 400.0, 0.92]
    kpts[KPT_RIGHT_ANKLE] = [360.0, 400.0, 0.92]

    bbox = [180, 130, 380, 420]
    posture, conf = classify_posture(kpts, bbox=bbox)
    assert posture == "SITTING", f"Expected SITTING, got {posture}"
    assert conf >= 0.8, f"Expected high confidence, got {conf}"
    print(f"  PASS: Classified as {posture} with conf={conf:.2f}")


def test_bending_synthetic_pose():
    """Verify BENDING posture geometry: torso inclined downward while hips elevated above feet."""
    print("\n--- Test: Bending Synthetic Pose ---")
    kpts = _make_empty_keypoints(conf=0.90)
    # Shoulders bent forward and low
    kpts[KPT_LEFT_SHOULDER] = [320.0, 280.0, 0.90]
    kpts[KPT_RIGHT_SHOULDER] = [350.0, 280.0, 0.90]
    # Hips elevated behind
    kpts[KPT_LEFT_HIP] = [210.0, 210.0, 0.90]
    kpts[KPT_RIGHT_HIP] = [240.0, 210.0, 0.90]
    # Knees supporting
    kpts[KPT_LEFT_KNEE] = [205.0, 340.0, 0.90]
    kpts[KPT_RIGHT_KNEE] = [235.0, 340.0, 0.90]
    # Ankles on ground
    kpts[KPT_LEFT_ANKLE] = [205.0, 460.0, 0.90]
    kpts[KPT_RIGHT_ANKLE] = [235.0, 460.0, 0.90]

    bbox = [180, 190, 380, 480]
    posture, conf = classify_posture(kpts, bbox=bbox)
    assert posture == "BENDING", f"Expected BENDING, got {posture}"
    print(f"  PASS: Classified as {posture} with conf={conf:.2f}")


def test_lying_down_synthetic_pose():
    """Verify LYING_DOWN posture geometry: predominantly horizontal body alignment."""
    print("\n--- Test: Lying Down / Possible Fall Synthetic Pose ---")
    kpts = _make_empty_keypoints(conf=0.94)
    # Shoulders
    kpts[KPT_LEFT_SHOULDER] = [120.0, 380.0, 0.94]
    kpts[KPT_RIGHT_SHOULDER] = [120.0, 420.0, 0.94]
    # Hips
    kpts[KPT_LEFT_HIP] = [260.0, 385.0, 0.94]
    kpts[KPT_RIGHT_HIP] = [260.0, 425.0, 0.94]
    # Knees
    kpts[KPT_LEFT_KNEE] = [380.0, 390.0, 0.94]
    kpts[KPT_RIGHT_KNEE] = [380.0, 430.0, 0.94]
    # Ankles
    kpts[KPT_LEFT_ANKLE] = [500.0, 395.0, 0.94]
    kpts[KPT_RIGHT_ANKLE] = [500.0, 435.0, 0.94]

    bbox = [100, 360, 520, 450] # wide horizontal box (W: 420, H: 90, AR=4.67)
    posture, conf = classify_posture(kpts, bbox=bbox)
    assert posture == "LYING_DOWN", f"Expected LYING_DOWN, got {posture}"
    print(f"  PASS: Classified as {posture} with conf={conf:.2f}")


def test_missing_keypoints_produces_unknown():
    """Verify that missing critical keypoints (e.g., both shoulders or both hips) yields UNKNOWN."""
    print("\n--- Test: Missing Keypoints ---")
    kpts = _make_empty_keypoints(conf=0.0)
    # Only knees and ankles present
    kpts[KPT_LEFT_KNEE] = [200.0, 300.0, 0.9]
    kpts[KPT_LEFT_ANKLE] = [200.0, 450.0, 0.9]

    posture, conf = classify_posture(kpts)
    assert posture == "UNKNOWN", f"Expected UNKNOWN for missing torso, got {posture}"
    print(f"  PASS: Handled missing keypoints safely -> {posture}")


def test_low_confidence_produces_unknown():
    """Verify that low-confidence keypoints yield UNKNOWN."""
    print("\n--- Test: Low Confidence ---")
    kpts = _make_empty_keypoints(conf=0.15)
    # Positions look like standing but confidence is far below 0.35
    kpts[KPT_LEFT_SHOULDER] = [200.0, 100.0, 0.15]
    kpts[KPT_LEFT_HIP] = [200.0, 250.0, 0.15]

    posture, conf = classify_posture(kpts, min_kpt_conf=0.35)
    assert posture == "UNKNOWN", f"Expected UNKNOWN for low confidence, got {posture}"
    print(f"  PASS: Handled low confidence safely -> {posture}")


def test_temporal_fall_confirmation_and_cooldown():
    """Verify temporal confirmation requirement and alert cooldown for POSSIBLE_FALL."""
    print("\n--- Test: Temporal Fall Confirmation & Alert Cooldown ---")
    now = time.time()
    track = TrackState(
        track_id=1, class_name="person", bbox=(100, 360, 520, 450), confidence=0.9,
        first_seen=now - 10, centroid=(310, 400),
        posture="LYING_DOWN", pose_confidence=0.92
    )
    tracked_entities = {"p1": track}

    # Create dummy mock pose detector that yields candidate matching track
    kpts = _make_empty_keypoints(conf=0.94)
    kpts[KPT_LEFT_SHOULDER] = [120.0, 380.0, 0.94]
    kpts[KPT_LEFT_HIP] = [260.0, 385.0, 0.94]
    kpts[KPT_LEFT_KNEE] = [380.0, 390.0, 0.94]
    kpts[KPT_LEFT_ANKLE] = [500.0, 395.0, 0.94]

    class MockResults:
        def __init__(self):
            import torch
            self.boxes = type("Boxes", (), {"xyxy": torch.tensor([[100.0, 360.0, 520.0, 450.0]])})()
            self.keypoints = type("Keypoints", (), {"data": torch.tensor([kpts])})()

    class MockPoseModel:
        def __call__(self, frame, **kwargs):
            return [MockResults()]

    mock_model = MockPoseModel()
    dummy_frame = np.zeros((600, 800, 3), dtype=np.uint8)

    # Frame 1 to 24: under temporal confirmation threshold (default 25 frames)
    for f in range(1, 25):
        alerts = estimate_postures(dummy_frame, tracked_entities, mock_model, current_time=now + f * 0.04)
        assert len(alerts) == 0, f"Frame {f} should NOT trigger alert before temporal confirmation"

    # Frame 25: reaches temporal confirmation threshold -> alert must fire!
    alerts = estimate_postures(dummy_frame, tracked_entities, mock_model, current_time=now + 25 * 0.04)
    assert len(alerts) == 1, "Frame 25 MUST trigger POSSIBLE_FALL alert"
    assert alerts[0]["behavior"] == "POSSIBLE_FALL"
    assert alerts[0]["track_id"] == 1
    print("  PASS: Temporal confirmation successfully gated possible-fall alert until 25 sustained frames.")

    # Frame 26 to 40 (within cooldown): must NOT trigger repeat alerts
    for f in range(26, 40):
        alerts = estimate_postures(dummy_frame, tracked_entities, mock_model, current_time=now + f * 0.04)
        assert len(alerts) == 0, f"Cooldown should suppress repeat alerts at frame {f}"
    print("  PASS: Cooldown correctly suppressed repetitive alerts.")


if __name__ == "__main__":
    test_standing_synthetic_pose()
    test_sitting_synthetic_pose()
    test_bending_synthetic_pose()
    test_lying_down_synthetic_pose()
    test_missing_keypoints_produces_unknown()
    test_low_confidence_produces_unknown()
    test_temporal_fall_confirmation_and_cooldown()
    print("\n" + "=" * 60)
    print("ALL POSTURE ESTIMATION UNIT TESTS PASSED!")
    print("=" * 60)

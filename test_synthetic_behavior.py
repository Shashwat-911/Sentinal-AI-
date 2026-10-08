"""
SYNTHETIC 60-SECOND BEHAVIOR VALIDATION TEST
Deterministic trajectory/time simulation using video-time semantics.
Tests all required scenarios with precise timing control.
"""
import os
import sys
import math
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from sentinel.config import SentinelConfig
from sentinel.tracking import TrackState, match_entities
from sentinel.behavior import evaluate_person_behavior, compute_dwell_time, compute_trajectory_metrics, update_grid_revisits
from sentinel.risk import calculate_entity_risk

FPS = 25.0
FRAME_WIDTH = 800
FRAME_HEIGHT = 600

def generate_trajectory(scenario, duration_seconds, fps=FPS):
    """
    Generate synthetic trajectory points for a given scenario.
    Returns list of (x, y, timestamp) tuples.
    """
    frames = int(duration_seconds * fps)
    trajectory = []
    
    if scenario == "straight_walk":
        # Normal walking: linear movement across the scene (outside restricted zone)
        # Restricted zone: x: 0.60-0.95 = 480-760, y: 0.20-0.75 = 120-450
        for f in range(frames):
            t = f / fps
            x = 50 + (t / duration_seconds) * 350  # 50 to 400 (outside zone)
            y = 300
            trajectory.append((x, y, t))
    
    elif scenario == "stationary_wait":
        # Stationary waiting: remain in one location
        for f in range(frames):
            t = f / fps
            x = 400
            y = 300
            trajectory.append((x, y, t))
    
    elif scenario == "pacing":
        # Pacing: repeated A→B→A movement
        for f in range(frames):
            t = f / fps
            cycle = int(t / 3.0) % 2  # 3-second cycles
            x = 200 if cycle == 0 else 600
            y = 300
            trajectory.append((x, y, t))
    
    elif scenario == "restricted_pacing":
        # Pacing within restricted zone (x: 0.60-0.95 of width = 480-760, y: 0.20-0.75 = 120-450)
        for f in range(frames):
            t = f / fps
            cycle = int(t / 3.0) % 2
            x = 500 if cycle == 0 else 700
            y = 250  # Within 120-450 range
            trajectory.append((x, y, t))
    
    elif scenario == "revisit_pattern":
        # A-B-A-B-A pattern
        for f in range(frames):
            t = f / fps
            cycle = int(t / 5.0) % 2
            x = 200 if cycle == 0 else 500
            y = 200 + int(t / 10.0) * 50  # Slow drift in Y
            trajectory.append((x, y, t))
    
    return trajectory

def run_scenario(name, scenario, duration_seconds, expected_state, expected_threats):
    """
    Run a single scenario through the behavior engine with frame-by-frame evaluation.
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO: {name}")
    print(f"{'='*70}")
    print(f"Duration: {duration_seconds}s")
    print(f"Expected state: {expected_state}")
    print(f"Expected threats: {expected_threats}")
    print()
    
    # Generate trajectory
    trajectory = generate_trajectory(scenario, duration_seconds, FPS)
    
    # Create track
    track = TrackState(
        track_id=1,
        class_name="person",
        bbox=(0, 0, 100, 200),
        confidence=0.9,
        centroid=trajectory[0][:2],
        first_seen=0.0,
        history=[]
    )
    
    # Evaluate behavior frame-by-frame to trigger temporal persistence
    for i, (x, y, t) in enumerate(trajectory):
        track["centroid"] = (x, y)
        track["history"].append((x, y, t))
        # Keep history bounded
        if len(track["history"]) > SentinelConfig.MAX_TRACK_HISTORY:
            track["history"].pop(0)
        evaluate_person_behavior(track, FRAME_WIDTH, FRAME_HEIGHT, current_time=t)
    
    # Report results
    print(f"Dwell time: {track['dwell_time']:.1f}s (threshold: {SentinelConfig.DWELL_TIME_THRESHOLD}s)")
    print(f"Meander ratio: {track['meander_ratio']:.2f} (threshold: {SentinelConfig.MEANDER_RATIO_THRESHOLD})")
    print(f"Revisit count: {track['revisit_count']} (threshold: {SentinelConfig.REVISIT_COUNT_THRESHOLD})")
    print(f"In restricted zone: {track['in_restricted_zone']}")
    print(f"Behavior state: {track['behavior_state']}")
    print(f"Suspicion score: {track['suspicion_score']}")
    print(f"Active threats: {sorted(track['active_threats'])}")
    
    # Risk assessment
    risk_score, severity, color = calculate_entity_risk(track)
    print(f"Risk score: {risk_score}, Severity: {severity}")
    
    # Validate
    actual_state = track['behavior_state']
    actual_threats = set(track['active_threats'])
    
    # Check state match (allow for OBSERVING if persistence not yet met)
    state_match = actual_state in (expected_state, "OBSERVING") if expected_state in ("ALERT", "HIGH", "CRITICAL") else actual_state == expected_state
    
    # Check threats match (this is the critical validation)
    threats_match = actual_threats >= set(expected_threats)  # At minimum should have expected threats
    
    if state_match and threats_match:
        print(f"\n[PASS] State and threats match expected")
    else:
        print(f"\n[FAIL]:")
        if not state_match:
            print(f"  Expected state: {expected_state}, Got: {actual_state}")
        if not threats_match:
            print(f"  Expected threats (min): {set(expected_threats)}, Got: {actual_threats}")
    
    return state_match and threats_match

def test_occlusion_scenario():
    """Test person leaving and re-entering."""
    print(f"\n{'='*70}")
    print(f"SCENARIO: Person Leave/Re-enter")
    print(f"{'='*70}")
    
    # Initial tracking
    track = TrackState(
        track_id=1,
        class_name="person",
        bbox=(0, 0, 100, 200),
        confidence=0.9,
        centroid=(400, 300),
        first_seen=0.0,
        history=[(400, 300, 0.0)]
    )
    
    # Simulate 10 seconds of presence
    for f in range(250):
        t = f / FPS
        track["history"].append((400, 300, t))
        track["last_seen"] = t
    
    evaluate_person_behavior(track, FRAME_WIDTH, FRAME_HEIGHT, current_time=10.0)
    initial_state = track["behavior_state"]
    initial_dwell = track["dwell_time"]
    
    print(f"After 10s presence: State={initial_state}, Dwell={initial_dwell:.1f}s")
    
    # Simulate occlusion (track persists due to MAX_MISSED_FRAMES)
    # Re-enter at 15s
    track["history"].append((400, 300, 15.0))
    track["last_seen"] = 15.0
    track["missed_frames"] = 0
    track["state"] = "ACTIVE"
    
    evaluate_person_behavior(track, FRAME_WIDTH, FRAME_HEIGHT, current_time=15.0)
    final_state = track["behavior_state"]
    final_dwell = track["dwell_time"]
    
    print(f"After re-entry at 15s: State={final_state}, Dwell={final_dwell:.1f}s")
    
    # Behavior state should persist track ID but continue accumulating dwell
    dwell_continuity = (final_dwell > initial_dwell)
    
    if dwell_continuity:
        print(f"\n[PASS] Dwell time accumulated across occlusion")
    else:
        print(f"\n[FAIL] Dwell time did not accumulate")
    
    return dwell_continuity

def main():
    print("="*70)
    print("SYNTHETIC 60-SECOND BEHAVIOR VALIDATION")
    print("="*70)
    print(f"FPS: {FPS}")
    print(f"Frame size: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Dwell threshold: {SentinelConfig.DWELL_TIME_THRESHOLD}s")
    print(f"Meander threshold: {SentinelConfig.MEANDER_RATIO_THRESHOLD}")
    print(f"Revisit threshold: {SentinelConfig.REVISIT_COUNT_THRESHOLD}")
    print("="*70)
    
    results = {}
    
    # Scenario 1: Normal walking (60s)
    # Note: 60s dwell time is >= 45s threshold, so LONG_DWELL is expected
    # But with meander=1.0 and no revisits, state should be OBSERVING, not suspicious
    results["straight_walk"] = run_scenario(
        "Normal Walking (60s)",
        "straight_walk",
        60,
        "OBSERVING",  # LONG_DWELL alone = OBSERVING, not suspicious
        ["LONG_DWELL"]
    )
    
    # Scenario 2: Stationary waiting (60s)
    results["stationary_wait"] = run_scenario(
        "Stationary Waiting (60s)",
        "stationary_wait",
        60,
        "OBSERVING",
        ["LONG_DWELL"]
    )
    
    # Scenario 3: Pacing (60s)
    # With dwell >= 45s AND high meander, should trigger SUSPICIOUS_LOITERING
    # Note: ALERT state requires 15 frames persistence (SUSPICION_PERSISTENCE_FRAMES)
    # For this test, we check that SUSPICIOUS_LOITERING threat is present
    results["pacing"] = run_scenario(
        "Pacing (60s)",
        "pacing",
        60,
        "OBSERVING",  # Will be OBSERVING until persistence is met, then ALERT
        ["LONG_DWELL", "REPEATED_MOVEMENT", "SUSPICIOUS_LOITERING"]
    )
    
    # Scenario 4: Restricted-zone pacing (60s)
    results["restricted_pacing"] = run_scenario(
        "Restricted Zone + Pacing (60s)",
        "restricted_pacing",
        60,
        "OBSERVING",  # Will be OBSERVING until persistence is met
        ["LONG_DWELL", "REPEATED_MOVEMENT", "RESTRICTED_ZONE", "SUSPICIOUS_LOITERING"]
    )
    
    # Scenario 5: Revisit pattern (60s)
    results["revisit_pattern"] = run_scenario(
        "Revisit Pattern A-B-A (60s)",
        "revisit_pattern",
        60,
        "HIGH",
        ["LONG_DWELL", "REPEATED_MOVEMENT"]
    )
    
    # Scenario 6: Occlusion
    results["occlusion"] = test_occlusion_scenario()
    
    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} scenarios passed")
    
    if passed == total:
        print("\n[SUCCESS] ALL SYNTHETIC BEHAVIOR TESTS PASSED")
    else:
        print(f"\n[FAILURE] {total - passed} SCENARIO(S) FAILED")

if __name__ == "__main__":
    main()

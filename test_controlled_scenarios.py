"""
SENTINEL AI — Controlled 60-Second Behavioral Validation Test Suite (Phases 3 & 6)
Deterministic, video-time frame-by-frame evaluation of all required scenarios:
1. Normal walking (A -> B -> C -> D -> E for 60s) -> NORMAL
2. Long dwell (Stationary for 60s) -> Before 45s: NO LONG_DWELL; At/after 45s: LONG_DWELL / OBSERVING (NOT suspicious)
3. Prolonged pacing (A <-> B for 60s) -> At/after 45s: LONG_DWELL + high meander + revisits -> SUSPICIOUS LOITERING (ALERT)
4. Restricted + pacing (Enters restricted zone and paces) -> RESTRICTED_ZONE_ENTRY -> CRITICAL / HIGH_CONFIDENCE_SUSPICIOUS
5. Normal person + one turn (A -> B -> C) -> NORMAL
6. Single revisit (A -> B -> A) -> Revisit recorded, but NOT immediately suspicious
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
from sentinel.alerting import draw_overlays

FPS = 25.0
FRAME_WIDTH = 800
FRAME_HEIGHT = 600

# Restricted zone in 800x600 coordinates:
# X: [0.60 * 800, 0.95 * 800] = [480, 760]
# Y: [0.20 * 600, 0.75 * 600] = [120, 450]


def run_controlled_scenario_1_normal_walking():
    """
    SCENARIO 1: NORMAL WALKING
    Person moves A -> B -> C -> D -> E across the scene for 60 seconds (outside restricted zone).
    Expected: NORMAL. No suspicious alert.
    """
    duration = 60.0
    total_frames = int(duration * FPS)
    track = TrackState(
        track_id=1, class_name="person", bbox=(30, 200, 70, 320), confidence=0.9,
        centroid=(50.0, 300.0), first_seen=0.0, current_time=0.0
    )
    
    alert_times = []
    
    # Linear transit from x=50 to x=650 over 60 seconds (outside restricted zone: y=500 is outside Y[120, 450])
    for f in range(total_frames):
        t = f / FPS
        # Waypoints: A(50), B(200), C(350), D(500), E(650) at y=520 (outside restricted zone)
        x = 50.0 + (t / duration) * 600.0
        y = 520.0
        track["centroid"] = (x, y)
        track["history"].append((x, y, t))
        if len(track["history"]) > SentinelConfig.MAX_TRACK_HISTORY:
            track["history"].pop(0)
            
        evaluate_person_behavior(track, FRAME_WIDTH, FRAME_HEIGHT, current_time=t)
        if track["behavior_state"] in ("ALERT", "CRITICAL", "HIGH") and not alert_times:
            alert_times.append(t)
            
    risk_score, severity, _ = calculate_entity_risk(track, current_time=duration)
    
    is_pass = (
        track["behavior_state"] == "NORMAL" and
        "SUSPICIOUS_LOITERING" not in track["active_threats"] and
        "LOITERING" not in track["active_threats"] and
        "RESTRICTED_ZONE" not in track["active_threats"] and
        len(alert_times) == 0
    )
    
    return {
        "scenario": "1. Normal Walking (A->B->C->D->E)",
        "duration": f"{duration:.1f}s",
        "expected_state": "NORMAL (No suspicious alert)",
        "observed_state": track["behavior_state"],
        "alert_time": "None" if not alert_times else f"{alert_times[0]:.1f}s",
        "dwell_time": f"{track['dwell_time']:.1f}s",
        "meander_ratio": f"{track['meander_ratio']:.2f}",
        "revisit_count": track["revisit_count"],
        "restricted_zone": "YES" if track["in_restricted_zone"] else "NO",
        "final_suspicion_state": f"{track['behavior_state']} (Risk: {severity}, {risk_score}%)",
        "pass": is_pass
    }


def run_controlled_scenario_2_long_dwell():
    """
    SCENARIO 2: LONG DWELL
    Person remains in one area for 60 seconds.
    Expected:
    Before 45 seconds: NO LONG_DWELL.
    At/after 45 seconds: LONG_DWELL / OBSERVING.
    But: NOT automatically suspicious.
    """
    duration = 60.0
    total_frames = int(duration * FPS)
    track = TrackState(
        track_id=1, class_name="person", bbox=(230, 230, 270, 350), confidence=0.9,
        centroid=(250.0, 250.0), first_seen=0.0, current_time=0.0
    )
    
    long_dwell_time = None
    suspicious_alert = False
    state_at_40s = None
    threats_at_40s = None
    
    for f in range(total_frames):
        t = f / FPS
        # Person remains stationary in the center of cell (row 2, col 2)
        x = 250.0
        y = 250.0
        track["centroid"] = (x, y)
        track["history"].append((x, y, t))
        if len(track["history"]) > SentinelConfig.MAX_TRACK_HISTORY:
            track["history"].pop(0)
            
        evaluate_person_behavior(track, FRAME_WIDTH, FRAME_HEIGHT, current_time=t)
        
        if math.isclose(t, 40.0, abs_tol=0.02):
            state_at_40s = track["behavior_state"]
            threats_at_40s = set(track["active_threats"])
            
        if "LONG_DWELL" in track["active_threats"] and long_dwell_time is None:
            long_dwell_time = t
            
        if track["behavior_state"] in ("ALERT", "CRITICAL", "HIGH"):
            suspicious_alert = True
            
    risk_score, severity, _ = calculate_entity_risk(track, current_time=duration)
    
    is_pass = (
        long_dwell_time is not None and
        long_dwell_time >= 44.9 and
        "LONG_DWELL" not in threats_at_40s and
        state_at_40s == "NORMAL" and
        track["behavior_state"] == "OBSERVING" and
        not suspicious_alert and
        severity in ("NORMAL", "LOW") and
        "SUSPICIOUS_LOITERING" not in track["active_threats"]
    )
    
    return {
        "scenario": "2. Long Dwell (Stationary Wait)",
        "duration": f"{duration:.1f}s",
        "expected_state": "OBSERVING / LONG_DWELL (NOT suspicious)",
        "observed_state": track["behavior_state"],
        "alert_time": f"LONG_DWELL @ {long_dwell_time:.1f}s (None suspicious)",
        "dwell_time": f"{track['dwell_time']:.1f}s",
        "meander_ratio": f"{track['meander_ratio']:.2f}",
        "revisit_count": track["revisit_count"],
        "restricted_zone": "YES" if track["in_restricted_zone"] else "NO",
        "final_suspicion_state": f"{track['behavior_state']} (Risk: {severity}, {risk_score}%)",
        "pass": is_pass
    }


def run_controlled_scenario_3_prolonged_pacing():
    """
    SCENARIO 3: PROLONGED PACING
    Person repeatedly moves A -> B -> A -> B -> A -> B -> A -> B for 60 seconds (outside restricted zone).
    Expected:
    LONG_DWELL + high meander evidence + revisit evidence.
    Eventually: SUSPICIOUS LOITERING (ALERT) after temporal persistence.
    """
    duration = 60.0
    total_frames = int(duration * FPS)
    track = TrackState(
        track_id=1, class_name="person", bbox=(130, 480, 170, 580), confidence=0.9,
        centroid=(150.0, 520.0), first_seen=0.0, current_time=0.0
    )
    
    alert_time = None
    long_dwell_time = None
    
    # Pacing between x=150 and x=350 at y=520 (outside restricted zone) every 3.0 seconds
    for f in range(total_frames):
        t = f / FPS
        phase = (t % 6.0) / 3.0
        if phase <= 1.0:
            x = 150.0 + phase * 200.0
        else:
            x = 350.0 - (phase - 1.0) * 200.0
        y = 520.0
        
        track["centroid"] = (x, y)
        track["history"].append((x, y, t))
        if len(track["history"]) > SentinelConfig.MAX_TRACK_HISTORY:
            track["history"].pop(0)
            
        evaluate_person_behavior(track, FRAME_WIDTH, FRAME_HEIGHT, current_time=t)
        
        if "LONG_DWELL" in track["active_threats"] and long_dwell_time is None:
            long_dwell_time = t
            
        if ("SUSPICIOUS_LOITERING" in track["active_threats"] or track["behavior_state"] in ("ALERT", "HIGH")) and alert_time is None:
            alert_time = t
            
    risk_score, severity, _ = calculate_entity_risk(track, current_time=duration)
    
    is_pass = (
        track["dwell_time"] >= 45.0 and
        track["meander_ratio"] >= SentinelConfig.MEANDER_RATIO_THRESHOLD and
        track["revisit_count"] >= SentinelConfig.REVISIT_COUNT_THRESHOLD and
        "SUSPICIOUS_LOITERING" in track["active_threats"] and
        track["behavior_state"] in ("ALERT", "HIGH") and
        alert_time is not None and
        alert_time >= 45.0  # Must not alert before dwell threshold!
    )
    
    return {
        "scenario": "3. Prolonged Pacing (A<->B for 60s)",
        "duration": f"{duration:.1f}s",
        "expected_state": "SUSPICIOUS LOITERING / ALERT",
        "observed_state": track["behavior_state"],
        "alert_time": f"ALERT @ {alert_time:.1f}s (after 45s dwell)",
        "dwell_time": f"{track['dwell_time']:.1f}s",
        "meander_ratio": f"{track['meander_ratio']:.2f}",
        "revisit_count": track["revisit_count"],
        "restricted_zone": "YES" if track["in_restricted_zone"] else "NO",
        "final_suspicion_state": f"{track['behavior_state']} (Risk: {severity}, {risk_score}%)",
        "pass": is_pass
    }


def run_controlled_scenario_4_restricted_pacing():
    """
    SCENARIO 4: RESTRICTED + PACING
    Person enters restricted zone. Then remains there and repeatedly paces.
    Expected:
    RESTRICTED_ZONE_ENTRY event, followed by CRITICAL / HIGH_CONFIDENCE_SUSPICIOUS.
    """
    duration = 60.0
    total_frames = int(duration * FPS)
    track = TrackState(
        track_id=1, class_name="person", bbox=(480, 240, 520, 360), confidence=0.9,
        centroid=(520.0, 280.0), first_seen=0.0, current_time=0.0
    )
    
    zone_entry_time = None
    critical_alert_time = None
    
    # Restricted zone X: 480 - 760, Y: 120 - 450
    # Pacing inside restricted zone between x=520 and x=700 at y=280
    for f in range(total_frames):
        t = f / FPS
        phase = (t % 5.0) / 2.5
        if phase <= 1.0:
            x = 520.0 + phase * 180.0
        else:
            x = 700.0 - (phase - 1.0) * 180.0
        y = 280.0
        
        track["centroid"] = (x, y)
        track["history"].append((x, y, t))
        if len(track["history"]) > SentinelConfig.MAX_TRACK_HISTORY:
            track["history"].pop(0)
            
        evaluate_person_behavior(track, FRAME_WIDTH, FRAME_HEIGHT, current_time=t)
        
        if "RESTRICTED_ZONE" in track["active_threats"] and zone_entry_time is None:
            zone_entry_time = t
            
        if ("HIGH_CONFIDENCE_SUSPICIOUS" in track["active_threats"] or track["behavior_state"] == "CRITICAL") and critical_alert_time is None:
            critical_alert_time = t
            
    risk_score, severity, _ = calculate_entity_risk(track, current_time=duration)
    
    is_pass = (
        zone_entry_time is not None and
        zone_entry_time < 2.0 and
        "RESTRICTED_ZONE" in track["active_threats"] and
        track["in_restricted_zone"] is True and
        "HIGH_CONFIDENCE_SUSPICIOUS" in track["active_threats"] and
        track["behavior_state"] == "CRITICAL" and
        severity == "CRITICAL"
    )
    
    return {
        "scenario": "4. Restricted Zone + Pacing",
        "duration": f"{duration:.1f}s",
        "expected_state": "RESTRICTED ZONE -> CRITICAL ALERT",
        "observed_state": track["behavior_state"],
        "alert_time": f"Zone @ {zone_entry_time:.1f}s | CRITICAL @ {critical_alert_time:.1f}s",
        "dwell_time": f"{track['dwell_time']:.1f}s",
        "meander_ratio": f"{track['meander_ratio']:.2f}",
        "revisit_count": track["revisit_count"],
        "restricted_zone": "YES",
        "final_suspicion_state": f"{track['behavior_state']} (Risk: {severity}, {risk_score}%)",
        "pass": is_pass
    }


def run_controlled_scenario_5_one_turn():
    """
    SCENARIO 5: NORMAL PERSON + ONE TURN
    Person walks A -> B -> C with one direction change (outside restricted zone).
    Expected: NORMAL.
    """
    duration = 20.0
    total_frames = int(duration * FPS)
    track = TrackState(
        track_id=1, class_name="person", bbox=(80, 140, 120, 260), confidence=0.9,
        centroid=(100.0, 200.0), first_seen=0.0, current_time=0.0
    )
    
    # 0s to 10s: moves (100, 200) -> (300, 200)
    # 10s to 20s: moves (300, 200) -> (300, 400)
    for f in range(total_frames):
        t = f / FPS
        if t <= 10.0:
            x = 100.0 + (t / 10.0) * 200.0
            y = 200.0
        else:
            x = 300.0
            y = 200.0 + ((t - 10.0) / 10.0) * 200.0
            
        track["centroid"] = (x, y)
        track["history"].append((x, y, t))
        if len(track["history"]) > SentinelConfig.MAX_TRACK_HISTORY:
            track["history"].pop(0)
            
        evaluate_person_behavior(track, FRAME_WIDTH, FRAME_HEIGHT, current_time=t)
        
    risk_score, severity, _ = calculate_entity_risk(track, current_time=duration)
    
    is_pass = (
        track["behavior_state"] == "NORMAL" and
        track["meander_ratio"] < SentinelConfig.MEANDER_RATIO_THRESHOLD and
        "SUSPICIOUS_LOITERING" not in track["active_threats"] and
        "REPEATED_MOVEMENT" not in track["active_threats"] and
        len(track["active_threats"]) == 0
    )
    
    return {
        "scenario": "5. Normal Person + One Turn (A->B->C)",
        "duration": f"{duration:.1f}s",
        "expected_state": "NORMAL",
        "observed_state": track["behavior_state"],
        "alert_time": "None",
        "dwell_time": f"{track['dwell_time']:.1f}s",
        "meander_ratio": f"{track['meander_ratio']:.2f}",
        "revisit_count": track["revisit_count"],
        "restricted_zone": "NO",
        "final_suspicion_state": f"{track['behavior_state']} (Risk: {severity}, {risk_score}%)",
        "pass": is_pass
    }


def run_controlled_scenario_6_single_revisit():
    """
    SCENARIO 6: SINGLE REVISIT
    Person walks A -> B -> A (one turnaround, outside restricted zone).
    Expected: Revisit evidence recorded (revisit_count >= 1), but NOT immediately suspicious.
    """
    duration = 20.0
    total_frames = int(duration * FPS)
    track = TrackState(
        track_id=1, class_name="person", bbox=(80, 140, 120, 260), confidence=0.9,
        centroid=(100.0, 520.0), first_seen=0.0, current_time=0.0
    )
    
    # 0s to 10s: moves (150, 520) [Cell 1] -> (250, 520) [Cell 2]
    # 10s to 20s: moves (250, 520) [Cell 2] -> (150, 520) [Cell 1] (revisit=1)
    for f in range(total_frames):
        t = f / FPS
        if t <= 10.0:
            x = 150.0 + (t / 10.0) * 100.0
        else:
            x = 250.0 - ((t - 10.0) / 10.0) * 100.0
        y = 520.0
        
        track["centroid"] = (x, y)
        track["history"].append((x, y, t))
        if len(track["history"]) > SentinelConfig.MAX_TRACK_HISTORY:
            track["history"].pop(0)
            
        evaluate_person_behavior(track, FRAME_WIDTH, FRAME_HEIGHT, current_time=t)
        
    risk_score, severity, _ = calculate_entity_risk(track, current_time=duration)
    
    is_pass = (
        track["revisit_count"] == 1 and
        track["behavior_state"] in ("NORMAL", "LOW") and
        "SUSPICIOUS_LOITERING" not in track["active_threats"] and
        severity in ("NORMAL", "LOW")
    )
    
    return {
        "scenario": "6. Single Revisit (A->B->A)",
        "duration": f"{duration:.1f}s",
        "expected_state": "Revisit recorded, NOT suspicious",
        "observed_state": f"Revisits={track['revisit_count']}, State={track['behavior_state']}",
        "alert_time": "None",
        "dwell_time": f"{track['dwell_time']:.1f}s",
        "meander_ratio": f"{track['meander_ratio']:.2f}",
        "revisit_count": track["revisit_count"],
        "restricted_zone": "NO",
        "final_suspicion_state": f"{track['behavior_state']} (Risk: {severity}, {risk_score}%)",
        "pass": is_pass
    }


def main():
    print("=" * 80)
    print("SENTINEL AI - CONTROLLED 60-SECOND BEHAVIORAL VALIDATION SUITE")
    print("=" * 80)
    print(f"Simulation FPS:            {FPS}")
    print(f"Dwell Threshold:           {SentinelConfig.DWELL_TIME_THRESHOLD}s")
    print(f"Meander Ratio Threshold:   {SentinelConfig.MEANDER_RATIO_THRESHOLD}")
    print(f"Revisit Count Threshold:   {SentinelConfig.REVISIT_COUNT_THRESHOLD}")
    print(f"Temporal Persistence:      {SentinelConfig.SUSPICION_PERSISTENCE_FRAMES} frames ({SentinelConfig.SUSPICION_PERSISTENCE_FRAMES / FPS:.2f}s)")
    print("=" * 80)
    
    results = [
        run_controlled_scenario_1_normal_walking(),
        run_controlled_scenario_2_long_dwell(),
        run_controlled_scenario_3_prolonged_pacing(),
        run_controlled_scenario_4_restricted_pacing(),
        run_controlled_scenario_5_one_turn(),
        run_controlled_scenario_6_single_revisit(),
    ]
    
    print("\n" + "=" * 80)
    print("PHASE 6 - VALIDATION METRICS TABLE")
    print("=" * 80)
    header = f"{'Scenario':<34} | {'Dur':<6} | {'Alert Time':<24} | {'Dwell':<6} | {'Meander':<7} | {'Rev':<3} | {'Zone':<4} | {'Result':<6}"
    print(header)
    print("-" * len(header))
    
    all_passed = True
    for r in results:
        status_str = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            all_passed = False
        print(f"{r['scenario']:<34} | {r['duration']:<6} | {r['alert_time']:<24} | {r['dwell_time']:<6} | {r['meander_ratio']:<7} | {r['revisit_count']:<3} | {r['restricted_zone']:<4} | {status_str:<6}")
        
    print("=" * 80)
    
    print("\nDETAILED SCENARIO REPORTS:")
    for r in results:
        status_label = "[PASS]" if r["pass"] else "[FAIL]"
        print(f"\n{status_label} {r['scenario']}")
        print(f"    Expected:              {r['expected_state']}")
        print(f"    Observed:              {r['observed_state']}")
        print(f"    Alert Trigger:         {r['alert_time']}")
        print(f"    Dwell Time:            {r['dwell_time']}")
        print(f"    Meander Ratio:         {r['meander_ratio']}")
        print(f"    Revisit Count:         {r['revisit_count']}")
        print(f"    Restricted Zone:       {r['restricted_zone']}")
        print(f"    Final Suspicion State: {r['final_suspicion_state']}")
        
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL 6 CONTROLLED BEHAVIORAL SCENARIOS PASSED PERFECTLY (100%)!")
    else:
        print("ONE OR MORE CONTROLLED SCENARIOS FAILED!")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

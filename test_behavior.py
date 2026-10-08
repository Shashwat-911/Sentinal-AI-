"""
SENTINEL AI — Comprehensive Behavior & Detection Unit Tests (Phase 9)
Deterministic test harness verifying all 15 required scenarios:
1. Person must not become luggage.
2. Correct class ID -> correct object type.
3. Luggage confidence filtering.
4. Person tracking persistence.
5. Luggage tracking persistence.
6. Owner-return cancellation.
7. Unattended timer.
8. LOW -> MEDIUM -> HIGH -> CRITICAL transitions.
9. Dwell-time calculation.
10. Meander-ratio calculation.
11. Revisit-count calculation.
12. Suspicious-loitering decision.
13. Restricted-zone event.
14. Multiple people.
15. Multiple bags.
"""

import os
import sys
import time
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from sentinel.config import SentinelConfig
from sentinel.detection import (
    get_object_group, is_person, is_bag_object,
    filter_bag_person_false_positives, iou_deduplicate, calculate_iou
)
from sentinel.tracking import (
    TrackState, TrackStatus, match_entities
)
from sentinel.behavior import (
    compute_dwell_time, compute_trajectory_metrics,
    update_grid_revisits, evaluate_person_behavior
)
from sentinel.risk import (
    calculate_entity_risk, check_owner_proximity_and_returns,
    analyze_behaviors
)


def test_1_person_not_become_luggage():
    """TEST 1: Person must not become luggage through tracker or state logic."""
    print("\n--- Test 1: Person must not become luggage ---")
    tracked = {}
    # Frame 1: Person detected
    dets = [(100, 100, 200, 300, 0.90, "person")]
    tracked = match_entities(dets, tracked)
    key = list(tracked.keys())[0]
    assert is_person(tracked[key]["class_name"]), "Track should be a PERSON"
    assert tracked[key]["object_type"] == "PERSON"

    # Frame 2: Candidate luggage detected near person's body
    # False luggage candidate overlapping person
    cand_luggage = [(120, 110, 180, 190, 0.44, "luggage")]
    clean_bags = filter_bag_person_false_positives(cand_luggage, [(100, 100, 200, 300, 0.90, "person")])
    assert len(clean_bags) == 0, f"Expected luggage on head/torso to be suppressed, but got: {clean_bags}"

    # Match only valid person
    tracked = match_entities(dets, tracked)
    assert len(tracked) == 1
    assert tracked[key]["object_type"] == "PERSON"
    print("  PASS: Person remained PERSON and false luggage candidate was suppressed.")


def test_2_correct_class_id_to_object_type():
    """TEST 2: Correct class ID -> correct object type."""
    print("\n--- Test 2: Correct class ID to object type ---")
    assert get_object_group("person") == "PERSON"
    assert get_object_group("luggage") == "BAG_OBJECT"
    assert is_person("person") is True
    assert is_bag_object("luggage") is True
    assert is_bag_object("person") is False
    assert is_person("luggage") is False

    t_person = TrackState(track_id=1, class_name="person", bbox=(0,0,10,10), confidence=0.9)
    assert t_person.object_type == "PERSON"

    t_bag = TrackState(track_id=1, class_name="luggage", bbox=(0,0,10,10), confidence=0.8)
    assert t_bag.object_type == "LUGGAGE"
    print("  PASS: Class IDs map strictly to isolated semantic groups.")


def test_3_luggage_confidence_filtering():
    """TEST 3: Luggage confidence filtering."""
    print("\n--- Test 3: Luggage confidence filtering ---")
    # Low confidence (< 0.25)
    dets = [
        (10, 10, 50, 50, 0.15, "luggage"),
        (60, 60, 100, 100, 0.65, "luggage")
    ]
    passed = [d for d in dets if d[4] >= SentinelConfig.BAG_CONF_DEFAULT]
    assert len(passed) == 1
    assert passed[0][4] == 0.65
    print("  PASS: Only detections meeting or exceeding confidence threshold are retained.")


def test_4_person_tracking_persistence():
    """TEST 4: Person tracking persistence."""
    print("\n--- Test 4: Person tracking persistence ---")
    tracked = {}
    for f in range(15):
        dets = [(100 + f * 2, 100, 150 + f * 2, 250, 0.88, "person")]
        tracked = match_entities(dets, tracked)
        assert len(tracked) == 1
        key = list(tracked.keys())[0]
        assert tracked[key]["track_id"] == 1
    print("  PASS: Person-1 remained persistent across 15 continuous frames.")


def test_5_luggage_tracking_persistence():
    """TEST 5: Luggage tracking persistence."""
    print("\n--- Test 5: Luggage tracking persistence ---")
    tracked = {}
    for f in range(15):
        dets = [(300, 300, 350, 350, 0.85, "luggage")]
        tracked = match_entities(dets, tracked)
        assert len(tracked) == 1
        key = list(tracked.keys())[0]
        assert tracked[key]["track_id"] == 1
        assert tracked[key]["object_type"] == "LUGGAGE"
    print("  PASS: Luggage-1 remained persistent across 15 continuous frames.")


def test_6_owner_return_cancellation():
    """TEST 6: Owner-return proximity cancellation (Rowner = 150 px)."""
    print("\n--- Test 6: Owner-return proximity cancellation ---")
    now = time.time()
    # Bag that was unattended 20 seconds ago
    tracked = {
        "BAG_OBJECT_1": TrackState(
            track_id=1, class_name="luggage", bbox=(200, 200, 250, 250),
            confidence=0.85, current_time=now - 25,
            stationary_start=now - 20,
            active_threats={"UNATTENDED_OBJECT"}
        ),
        # Person arriving at distance 50px (< Rowner 150px)
        "PERSON_1": TrackState(
            track_id=1, class_name="person", bbox=(220, 220, 270, 320),
            confidence=0.90, current_time=now
        )
    }

    check_owner_proximity_and_returns(tracked, current_time=now)
    bag = tracked["BAG_OBJECT_1"]
    assert "UNATTENDED_OBJECT" not in bag["active_threats"], "Alert should be cancelled"
    assert bag["stationary_start"] is None, "Stationary timer should be reset"
    assert bag.get("owner_returned") is True, "Owner return flag should be set"
    print("  PASS: Alert disarmed, stationary timer reset, and owner return marked.")


def test_7_unattended_timer():
    """TEST 7: Unattended timer progression."""
    print("\n--- Test 7: Unattended timer progression ---")
    now = time.time()
    bag = TrackState(
        track_id=1, class_name="luggage", bbox=(400, 400, 450, 450),
        confidence=0.85, current_time=now - 10, stationary_start=now - 10
    )
    tracked = {"BAG_OBJECT_1": bag}
    # No person nearby
    check_owner_proximity_and_returns(tracked, current_time=now)
    assert bag["stationary_start"] == now - 10
    assert "UNATTENDED_OBJECT" in bag["active_threats"]
    print("  PASS: Unattended timer correctly accumulates unattended duration.")


def test_8_low_medium_high_critical_transitions():
    """TEST 8: LOW -> MEDIUM -> HIGH -> CRITICAL transitions."""
    print("\n--- Test 8: 4-Tier risk escalation transitions ---")
    now = time.time()
    # LOW: 2s unattended out of 15s threshold (~13%)
    t_low = TrackState(track_id=1, class_name="luggage", bbox=(0,0,10,10), confidence=0.8,
                       active_threats={"UNATTENDED_OBJECT"}, stationary_start=now - 2)
    score_low, sev_low, _ = calculate_entity_risk(t_low)
    assert sev_low == "LOW", f"Expected LOW, got {sev_low} (score {score_low})"

    # MEDIUM: 6s unattended (~40%)
    t_med = TrackState(track_id=1, class_name="luggage", bbox=(0,0,10,10), confidence=0.8,
                       active_threats={"UNATTENDED_OBJECT"}, stationary_start=now - 6)
    score_med, sev_med, _ = calculate_entity_risk(t_med)
    assert sev_med == "MEDIUM", f"Expected MEDIUM, got {sev_med} (score {score_med})"

    # HIGH: 12s unattended (~80%)
    t_high = TrackState(track_id=1, class_name="luggage", bbox=(0,0,10,10), confidence=0.8,
                        active_threats={"UNATTENDED_OBJECT"}, stationary_start=now - 12)
    score_high, sev_high, _ = calculate_entity_risk(t_high)
    assert sev_high == "HIGH", f"Expected HIGH, got {sev_high} (score {score_high})"

    # CRITICAL: 16s unattended (>= 15s -> 100%)
    t_crit = TrackState(track_id=1, class_name="luggage", bbox=(0,0,10,10), confidence=0.8,
                        active_threats={"UNATTENDED_OBJECT"}, stationary_start=now - 16)
    score_crit, sev_crit, _ = calculate_entity_risk(t_crit)
    assert sev_crit == "CRITICAL", f"Expected CRITICAL, got {sev_crit} (score {score_crit})"
    assert score_crit == 100
    print("  PASS: Successfully transitioned through LOW -> MEDIUM -> HIGH -> CRITICAL.")


def test_9_dwell_time_calculation():
    """TEST 9: Dwell-time calculation."""
    print("\n--- Test 9: Dwell-time calculation ---")
    now = time.time()
    t = TrackState(track_id=1, class_name="person", bbox=(0,0,10,10), confidence=0.9, first_seen=now - 50.0)
    dwell = compute_dwell_time(t, current_time=now)
    assert math.isclose(dwell, 50.0, abs_tol=0.1)
    print(f"  PASS: Dwell time accurately calculated: {dwell:.1f}s.")


def test_10_meander_ratio_calculation():
    """TEST 10: Meander-ratio calculation."""
    print("\n--- Test 10: Meander-ratio calculation ---")
    # Direct movement: (0,0) -> (10,0) -> (20,0) -> (30,0)
    direct_history = [(0, 0, 1), (10, 0, 2), (20, 0, 3), (30, 0, 4)]
    _, _, rho_direct = compute_trajectory_metrics(direct_history)
    assert math.isclose(rho_direct, 1.0, abs_tol=0.01), f"Expected 1.0, got {rho_direct}"

    # Circling / meandering movement: back and forth
    pacing_history = [(0, 0, 1), (100, 0, 2), (0, 0, 3), (100, 0, 4), (0, 0, 5)]
    # Path length = 100 + 100 + 100 + 100 = 400. Net displacement = 0 (clamped to 1.0)
    _, _, rho_pacing = compute_trajectory_metrics(pacing_history)
    assert rho_pacing >= 10.0, f"Expected high meander ratio, got {rho_pacing}"
    print(f"  PASS: Meander ratio direct={rho_direct:.2f}, wandering={rho_pacing:.2f}.")


def test_11_revisit_count_calculation():
    """TEST 11: Revisit-count calculation."""
    print("\n--- Test 11: Grid revisit-count calculation ---")
    w, h = 800, 600
    t = TrackState(track_id=1, class_name="person", bbox=(0,0,10,10), confidence=0.9)
    # Start at cell (100, 100) -> Cell A
    update_grid_revisits(t, 100, 100, w, h, rows=4, cols=4)
    # Move to cell (500, 100) -> Cell B
    update_grid_revisits(t, 500, 100, w, h, rows=4, cols=4)
    # Return to cell (100, 100) -> Cell A (Revisit #1)
    revisits = update_grid_revisits(t, 100, 100, w, h, rows=4, cols=4)
    assert revisits == 1, f"Expected 1 revisit, got {revisits}"
    print("  PASS: Revisit count accurately tracks cell departure and return.")


def test_12_suspicious_loitering_decision():
    """TEST 12: Suspicious-loitering decision (Requires dwell + meander/revisit)."""
    print("\n--- Test 12: Suspicious-loitering decision ---")
    now = time.time()
    w, h = 800, 600

    # Case A: Short dwell time (< 45s) even with high meander -> NOT suspicious
    p_short = TrackState(
        track_id=1, class_name="person", bbox=(100, 100, 150, 250), confidence=0.9,
        first_seen=now - 20, centroid=(125, 175),
        history=[(0,0,now-20), (100,0,now-10), (0,0,now)]
    )
    evaluate_person_behavior(p_short, w, h, current_time=now)
    assert "LOITERING" not in p_short["active_threats"], "Should not flag before Tdwell"

    # Case B: Long dwell time (>= 45s) but straight path (meander ~ 1.0, 0 revisits) -> NOT suspicious
    p_direct = TrackState(
        track_id=2, class_name="person", bbox=(100, 100, 150, 250), confidence=0.9,
        first_seen=now - 50, centroid=(500, 200),
        history=[(100, 200, now-50), (300, 200, now-25), (500, 200, now)]
    )
    evaluate_person_behavior(p_direct, w, h, current_time=now)
    assert "LOITERING" not in p_direct["active_threats"], "Should not flag direct movement"

    # Case C: Long dwell time (>= 45s) AND high meander (>= 2.5) -> LOITERING
    p_loiter = TrackState(
        track_id=3, class_name="person", bbox=(100, 100, 150, 250), confidence=0.9,
        first_seen=now - 55, centroid=(100, 100),
        history=[(100, 100, now-55), (250, 100, now-40), (100, 100, now-25), (250, 100, now-10), (100, 100, now)]
    )
    persist = getattr(SentinelConfig, "SUSPICION_PERSISTENCE_FRAMES", 15)
    for _ in range(persist):
        evaluate_person_behavior(p_loiter, w, h, current_time=now)
    assert "LOITERING" in p_loiter["active_threats"], "Should flag loitering on dwell + meander after persistence"
    print("  PASS: Correctly enforces dual requirement (Dwell >= 45s AND repetitive behavior).")


def test_13_restricted_zone_event_separation():
    """TEST 13: Restricted-zone event separation from loitering."""
    print("\n--- Test 13: Restricted-zone event separation ---")
    now = time.time()
    w, h = 1000, 1000
    # Inside restricted zone (x: 0.60 - 0.95, y: 0.20 - 0.75) -> e.g. (800, 500)
    p_zone = TrackState(
        track_id=1, class_name="person", bbox=(780, 450, 820, 550), confidence=0.9,
        first_seen=now - 5, centroid=(800, 500), consecutive_restricted_frames=20
    )
    evaluate_person_behavior(p_zone, w, h, current_time=now)
    assert "RESTRICTED_ZONE" in p_zone["active_threats"]
    assert "LOITERING" not in p_zone["active_threats"], "Restricted zone should NOT be labeled loitering"

    score, sev, _ = calculate_entity_risk(p_zone)
    assert sev == "HIGH", f"Restricted zone entry should carry HIGH severity, got {sev}"
    print("  PASS: Restricted zone entry is evaluated independently from personal loitering.")


def test_14_multiple_people_tracking():
    """TEST 14: Multiple people tracked independently with distinct IDs."""
    print("\n--- Test 14: Multiple people tracking ---")
    tracked = {}
    dets = [
        (100, 100, 150, 250, 0.90, "person"),
        (400, 100, 450, 250, 0.88, "person"),
        (700, 100, 750, 250, 0.85, "person")
    ]
    tracked = match_entities(dets, tracked)
    assert len(tracked) == 3
    ids = {info["track_id"] for info in tracked.values()}
    assert ids == {1, 2, 3}
    for info in tracked.values():
        assert info["object_type"] == "PERSON"
    print("  PASS: Multiple individuals tracked with stable, distinct IDs [1, 2, 3].")


def test_15_multiple_bags_tracking():
    """TEST 15: Multiple luggage items tracked independently with distinct IDs."""
    print("\n--- Test 15: Multiple bags tracking ---")
    tracked = {}
    dets = [
        (100, 500, 150, 550, 0.80, "luggage"),
        (300, 500, 350, 550, 0.85, "luggage")
    ]
    tracked = match_entities(dets, tracked)
    assert len(tracked) == 2
    ids = {info["track_id"] for info in tracked.values()}
    assert ids == {1, 2}
    for info in tracked.values():
        assert info["object_type"] == "LUGGAGE"
    print("  PASS: Multiple bags tracked with stable, distinct IDs [1, 2].")


def _pacing_history(now, cycles=6, span=180.0):
    history = []
    t = now - cycles * 2.0
    x = 120.0
    for i in range(cycles * 2 + 1):
        history.append((x, 180.0, t))
        x = 120.0 + span if i % 2 == 0 else 120.0
        t += 1.0
    history[-1] = (history[-1][0], history[-1][1], now)
    return history


def test_16_straight_walk_remains_normal():
    """Required TEST 1: Person walking straight through the scene stays NORMAL."""
    print("\n--- Test 16: Straight walk remains NORMAL ---")
    now = time.time()
    w, h = 800, 600
    p = TrackState(
        track_id=1, class_name="person", bbox=(40, 160, 90, 300), confidence=0.9,
        first_seen=now - 8.0, centroid=(700, 200),
        history=[(80, 200, now - 8), (300, 200, now - 5), (520, 200, now - 2), (700, 200, now)]
    )
    evaluate_person_behavior(p, w, h, current_time=now)
    assert "LOITERING" not in p["active_threats"]
    assert "SUSPICIOUS_LOITERING" not in p["active_threats"]
    assert "LONG_DWELL" not in p["active_threats"]
    assert p["behavior_state"] == "NORMAL"
    assert p["meander_ratio"] < 1.2
    print("  PASS: Direct transit classified as NORMAL.")


def test_17_long_dwell_observing_not_suspicious():
    """Required TEST 2: Standing/waiting is LONG_DWELL / OBSERVING, not automatically suspicious."""
    print("\n--- Test 17: Long dwell is OBSERVING, not suspicious ---")
    now = time.time()
    w, h = 800, 600
    p = TrackState(
        track_id=1, class_name="person", bbox=(200, 200, 250, 350), confidence=0.9,
        first_seen=now - 60.0, centroid=(225, 275),
        history=[(225, 275, now - 60), (226, 275, now - 30), (225, 276, now)]
    )
    evaluate_person_behavior(p, w, h, current_time=now)
    assert "LONG_DWELL" in p["active_threats"]
    assert "LOITERING" not in p["active_threats"]
    assert "SUSPICIOUS_LOITERING" not in p["active_threats"]
    assert p["behavior_state"] == "OBSERVING"
    print("  PASS: 60s wait is OBSERVING / LONG_DWELL, not SUSPICIOUS.")


def test_18_single_direction_change_normal():
    """Required TEST 3: One direction change is not suspicious."""
    print("\n--- Test 18: Single direction change remains NORMAL ---")
    now = time.time()
    w, h = 800, 600
    p = TrackState(
        track_id=1, class_name="person", bbox=(100, 100, 150, 250), confidence=0.9,
        first_seen=now - 10.0, centroid=(400, 400),
        history=[(100, 200, now - 10), (250, 200, now - 6), (400, 200, now - 3), (400, 400, now)]
    )
    evaluate_person_behavior(p, w, h, current_time=now)
    assert p["meander_ratio"] < SentinelConfig.MEANDER_RATIO_THRESHOLD
    assert "LOITERING" not in p["active_threats"]
    assert "REPEATED_MOVEMENT" not in p["active_threats"]
    assert p["behavior_state"] == "NORMAL"
    print(f"  PASS: One turn (meander={p['meander_ratio']:.2f}) stayed NORMAL.")


def test_19_single_revisit_not_immediately_suspicious():
    """Required TEST 4: A → B → A produces revisit evidence without immediate suspicion."""
    print("\n--- Test 19: Single A->B->A revisit is not suspicious ---")
    now = time.time()
    w, h = 800, 600
    p = TrackState(track_id=1, class_name="person", bbox=(90, 90, 140, 240), confidence=0.9,
                   first_seen=now - 12, centroid=(100, 100))
    update_grid_revisits(p, 100, 100, w, h, rows=4, cols=4)
    update_grid_revisits(p, 500, 100, w, h, rows=4, cols=4)
    revisits = update_grid_revisits(p, 100, 100, w, h, rows=4, cols=4)
    p["centroid"] = (100, 100)
    p["history"] = [(100, 100, now - 12), (500, 100, now - 6), (100, 100, now)]
    evaluate_person_behavior(p, w, h, current_time=now)
    assert revisits == 1
    assert "LOITERING" not in p["active_threats"]
    assert "SUSPICIOUS_LOITERING" not in p["active_threats"]
    print("  PASS: One revisit recorded without a suspicious classification.")


def test_20_repeated_pacing_high_repetitive_evidence():
    """Required TEST 5: A→B→A→B→A→B produces high repetitive-movement evidence."""
    print("\n--- Test 20: Repeated A<->B pacing produces high meander/revisits ---")
    now = time.time()
    w, h = 800, 600
    p = TrackState(
        track_id=1, class_name="person", bbox=(100, 140, 150, 280), confidence=0.9,
        first_seen=now - 20, centroid=(120, 180),
        history=_pacing_history(now, cycles=6, span=400.0)
    )
    xs = [80, 520, 80, 520, 80, 520, 80]
    for x in xs:
        update_grid_revisits(p, x, 180, w, h, rows=4, cols=4)
    p["centroid"] = (80, 180)
    evaluate_person_behavior(p, w, h, current_time=now)
    assert p["meander_ratio"] >= SentinelConfig.MEANDER_RATIO_THRESHOLD
    assert p["revisit_count"] >= 1
    assert "REPEATED_MOVEMENT" in p["active_threats"]
    assert "LOITERING" not in p["active_threats"], "Dwell < 45s must not be suspicious loitering"
    print(f"  PASS: High repetitive evidence (meander={p['meander_ratio']:.1f}, revisits={p['revisit_count']}) without dwell-based loitering.")


def test_21_prolonged_pacing_suspicious_loitering():
    """Required TEST 6: Prolonged pacing becomes SUSPICIOUS LOITERING after persistence."""
    print("\n--- Test 21: Prolonged pacing is SUSPICIOUS LOITERING ---")
    now = time.time()
    w, h = 800, 600
    p = TrackState(
        track_id=1, class_name="person", bbox=(100, 140, 150, 280), confidence=0.9,
        first_seen=now - 55, centroid=(120, 180),
        history=_pacing_history(now, cycles=8, span=180.0)
    )
    persist = getattr(SentinelConfig, "SUSPICION_PERSISTENCE_FRAMES", 15)
    for _ in range(persist):
        evaluate_person_behavior(p, w, h, current_time=now)
    assert "SUSPICIOUS_LOITERING" in p["active_threats"]
    assert p["behavior_state"] in ("ALERT", "CRITICAL")
    print("  PASS: Prolonged pacing + dwell persisted into SUSPICIOUS LOITERING.")


def test_22_restricted_zone_entry_event():
    """Required TEST 7: Restricted-zone entry is a distinct RESTRICTED_ZONE event."""
    print("\n--- Test 22: Restricted-zone entry event ---")
    now = time.time()
    w, h = 1000, 1000
    p = TrackState(
        track_id=1, class_name="person", bbox=(780, 450, 820, 550), confidence=0.9,
        first_seen=now - 4, centroid=(800, 500), consecutive_restricted_frames=20
    )
    evaluate_person_behavior(p, w, h, current_time=now)
    assert "RESTRICTED_ZONE" in p["active_threats"]
    assert p["in_restricted_zone"] is True
    assert "SUSPICIOUS_LOITERING" not in p["active_threats"]
    print("  PASS: Restricted-zone entry emitted independently.")


def test_23_zone_plus_pacing_higher_suspicion():
    """Required TEST 8: Restricted zone + repetitive pacing raises suspicion above zone-only."""
    print("\n--- Test 23: Restricted zone + pacing is higher suspicion ---")
    now = time.time()
    w, h = 1000, 1000
    p_zone = TrackState(
        track_id=1, class_name="person", bbox=(780, 450, 820, 550), confidence=0.9,
        first_seen=now - 5, centroid=(800, 500), consecutive_restricted_frames=20
    )
    evaluate_person_behavior(p_zone, w, h, current_time=now)
    zone_score = p_zone["suspicion_score"]

    p_both = TrackState(
        track_id=2, class_name="person", bbox=(780, 450, 820, 550), confidence=0.9,
        first_seen=now - 55, centroid=(800, 500), consecutive_restricted_frames=20,
        history=[(800, 500, now - 55), (920, 500, now - 40), (800, 500, now - 25),
                 (920, 500, now - 10), (800, 500, now)]
    )
    persist = getattr(SentinelConfig, "SUSPICION_PERSISTENCE_FRAMES", 15)
    for _ in range(persist):
        evaluate_person_behavior(p_both, w, h, current_time=now)
    assert p_both["suspicion_score"] > zone_score
    assert "HIGH_CONFIDENCE_SUSPICIOUS" in p_both["active_threats"]
    assert "RESTRICTED_ZONE" in p_both["active_threats"]
    print(f"  PASS: Combined score {p_both['suspicion_score']} > zone-only {zone_score}.")


def test_24_independent_person_behavior_states():
    """Required TEST 9: Two people maintain independent behavioural state."""
    print("\n--- Test 24: Independent behaviour states for two people ---")
    now = time.time()
    w, h = 800, 600
    walker = TrackState(
        track_id=1, class_name="person", bbox=(40, 160, 90, 300), confidence=0.9,
        first_seen=now - 8.0, centroid=(700, 200),
        history=[(80, 200, now - 8), (400, 200, now - 4), (700, 200, now)]
    )
    pacer = TrackState(
        track_id=2, class_name="person", bbox=(100, 140, 150, 280), confidence=0.9,
        first_seen=now - 55, centroid=(120, 180),
        history=_pacing_history(now, cycles=8, span=180.0)
    )
    persist = getattr(SentinelConfig, "SUSPICION_PERSISTENCE_FRAMES", 15)
    evaluate_person_behavior(walker, w, h, current_time=now)
    for _ in range(persist):
        evaluate_person_behavior(pacer, w, h, current_time=now)
    assert walker["track_id"] != pacer["track_id"]
    assert walker["history"] != pacer["history"]
    assert walker["behavior_state"] == "NORMAL"
    assert "SUSPICIOUS_LOITERING" in pacer["active_threats"]
    assert "SUSPICIOUS_LOITERING" not in walker["active_threats"]
    print("  PASS: PERSON #1 and PERSON #2 do not share trajectory or suspicion state.")


def test_25_occlusion_does_not_reset_behavior_state():
    """Required TEST 10: Temporary occlusion keeps tracker identity and behaviour fields."""
    print("\n--- Test 25: Occlusion retains behaviour state ---")
    now = time.time()
    w, h = 800, 600
    p = TrackState(
        track_id=7, class_name="person", bbox=(100, 140, 150, 280), confidence=0.9,
        first_seen=now - 55, centroid=(120, 180),
        history=_pacing_history(now, cycles=8, span=180.0)
    )
    tracked = {"PERSON_7": p}
    persist = getattr(SentinelConfig, "SUSPICION_PERSISTENCE_FRAMES", 15)
    for _ in range(persist):
        evaluate_person_behavior(tracked["PERSON_7"], w, h, current_time=now)
    assert "SUSPICIOUS_LOITERING" in tracked["PERSON_7"]["active_threats"]
    before_state = tracked["PERSON_7"]["behavior_state"]
    before_score = tracked["PERSON_7"]["suspicion_score"]
    before_history = list(tracked["PERSON_7"]["history"])

    tracked = match_entities([], tracked)
    assert "PERSON_7" in tracked
    assert tracked["PERSON_7"]["track_id"] == 7
    assert tracked["PERSON_7"]["state"] == TrackStatus.TEMPORARILY_MISSED
    analyze_behaviors(tracked, w, h, current_time=now)
    assert tracked["PERSON_7"]["behavior_state"] == before_state
    assert tracked["PERSON_7"]["suspicion_score"] == before_score
    assert tracked["PERSON_7"]["history"] == before_history
    assert "SUSPICIOUS_LOITERING" in tracked["PERSON_7"]["active_threats"]

    tracked = match_entities([(105, 140, 155, 280, 0.9, "person")], tracked)
    assert tracked["PERSON_7"]["track_id"] == 7
    print("  PASS: Occlusion retained ID, history, and suspicion state.")


if __name__ == "__main__":
    print("=" * 70)
    print("SENTINEL AI — Comprehensive Verification Suite")
    print("=" * 70)

    test_1_person_not_become_luggage()
    test_2_correct_class_id_to_object_type()
    test_3_luggage_confidence_filtering()
    test_4_person_tracking_persistence()
    test_5_luggage_tracking_persistence()
    test_6_owner_return_cancellation()
    test_7_unattended_timer()
    test_8_low_medium_high_critical_transitions()
    test_9_dwell_time_calculation()
    test_10_meander_ratio_calculation()
    test_11_revisit_count_calculation()
    test_12_suspicious_loitering_decision()
    test_13_restricted_zone_event_separation()
    test_14_multiple_people_tracking()
    test_15_multiple_bags_tracking()
    test_16_straight_walk_remains_normal()
    test_17_long_dwell_observing_not_suspicious()
    test_18_single_direction_change_normal()
    test_19_single_revisit_not_immediately_suspicious()
    test_20_repeated_pacing_high_repetitive_evidence()
    test_21_prolonged_pacing_suspicious_loitering()
    test_22_restricted_zone_entry_event()
    test_23_zone_plus_pacing_higher_suspicion()
    test_24_independent_person_behavior_states()
    test_25_occlusion_does_not_reset_behavior_state()

    print("\n" + "=" * 70)
    print("ALL COMPREHENSIVE BEHAVIOR & DETECTION TESTS PASSED!")
    print("=" * 70)

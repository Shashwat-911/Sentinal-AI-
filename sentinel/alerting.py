"""
SENTINEL AI — Alerting & HUD Visualization Module
Renders cyber-HUD overlays with:
- Strict visual distinction between PERSON and LUGGAGE
- Clear behavioral indicators (dwell time, loitering, owner returned)
- Cooldown-managed security alert dispatching.
"""

import time
import cv2
import numpy as np
from sentinel.config import SentinelConfig
from sentinel.detection import is_person, is_bag_object
from sentinel.risk import calculate_entity_risk, analyze_behaviors


def draw_overlays(frame, tracked_entities, frame_count=0, current_time=None):
    """
    Draw bounding boxes, threat overlays, and Restricted Zone.
    Returns (annotated_frame, list_of_new_alerts).
    
    Args:
        current_time: Optional. If None, uses wall-clock time (for real-time camera).
                      If provided, uses the given time (for video-time testing).
    """
    h, w = frame.shape[:2]
    if current_time is None:
        current_time = time.time()
    new_alerts = []

    # Run behavior and risk analysis
    analyze_behaviors(tracked_entities, w, h, current_time)

    # 1. Draw Restricted Zone Polygon
    if SentinelConfig.RESTRICTED_ZONE_ENABLED:
        poly_pts = np.array([
            [int(pt[0] * w), int(pt[1] * h)] for pt in SentinelConfig.RESTRICTED_ZONE_COORDS
        ], dtype=np.int32)

        zone_violated = any("RESTRICTED_ZONE" in info["active_threats"]
                            for info in tracked_entities.values()
                            if is_person(info["class_name"]))
        zone_color = (0, 0, 255) if zone_violated else (255, 200, 0)

        cv2.polylines(frame, [poly_pts], True, zone_color, 2)
        cv2.putText(frame, "RESTRICTED SURVEILLANCE ZONE", (poly_pts[0][0] + 5, max(20, poly_pts[0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone_color, 1)

    # 2. Draw Bounding Boxes and Status Badges
    for key, info in tracked_entities.items():
        if info.get("alert_dismissed", False) or info.get("missed_frames", 0) > 5:
            continue

        x1, y1, x2, y2 = info["bbox"]
        class_name = info["class_name"]
        track_id = info["track_id"]
        conf = int(info["confidence"] * 100)
        is_human = is_person(class_name)

        risk_score, severity, color = calculate_entity_risk(info, current_time)
        info["risk_score"] = risk_score
        info["severity"] = severity

        # Flashing for critical security threats
        if severity == "CRITICAL" and (frame_count // 8) % 2 == 0:
            color = (255, 255, 255)

        # Base box color by entity type if NORMAL
        if severity == "NORMAL":
            color = (0, 255, 136) if is_human else (255, 170, 0)

        # Draw main bounding box
        thickness = 3 if severity in ("HIGH", "CRITICAL") else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Build clean HUD label
        if is_human:
            dwell = int(info.get("dwell_time", current_time - info.get("first_seen", current_time)))
            meander = info.get("meander_ratio", 1.0)
            revisits = info.get("revisit_count", 0)
            in_zone = info.get("in_restricted_zone", False)
            b_state = info.get("behavior_state", severity)
            zone_str = "RESTRICTED" if in_zone else "NORMAL"

            behaviour_label = {
                "ALERT": "SUSPICIOUS",
                "CRITICAL": "SUSPICIOUS",
                "HIGH": "SUSPICIOUS",
                "COOLING_DOWN": "COOLING",
            }.get(str(b_state), str(b_state))
            label = f"PERSON #{track_id} | {behaviour_label} | Risk: {severity}"
            sub_label = (
                f"Dwell: {dwell}s | Meander: {meander:.1f} | Revisits: {revisits} | "
                f"Zone: {zone_str}"
            )
        else:
            if info.get("owner_returned"):
                label = f"LUGGAGE #{track_id} | OWNER RETURNED"
                sub_label = None
                color = (255, 255, 0)  # Cyan
            else:
                label = f"LUGGAGE #{track_id} | {conf}% | {severity} ({risk_score}%)"
                sub_label = None

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, max(th + 8, y1)), color, -1)
        cv2.putText(frame, label, (x1 + 3, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)

        # Telemetry badge for persons
        badge_y = y2
        if is_human and sub_label:
            (stw, sth), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            cv2.rectangle(frame, (x1, badge_y), (x1 + stw + 6, badge_y + sth + 6), (20, 20, 20), -1)
            cv2.putText(frame, sub_label, (x1 + 3, badge_y + sth + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)
            badge_y += sth + 8

        # Render active threats badge below bounding box
        threats = info.get("active_threats", set())
        if threats:
            threat_display_map = {
                "RESTRICTED_ZONE": "⚠ RESTRICTED ZONE ENTRY",
                "LONG_DWELL": "⚠ LONG DWELL",
                "REPEATED_MOVEMENT": "⚠ REPEATED MOVEMENT",
                "SUSPICIOUS_LOITERING": "⚠ SUSPICIOUS LOITERING",
                "LOITERING": "⚠ LOITERING",
                "HIGH_CONFIDENCE_SUSPICIOUS": "🚨 HIGH-CONFIDENCE SUSPICIOUS BEHAVIOUR",
                "SUDDEN_RUNNING": "⚠ SUDDEN RUNNING",
                "UNATTENDED_OBJECT": "🚨 UNATTENDED BAGGAGE" if severity != "CRITICAL" else "🚨 CRITICAL BAGGAGE",
            }
            # Avoid displaying redundant aliases
            display_threats = [t for t in threats if not (t == "LOITERING" and "SUSPICIOUS_LOITERING" in threats)]
            threat_str = " | ".join(threat_display_map.get(t, t) for t in display_threats)
            (ttw, tth), _ = cv2.getTextSize(threat_str, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(frame, (x1, badge_y), (x1 + ttw + 6, badge_y + tth + 8), (0, 0, 0), -1)
            cv2.putText(frame, threat_str, (x1 + 3, badge_y + tth + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Fire alerts with per-behaviour cooldown (zone vs loiter stay independent)
            last_alert_times = info.setdefault("last_alert_times", {})
            for t in threats:
                if t == "LOITERING" and "SUSPICIOUS_LOITERING" in threats:
                    continue  # suppress duplicate alias alert

                last_alert = last_alert_times.get(t, info.get("last_alert_time", 0.0))
                if current_time - last_alert >= SentinelConfig.ALERT_COOLDOWN:
                    last_alert_times[t] = current_time
                    info["last_alert_time"] = current_time
                    event_name = threat_display_map.get(t, t)
                    evidence_desc = (
                        f"Dwell={dwell}s, Meander={meander:.1f}, Revisits={revisits}, Zone={zone_str}"
                        if is_human else ""
                    )
                    new_alerts.append({
                        "key": key,
                        "track_id": track_id,
                        "class_name": class_name,
                        "behavior": t,
                        "event_name": event_name,
                        "severity": severity,
                        "score": risk_score,
                        "evidence": info.get("suspicion_evidence", {}),
                        "description": (
                            f"{'PERSON' if is_human else 'LUGGAGE'} #{track_id} — {event_name} ({severity})"
                            + (f" [{evidence_desc}]" if evidence_desc else "")
                        )
                    })

    # 3. Draw Flashing Alarm Banner for Active Threats
    active_threat_items = [
        info for info in tracked_entities.values()
        if info.get("active_threats") and not info.get("alert_dismissed")
    ]
    if active_threat_items:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (10, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        banner_text = " \u26A0 ALERT: " + " | ".join(
            f"{'PERSON' if is_person(i['class_name']) else 'LUGGAGE'}#{i['track_id']} ({','.join(t for t in i['active_threats'] if t != 'LOITERING') or 'ALERT'})"
            for i in active_threat_items[:2]
        )[:70] + " — SECURITY ATTENTION REQUIRED"
        cv2.putText(frame, banner_text, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 255), 2)
    return frame, new_alerts

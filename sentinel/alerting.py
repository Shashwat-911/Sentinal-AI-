"""
SENTINEL AI — Alerting & HUD Visualization Module
Renders person detection, posture, behavior, and incident overlays.
"""

import time
import cv2
import numpy as np
from sentinel.config import SentinelConfig
from sentinel.detection import is_person
from sentinel.risk import calculate_entity_risk, analyze_behaviors


def draw_overlays(frame, tracked_entities, frame_count=0, current_time=None, ai_context=None, run_behavior=True):
    """
    Draw bounding boxes, posture skeletons, threat overlays, and Cyber-HUD.
    Returns (annotated_frame, list_of_new_alerts).
    
    Args:
        current_time: Optional. If None, uses wall-clock time (for real-time camera).
                      If provided, uses the given time (for video-time testing).
        ai_context: Optional dict with latest AI classification ("not_suspicious",
                    "suspicious", "safety_concern", "uncertain") to align HUD with AI.
        run_behavior: Whether to advance behavior analysis on this call.
    """
    h, w = frame.shape[:2]
    if current_time is None:
        current_time = time.time()
    new_alerts = []
    ai_cls = (ai_context or {}).get("classification")

    # Run behavior and risk analysis
    if run_behavior:
        analyze_behaviors(tracked_entities, w, h, current_time)

    # 1. Draw Bounding Boxes, Posture Skeletons, and Status Badges
    for key, info in tracked_entities.items():
        if info.get("alert_dismissed", False) or info.get("missed_frames", 0) > 5:
            continue

        x1, y1, x2, y2 = info["bbox"]
        class_name = info["class_name"]
        track_id = info["track_id"]
        is_human = is_person(class_name)
        if not is_human:
            continue

        # Draw posture skeleton for persons when available
        if is_human and info.get("keypoints") is not None:
            from sentinel.posture import draw_pose_skeleton
            draw_pose_skeleton(frame, info["keypoints"], info.get("posture", "UNKNOWN"))

        risk_score, severity, color = calculate_entity_risk(info, current_time, ai_context=ai_context)
        info["risk_score"] = risk_score
        info["severity"] = severity

        threats = info.get("active_threats", set())
        has_fall = "POSSIBLE_FALL" in threats
        ai_cleared = (ai_cls == "not_suspicious" and not has_fall)

        # Flashing for critical security threats
        if severity == "CRITICAL" and (frame_count // 8) % 2 == 0:
            color = (255, 255, 255)

        # Base box color for normal person tracks
        if severity == "NORMAL":
            color = (0, 255, 136)

        # Draw main bounding box
        thickness = 3 if severity in ("HIGH", "CRITICAL") else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        dwell = int(info.get("dwell_time", current_time - info.get("first_seen", current_time)))
        meander = info.get("meander_ratio", 1.0)
        revisits = info.get("revisit_count", 0)
        posture = info.get("posture", "UNKNOWN")
        pose_conf = int(info.get("pose_confidence", 0.0) * 100)
        b_state = info.get("behavior_state", severity)

        if ai_cleared:
            behaviour_label = "NORMAL (AI)"
        elif ai_cls == "suspicious":
            behaviour_label = "SUSPICIOUS"
        elif ai_cls == "safety_concern":
            behaviour_label = "SAFETY CONCERN"
        else:
            behaviour_label = {
                "ALERT": "SUSPICIOUS",
                "CRITICAL": "SUSPICIOUS",
                "HIGH": "SUSPICIOUS",
                "COOLING_DOWN": "COOLING",
            }.get(str(b_state), str(b_state))
        label = f"PERSON #{track_id} | {behaviour_label} | Risk: {severity}"
        sub_label = (
            f"Posture: {posture} ({pose_conf}%) | Dwell: {dwell}s | "
            f"Meander: {meander:.1f} | Revisits: {revisits}"
        )

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

        # Render active threats badge below bounding box (suppressed when AI cleared scene as benign)
        effective_threats = ({"POSSIBLE_FALL"} if has_fall else set()) if ai_cleared else set(threats)
        if not effective_threats and ai_cls in ("suspicious", "safety_concern"):
            ai_badge = "⚠ AI: SAFETY CONCERN" if ai_cls == "safety_concern" else "⚠ AI: SUSPICIOUS CONTEXT"
            (ttw, tth), _ = cv2.getTextSize(ai_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(frame, (x1, badge_y), (x1 + ttw + 6, badge_y + tth + 8), (0, 0, 0), -1)
            cv2.putText(frame, ai_badge, (x1 + 3, badge_y + tth + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        elif effective_threats:
            threat_display_map = {
                "POSSIBLE_FALL": "⚠ POSSIBLE FALL / ASSISTANCE REQUIRED",
                "LONG_DWELL": "⚠ LONG DWELL",
                "REPEATED_MOVEMENT": "⚠ REPEATED MOVEMENT",
                "SUSPICIOUS_LOITERING": "⚠ SUSPICIOUS LOITERING",
                "LOITERING": "⚠ LOITERING",
                "HIGH_CONFIDENCE_SUSPICIOUS": "🚨 HIGH-CONFIDENCE SUSPICIOUS BEHAVIOUR",
                "SUDDEN_RUNNING": "⚠ SUDDEN RUNNING",
            }
            # Avoid displaying redundant aliases
            display_threats = [t for t in effective_threats if not (t == "LOITERING" and "SUSPICIOUS_LOITERING" in effective_threats)]
            threat_str = " | ".join(threat_display_map.get(t, t) for t in display_threats)
            (ttw, tth), _ = cv2.getTextSize(threat_str, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(frame, (x1, badge_y), (x1 + ttw + 6, badge_y + tth + 8), (0, 0, 0), -1)
            cv2.putText(frame, threat_str, (x1 + 3, badge_y + tth + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Fire alerts with per-behaviour cooldown
            last_alert_times = info.setdefault("last_alert_times", {})
            for t in effective_threats:
                if t == "LOITERING" and "SUSPICIOUS_LOITERING" in effective_threats:
                    continue  # suppress duplicate alias alert

                last_alert = last_alert_times.get(t, info.get("last_alert_time", 0.0))
                if current_time - last_alert >= SentinelConfig.ALERT_COOLDOWN:
                    last_alert_times[t] = current_time
                    info["last_alert_time"] = current_time
                    event_name = threat_display_map.get(t, t)
                    evidence_desc = (
                        f"Posture={posture}, Dwell={dwell}s, Meander={meander:.1f}, Revisits={revisits}"
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
                            f"PERSON #{track_id} — {event_name} ({severity})"
                            + (f" [{evidence_desc}]" if evidence_desc else "")
                        )
                    })

    # 3. Draw Flashing Alarm Banner for Active Threats (aligned with AI context)
    active_threat_items = [
        info for info in tracked_entities.values()
        if not info.get("alert_dismissed") and info.get("missed_frames", 0) <= 5 and (
            ("POSSIBLE_FALL" in info.get("active_threats", set()))
            or (ai_cls != "not_suspicious" and (info.get("active_threats") or ai_cls in ("suspicious", "safety_concern")))
        )
    ]
    if active_threat_items:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (10, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        parts = []
        for i in active_threat_items[:2]:
            if is_person(i.get("class_name", "")):
                t_labels = [t for t in i.get("active_threats", set()) if t != "LOITERING"]
                if not t_labels and ai_cls in ("suspicious", "safety_concern"):
                    t_labels = [f"AI_{ai_cls.upper()}"]
                parts.append(f"PERSON#{i['track_id']} ({','.join(t_labels) or 'ALERT'})")
        banner_text = (" \u26A0 ALERT: " + " | ".join(parts))[:70] + " — SECURITY ATTENTION REQUIRED"
        cv2.putText(frame, banner_text, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 255), 2)
    return frame, new_alerts

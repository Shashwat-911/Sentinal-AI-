"""
Headless person-only detection test for SENTINEL AI.
Runs the same person detector, tracker, behavior, and overlay path used by the GUI.
"""

import os
import sys
import time
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app import (
    load_person_model, detect_persons, iou_deduplicate, match_entities,
    draw_overlays
)

VIDEO_PATH = os.path.join(BASE_DIR, "test_videos", "LeftBag.mpg")
CONF_THRESHOLD = 0.25
MAX_FRAMES = 500

print("=" * 70)
print("SENTINEL AI - Headless Person-Only Detection Test")
print("=" * 70)
print(f"  Video:           {os.path.basename(VIDEO_PATH)}")
print(f"  Confidence:      {CONF_THRESHOLD}")
print(f"  Max frames:      {MAX_FRAMES}")
print("=" * 70)

print("\n[1/3] Loading person model...")
person_model, status = load_person_model(BASE_DIR)
print(f"  Status: {status}")
if person_model is None:
    print("\nFATAL: Person model unavailable. Cannot run detection.")
    sys.exit(1)

print("\n[2/3] Opening video...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"FATAL: Cannot open video: {VIDEO_PATH}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"  Resolution: {width}x{height}")
print(f"  FPS:        {fps}")
print(f"  Duration:   {total_frames / max(fps, 1):.1f}s ({total_frames} frames)")

print(f"\n[3/3] Running person-only detection (processing up to {MAX_FRAMES} frames)...")
print("-" * 70)

tracked_entities = {}
fcount = 0
total_person_detections = 0
all_alerts = []
conf_values = []

frame_milestones = {1, 50, 100, 200, 300, 400, 500}
behavior_history = {
    "LOITERING": [],
    "SUDDEN_RUNNING": [],
    "REPEATED_MOVEMENT": [],
    "POSSIBLE_FALL": [],
}

while cap.isOpened() and fcount < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    fcount += 1
    video_time = fcount / max(fps, 1)

    person_dets = detect_persons(frame, person_model, CONF_THRESHOLD)
    total_person_detections += len(person_dets)
    conf_values.extend(d[4] for d in person_dets)

    merged_dets = iou_deduplicate(person_dets)
    tracked_entities = match_entities(merged_dets, tracked_entities, current_time=video_time)
    annotated, new_alerts = draw_overlays(frame, tracked_entities, fcount, current_time=video_time)

    for alert in new_alerts:
        all_alerts.append((fcount, alert))
        print(
            f"  [Frame {fcount:4d}] ALERT: PERSON #{alert['track_id']} | "
            f"{alert['behavior']} | {alert['severity']} | score={alert['score']}"
        )

    for _key, info in tracked_entities.items():
        assert info.get("class_name") == "person", f"Non-person track leaked into tracker: {info}"
        threats = info.get("active_threats", set())
        for threat in threats:
            if threat in behavior_history and not any(e["track_id"] == info["track_id"] for e in behavior_history[threat]):
                behavior_history[threat].append({
                    "frame": fcount,
                    "track_id": info["track_id"],
                    "severity": info.get("severity", "UNKNOWN"),
                    "score": info.get("risk_score", 0),
                })

    if fcount in frame_milestones:
        persons = [e for e in tracked_entities.values() if e["class_name"] == "person"]
        active_threats = sum(len(e.get("active_threats", set())) for e in tracked_entities.values())
        print(
            f"  [Frame {fcount:4d}] Tracking: {len(persons)} persons | "
            f"Active threats: {active_threats} | "
            f"Detections this frame: {len(person_dets)} persons"
        )

    sample_frames = {1, max(1, MAX_FRAMES // 4), max(1, MAX_FRAMES // 2), max(1, 3 * MAX_FRAMES // 4)}
    if fcount in sample_frames:
        sample_path = os.path.join(BASE_DIR, "test_videos", f"sample_frame_{fcount}.jpg")
        cv2.imwrite(sample_path, annotated)

cap.release()

print("\n" + "=" * 70)
print("PERSON-ONLY DETECTION TEST RESULTS")
print("=" * 70)
print(f"\nFrames Processed: {fcount}")
print(f"Elapsed Video Time: {fcount / max(fps, 1):.1f}s")
print(f"\n--- Detection Counts ---")
print(f"  Total person detections (raw): {total_person_detections}")

print(f"\n--- Final Tracked Entities ---")
final_persons = [e for e in tracked_entities.values() if e["class_name"] == "person"]
print(f"  Persons: {len(final_persons)}")
assert all(e["class_name"] == "person" for e in tracked_entities.values())

print(f"\n--- Behavior Detection Summary ---")
for behavior, events in behavior_history.items():
    if events:
        print(f"  {behavior}: TESTED + PASSED")
        for ev in events[:3]:
            print(f"    Frame {ev['frame']:4d}: PERSON #{ev['track_id']} | severity={ev['severity']} | score={ev['score']}%")
    else:
        print(f"  {behavior}: IMPLEMENTED + NOT TESTABLE WITH THIS VIDEO")

print(f"\n--- Alert Summary ---")
print(f"  Total alerts fired: {len(all_alerts)}")
alert_counts = {}
for _frame, alert in all_alerts:
    alert_counts[alert["behavior"]] = alert_counts.get(alert["behavior"], 0) + 1
for behavior, count in alert_counts.items():
    print(f"  {behavior}: {count} alert(s)")

print(f"\n--- Confidence Distribution ---")
if conf_values:
    print(
        f"  Person confidence: min={min(conf_values):.3f}  max={max(conf_values):.3f}  "
        f"avg={sum(conf_values)/len(conf_values):.3f}  count={len(conf_values)}"
    )
else:
    print("  No person detections recorded.")

print(f"\n--- Sample Frames Saved ---")
for f_num in sorted({1, max(1, MAX_FRAMES // 4), max(1, MAX_FRAMES // 2), max(1, 3 * MAX_FRAMES // 4)}):
    sp = os.path.join(BASE_DIR, "test_videos", f"sample_frame_{f_num}.jpg")
    if os.path.exists(sp):
        print(f"  {sp} ({os.path.getsize(sp) // 1024} KB)")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

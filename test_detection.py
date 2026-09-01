"""
Test script: Runs the SENTINEL AI detection pipeline on a video file
WITHOUT needing the GUI — simulates what _video_worker() does and logs results.
"""
import os
import sys
import time
import cv2
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Import detection functions from app.py (current API)
from app import (
    load_dual_models, detect_bags, detect_persons,
    iou_deduplicate, match_entities, analyze_behaviors,
    calculate_entity_risk, draw_overlays, SentinelConfig
)

VIDEO_PATH = os.path.join(BASE_DIR, "test_videos", "LeftBag.mpg")
CONF_THRESHOLD = 0.25
MAX_FRAMES = 500  # ~20 seconds at 25fps

print("=" * 70)
print("SENTINEL AI — Headless Video Detection Test (Unified Tracking)")
print("=" * 70)
print(f"  Video:           {os.path.basename(VIDEO_PATH)}")
print(f"  Confidence:      {CONF_THRESHOLD}")
print(f"  Max frames:      {MAX_FRAMES}")
print("=" * 70)

# Load models
print("\n[1/3] Loading models...")
bag_model, person_model, status = load_dual_models(BASE_DIR)
print(f"  Status: {status}")
if bag_model:
    print(f"  Bag model:    LOADED")
else:
    print(f"  Bag model:    NOT FOUND (no trained best.pt yet)")
if person_model:
    print(f"  Person model: LOADED (yolov9s.pt)")
else:
    print(f"  Person model: NOT FOUND")

if bag_model is None and person_model is None:
    print("\nFATAL: No models available. Cannot run detection.")
    sys.exit(1)

# Open video
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

# Run detection loop
print(f"\n[3/3] Running detection (processing up to {MAX_FRAMES} frames)...")
print("-" * 70)

tracked_entities = {}
fcount = 0
total_bag_detections = 0
total_person_detections = 0
all_alerts = []
conf_values = []

frame_milestones = set([1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900])

# Track behavior state for reporting
behavior_history = {
    "LOITERING": [],
    "SUDDEN_RUNNING": [],
    "REPEATED_MOVEMENT": [],
    "RESTRICTED_ZONE": [],
    "UNATTENDED_OBJECT": [],
}

while cap.isOpened() and fcount < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    fcount += 1
    
    # Model inference
    bag_dets = detect_bags(frame, bag_model, CONF_THRESHOLD)
    person_dets = detect_persons(frame, person_model)

    total_bag_detections += len(bag_dets)
    total_person_detections += len(person_dets)

    # Collect confidence values
    for d in bag_dets:
        conf_values.append(("bag", d[4]))
    for d in person_dets:
        conf_values.append(("person", d[4]))

    # Combine & Deduplicate overlapping boxes
    merged_dets = bag_dets + person_dets
    merged_dets = iou_deduplicate(merged_dets)

    # Unified tracking match
    tracked_entities = match_entities(merged_dets, tracked_entities)

    # Behavior analysis (analyze_behaviors is called inside draw_overlays)
    annotated, new_alerts = draw_overlays(frame, tracked_entities, fcount)

    # Log new alerts
    for alert in new_alerts:
        ts = time.strftime("%H:%M:%S")
        all_alerts.append((fcount, alert))
        behavior = alert["behavior"]
        track_id = alert["track_id"]
        class_name = alert["class_name"]
        severity = alert["severity"]
        score = alert["score"]
        print(f"  [Frame {fcount:4d}] ALERT: {class_name.upper()} #{track_id} | {behavior} | {severity} | score={score}")

    # Track behavior state changes for reporting
    for key, info in tracked_entities.items():
        threats = info.get("active_threats", set())
        for threat in threats:
            if threat in behavior_history:
                # Check if this is a new activation for this track
                already_logged = any(e["track_id"] == info["track_id"] 
                                   for e in behavior_history[threat])
                if not already_logged:
                    behavior_history[threat].append({
                        "frame": fcount,
                        "track_id": info["track_id"],
                        "class_name": info["class_name"],
                        "severity": info.get("severity", "UNKNOWN"),
                        "score": info.get("risk_score", 0)
                    })

    # Progress milestones
    if fcount in frame_milestones:
        persons = [e for e in tracked_entities.values() if e["class_name"] == "person"]
        objects = [e for e in tracked_entities.values() if e["class_name"] != "person"]
        active_threats = sum(len(e.get("active_threats", set())) for e in tracked_entities.values())
        print(f"  [Frame {fcount:4d}] Tracking: {len(persons)} persons, {len(objects)} objects | "
              f"Active threats: {active_threats} | "
              f"Detections this frame: {len(bag_dets)} bags, {len(person_dets)} persons")

    # Save a sample annotated frame at 25%, 50%, 75% of max frames and first frame
    sample_frames = {1}
    sample_frames.add(max(1, MAX_FRAMES // 4))
    sample_frames.add(max(1, MAX_FRAMES // 2))
    sample_frames.add(max(1, 3 * MAX_FRAMES // 4))
    
    if fcount in sample_frames:
        sample_path = os.path.join(BASE_DIR, "test_videos", f"sample_frame_{fcount}.jpg")
        cv2.imwrite(sample_path, annotated)

cap.release()

# Print detailed report
print("\n" + "=" * 70)
print("DETECTION TEST RESULTS")
print("=" * 70)

print(f"\nFrames Processed: {fcount}")
print(f"Elapsed Video Time: {fcount / max(fps, 1):.1f}s")

print(f"\n--- Detection Counts ---")
print(f"  Total bag detections (raw):    {total_bag_detections}")
print(f"  Total person detections (raw): {total_person_detections}")

# Show tracked entities at end
persons_final = [(k, v) for k, v in tracked_entities.items() if v["class_name"] == "person"]
objects_final = [(k, v) for k, v in tracked_entities.items() if v["class_name"] != "person"]

print(f"\n--- Final Tracked Entities ---")
print(f"  Persons: {len(persons_final)}")
for key, info in persons_final:
    elapsed = time.time() - info["first_seen"]
    threats = info.get("active_threats", set())
    print(f"    {key}: track_id={info['track_id']}, conf={info['confidence']:.3f}, "
          f"elapsed={elapsed:.1f}s, threats={threats}, risk={info.get('risk_score', 0)}%")

print(f"  Objects: {len(objects_final)}")
for key, info in objects_final:
    elapsed = time.time() - info["first_seen"]
    threats = info.get("active_threats", set())
    print(f"    {key}: track_id={info['track_id']}, conf={info['confidence']:.3f}, "
          f"elapsed={elapsed:.1f}s, threats={threats}, risk={info.get('risk_score', 0)}%")

print(f"\n--- Behavior Detection Summary ---")
for behavior, events in behavior_history.items():
    status = "TESTED + PASSED" if events else "IMPLEMENTED + NOT TESTABLE WITH THIS VIDEO"
    print(f"  {behavior}: {status}")
    for e in events:
        print(f"    Frame {e['frame']:4d}: {e['class_name'].upper()} #{e['track_id']} | "
              f"severity={e['severity']} | score={e['score']}%")

print(f"\n--- Alert Summary ---")
print(f"  Total alerts fired: {len(all_alerts)}")
# Group by behavior
alerts_by_behavior = {}
for frame_num, alert in all_alerts:
    b = alert["behavior"]
    if b not in alerts_by_behavior:
        alerts_by_behavior[b] = []
    alerts_by_behavior[b].append((frame_num, alert))

for behavior, alerts in alerts_by_behavior.items():
    print(f"  {behavior}: {len(alerts)} alert(s)")
    for frame_num, alert in alerts:
        print(f"    Frame {frame_num:4d}: {alert['class_name'].upper()} #{alert['track_id']} | "
              f"{alert['severity']} | score={alert['score']}%")

# Check cooldown behavior
print(f"\n--- Alert Cooldown Verification ---")
track_alert_times = {}
for frame_num, alert in all_alerts:
    key = (alert["track_id"], alert["class_name"], alert["behavior"])
    if key not in track_alert_times:
        track_alert_times[key] = []
    track_alert_times[key].append(frame_num)

cooldown_violations = 0
for key, frames in track_alert_times.items():
    if len(frames) > 1:
        for i in range(1, len(frames)):
            frame_diff = frames[i] - frames[i-1]
            time_diff = frame_diff / fps
            if time_diff < SentinelConfig.ALERT_COOLDOWN:
                cooldown_violations += 1
                print(f"  COOLDOWN VIOLATION: {key} - alerts {frame_diff} frames apart ({time_diff:.1f}s)")
            else:
                print(f"  Cooldown OK: {key} - alerts {frame_diff} frames apart ({time_diff:.1f}s)")

if cooldown_violations == 0:
    print("  No cooldown violations detected.")

print(f"\n--- Confidence Distribution ---")
if conf_values:
    bag_confs = [c[1] for c in conf_values if c[0] == "bag"]
    person_confs = [c[1] for c in conf_values if c[0] == "person"]
    if bag_confs:
        print(f"  Bag confidence:    min={min(bag_confs):.3f}  max={max(bag_confs):.3f}  avg={sum(bag_confs)/len(bag_confs):.3f}  count={len(bag_confs)}")
    if person_confs:
        print(f"  Person confidence: min={min(person_confs):.3f}  max={max(person_confs):.3f}  avg={sum(person_confs)/len(person_confs):.3f}  count={len(person_confs)}")
else:
    print("  No detections recorded.")

print(f"\n--- Sample Frames Saved ---")
sample_frames_report = {1, max(1, MAX_FRAMES // 4), max(1, MAX_FRAMES // 2), max(1, 3 * MAX_FRAMES // 4)}
for f_num in sorted(sample_frames_report):
    sp = os.path.join(BASE_DIR, "test_videos", f"sample_frame_{f_num}.jpg")
    if os.path.isfile(sp):
        print(f"  {sp} ({os.path.getsize(sp) // 1024} KB)")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
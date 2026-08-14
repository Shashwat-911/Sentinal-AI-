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

# Import detection functions from app.py
from app import (
    load_dual_models, detect_bags, detect_persons,
    match_bag_centroids, match_person_centroids,
    draw_overlays, compute_risk_score, check_owner_return
)

VIDEO_PATH = os.path.join(BASE_DIR, "test_videos", "LeftBag.mpg")
CONF_THRESHOLD = 0.25
ALERT_THRESHOLD_SEC = 15
LOITER_THRESHOLD_SEC = 20
MAX_FRAMES = 900  # ~36 seconds at 25fps, enough to trigger alerts

print("=" * 70)
print("SENTINEL AI — Headless Video Detection Test")
print("=" * 70)
print(f"  Video:           {os.path.basename(VIDEO_PATH)}")
print(f"  Confidence:      {CONF_THRESHOLD}")
print(f"  Alert threshold: {ALERT_THRESHOLD_SEC}s")
print(f"  Loiter threshold:{LOITER_THRESHOLD_SEC}s")
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

tracked_bags = {}
tracked_persons = {}
fcount = 0
total_bag_detections = 0
total_person_detections = 0
all_bag_alerts = []
all_loiter_alerts = []
all_owner_returns = []
conf_values = []

frame_milestones = set([1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900])

while cap.isOpened() and fcount < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    fcount += 1
    bag_dets = detect_bags(frame, bag_model, CONF_THRESHOLD)
    person_dets = detect_persons(frame, person_model)

    total_bag_detections += len(bag_dets)
    total_person_detections += len(person_dets)

    # Collect confidence values
    for d in bag_dets:
        conf_values.append(("bag", d[4]))
    for d in person_dets:
        conf_values.append(("person", d[4]))

    tracked_bags = match_bag_centroids(bag_dets, tracked_bags)
    tracked_persons = match_person_centroids(person_dets, tracked_persons)
    annotated, new_bag_alerts, new_loiter, returned_ids = draw_overlays(
        frame, tracked_bags, tracked_persons,
        ALERT_THRESHOLD_SEC, LOITER_THRESHOLD_SEC, fcount
    )

    # Log new alerts
    for bag_id, elapsed, severity in new_bag_alerts:
        ts = time.strftime("%H:%M:%S")
        all_bag_alerts.append((fcount, bag_id, elapsed, severity))
        print(f"  [Frame {fcount:4d}] ALERT FIRED: BAG #{bag_id} | {severity} | {int(elapsed)}s unattended")

    for pid, dwell in new_loiter:
        all_loiter_alerts.append((fcount, pid, dwell))
        print(f"  [Frame {fcount:4d}] LOITER ALERT: PERSON #{pid} | {int(dwell)}s loitering")

    for rid in returned_ids:
        all_owner_returns.append((fcount, rid))
        print(f"  [Frame {fcount:4d}] OWNER RETURNED: BAG #{rid} - alert cancelled")

    # Progress milestones
    if fcount in frame_milestones:
        current_bags = len([b for b in tracked_bags.values() if not b.get("alert_dismissed")])
        current_persons = len([p for p in tracked_persons.values() if not p.get("loiter_dismissed")])
        print(f"  [Frame {fcount:4d}] Tracking: {current_bags} bags, {current_persons} persons | "
              f"Detections this frame: {len(bag_dets)} bags, {len(person_dets)} persons")

    # Save a sample annotated frame
    if fcount in [1, 250, 500, 750]:
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
print(f"  Unique bags tracked:           {len(tracked_bags)}")
print(f"  Unique persons tracked:        {len(tracked_persons)}")

print(f"\n--- Alert Summary ---")
print(f"  Bag alerts fired:    {len(all_bag_alerts)}")
for a in all_bag_alerts:
    print(f"    Frame {a[0]:4d}: BAG #{a[1]} | {a[3]} | {int(a[2])}s")

print(f"  Loiter alerts fired: {len(all_loiter_alerts)}")
for a in all_loiter_alerts:
    print(f"    Frame {a[0]:4d}: PERSON #{a[1]} | {int(a[2])}s")

print(f"  Owner returns:       {len(all_owner_returns)}")
for a in all_owner_returns:
    print(f"    Frame {a[0]:4d}: BAG #{a[1]}")

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

print(f"\n--- Tracked Bag Details ---")
for bag_id, info in tracked_bags.items():
    elapsed = time.time() - info["first_seen"]
    print(f"  BAG #{bag_id}: risk={info['risk_score']}% | "
          f"alerted={info['is_alerted']} | dismissed={info.get('alert_dismissed', False)} | "
          f"elapsed={int(elapsed)}s | bbox={info['bbox']}")

print(f"\n--- Tracked Person Details ---")
for pid, info in tracked_persons.items():
    print(f"  PERSON #{pid}: dwell={int(info.get('dwell_seconds', 0))}s | "
          f"loitering={info.get('is_loitering', False)} | dismissed={info.get('loiter_dismissed', False)} | "
          f"bbox={info['bbox']}")

print(f"\n--- Sample Frames Saved ---")
for f_num in [1, 250, 500, 750]:
    sp = os.path.join(BASE_DIR, "test_videos", f"sample_frame_{f_num}.jpg")
    if os.path.isfile(sp):
        print(f"  {sp} ({os.path.getsize(sp) // 1024} KB)")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

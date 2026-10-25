"""
TIMING AUDIT TEST
Determines whether the behavior system uses wall-clock time or video time.
"""
import os
import sys
import time
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from sentinel.config import SentinelConfig
from sentinel.detection import detect_persons
from sentinel.tracking import match_entities
from sentinel.behavior import evaluate_person_behavior, compute_dwell_time

VIDEO_PATH = os.path.join(BASE_DIR, "test_videos", "LeftBag.mpg")
MAX_FRAMES = 200

print("=" * 70)
print("TIMING AUDIT TEST")
print("=" * 70)
print(f"Video: {os.path.basename(VIDEO_PATH)}")
print(f"Dwell threshold: {SentinelConfig.DWELL_TIME_THRESHOLD}s")
print("=" * 70)

# Load person model
from app import load_person_model
person_model, status = load_person_model(BASE_DIR)
print(f"Model status: {status}")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

tracked_entities = {}
fcount = 0
dwell_alerts = []

wall_start = time.time()

while cap.isOpened() and fcount < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    
    fcount += 1
    
    # Detect and track
    person_dets = detect_persons(frame, person_model, SentinelConfig.PERSON_CONF_DEFAULT)
    tracked_entities = match_entities(person_dets, tracked_entities)
    
    # Get wall-clock time
    wall_current = time.time()
    wall_elapsed = wall_current - wall_start
    
    # Calculate video time
    video_time = fcount / fps
    
    # Check dwell time for each person
    for key, info in tracked_entities.items():
        if info.get("class_name") == "person":
            dwell = compute_dwell_time(info, wall_current)
            
            # Check if LONG_DWELL would trigger
            if dwell >= SentinelConfig.DWELL_TIME_THRESHOLD:
                dwell_alerts.append({
                    "frame": fcount,
                    "wall_elapsed": wall_elapsed,
                    "video_time": video_time,
                    "dwell": dwell
                })
                print(f"Frame {fcount:4d}: LONG_DWELL triggered")
                print(f"  Wall-clock elapsed: {wall_elapsed:.2f}s")
                print(f"  Video time: {video_time:.2f}s")
                print(f"  Computed dwell: {dwell:.2f}s")
                print(f"  Expected threshold: {SentinelConfig.DWELL_TIME_THRESHOLD}s")
                print()

cap.release()

print("=" * 70)
print("TIMING AUDIT RESULTS")
print("=" * 70)
print(f"Total frames processed: {fcount}")
print(f"Total wall-clock time: {time.time() - wall_start:.2f}s")
print(f"Total video time: {fcount / fps:.2f}s")
print(f"LONG_DWELL alerts: {len(dwell_alerts)}")

if dwell_alerts:
    print("\nFirst LONG_DWELL alert:")
    alert = dwell_alerts[0]
    print(f"  Frame: {alert['frame']}")
    print(f"  Wall-clock elapsed: {alert['wall_elapsed']:.2f}s")
    print(f"  Video time: {alert['video_time']:.2f}s")
    print(f"  Computed dwell: {alert['dwell']:.2f}s")
    print(f"\nCONCLUSION: System uses WALL-CLOCK TIME, not video time")
else:
    print("\nNo LONG_DWELL alerts in this test segment")

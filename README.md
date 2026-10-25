# SENTINEL AI

SENTINEL AI is a desktop surveillance-analysis dashboard for person detection, persistent person tracking, behavior analysis, posture estimation, and non-modal incident review.

The current application is person-only. It uses a YOLO person detector filtered to COCO class `0` and never displays or tracks luggage or other object classes.

## Features

- Live webcam monitoring with person bounding boxes and persistent `PERSON-*` style IDs.
- Recorded-video analysis through the same inference, tracking, behavior, and overlay pipeline used by live monitoring.
- Person movement analysis:
  - dwell time
  - trajectory path length
  - meander ratio
  - grid revisits
  - repeated movement
  - suspicious loitering evidence
- Human posture estimation using `yolov8n-pose.pt`:
  - `STANDING`
  - `SITTING`
  - `BENDING`
  - `LYING_DOWN`
  - `UNKNOWN`
- Temporally confirmed `POSSIBLE_FALL` safety events.
- Privacy-preserving pose overlays with face keypoints excluded.
- Non-modal incident feed with severity, timestamps, track IDs, evidence summaries, acknowledgement, dismissal, and snapshot paths.
- Dataset verification, model training controls, training charts, session statistics, and PDF report export where supported by the current code.

## What The System Does Not Do

- It does not detect luggage, bags, suitcases, backpacks, or unattended objects.
- It does not load or use a custom luggage model such as `models/best.pt`.
- It does not create `BAG-*` tracks or perform owner-luggage association.
- It does not use restricted-zone logic.
- It does not classify behavior as criminality. Behavior scores are explainable risk signals for demonstration and review.

## Runtime Models

- Person detection: `yolov9s.pt`, queried with `classes=[0]`.
- Posture estimation: `yolov8n-pose.pt`.

If a model file is missing or fails to load, the dashboard reports that inline instead of showing a blocking runtime alert.

## Launch

```bat
run.bat
```

Or from PowerShell:

```powershell
cd C:\Users\User\Downloads\Final_ye\Sentinal-AI-
.\run.bat
```

If the local virtual environment is unavailable, install the dependencies in `requirements.txt` and run:

```powershell
python app.py
```

## Main Screens

- **Overview**: Live stream state, processing FPS, active person count, active incident count, recent events, and model status.
- **Live Monitoring**: Webcam source selection, start/stop controls, confidence threshold, video viewport, target details, behavior evidence, posture, risk state, and incident feed.
- **Video Analysis**: File selection, playback controls, progress/time display, person detection and tracking overlays, posture analysis, behavior events, and possible-fall incidents.
- **Incidents**: Incident history with category/severity filters, evidence, acknowledgement, dismissal, and snapshot paths.
- **Dataset**: Existing dataset inspection and verification workflow.
- **Training**: Existing training controls, logs, charts, and exported run artifacts.
- **Statistics**: Session person counts and behavior/posture incident summaries.
- **Settings**: Runtime confidence thresholds, camera index, model status, and behavior/posture configuration values.

## Tests

Relevant regression tests:

```powershell
python test_posture.py
python test_tracker.py
python test_behavior.py
python test_controlled_scenarios.py
python test_synthetic_behavior.py
python test_detection.py
python -c "import app; print('import app ok')"
```

The detection test is headless and uses the sample video in `test_videos/`. It verifies that only person detections enter the tracker.

## Project Layout

```text
app.py                  Tkinter desktop dashboard and UI integration
sentinel/config.py      Tunable runtime thresholds
sentinel/detection.py   Person-only YOLO detection helpers
sentinel/tracking.py    Persistent person tracker
sentinel/behavior.py    Dwell, trajectory, meander, revisit, and loitering logic
sentinel/posture.py     Pose model loading, posture labels, fall confirmation, skeleton drawing
sentinel/risk.py        Person behavior and safety-event risk scoring
sentinel/alerting.py    Video overlays and non-modal incident event creation
```

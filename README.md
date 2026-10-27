# SENTINEL AI

> **Playwright branch:** Runnable browser tests and workshop instructions are in [docs/PLAYWRIGHT.md](docs/PLAYWRIGHT.md). Start with `npm ci`, `npx playwright install chromium`, then `npm run test:e2e`. For ATF repository import, set the branch/ref to **playwright**.

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

## Smart context inside Video Demo

Open **Video Demo**, upload a video and press **PLAY**. The first frame is shown as a
preview immediately; detector loading and processing run in a background worker.
The **Video Context · Local Vision** panel is beside the video, with the existing
person IDs, skeleton overlay, confidence chart, target details, baseline risk and
incident events retained. Only one camera/video source runs at a time.

The added context feature defaults to local Ollama **qwen3.5:4b**, a vision-language model.
Install it with `ollama pull qwen3.5:4b` and keep Ollama running on localhost:11434.
The smaller `qwen3.5:2b` is selectable in **Environment** for faster responses.
The older `qwen3:1.7b` text model may remain installed but is not used for visual context.

- The model receives up to five chronological clean frames plus recent pose, speed,
  path length, meander, reversals, revisits, confidence and dwell measurements.
- Scene description and actual recording start time are optional. For time, use an
  ISO value such as `2026-10-26T14:00:00+05:30`. Blank remains unknown; it is never
  replaced by today's date. Click **Apply** after editing either field.
- **Environment** also accepts usual hours (`09:00-18:00`, including overnight
  schedules such as `22:00-06:00`) and whether activity is expected. Weekday,
  clock period and within/outside usual hours are calculated from the supplied
  recording timestamp plus elapsed video time. Unknown time stays unknown.
- Reports include an action sequence, environment observations, date/time influence,
  possible tasks with supporting evidence and limitations, concern/reassuring factors,
  missing information and classification: not suspicious, suspicious, uncertain or
  safety concern. Possibilities include subjective model percentage estimates, not calibrated probabilities.
  They can overlap and need not sum to 100. The default panel shows three short
  points and up to three hypotheses; **Full report** retains supporting details.
  **Full report** opens a larger scrollable view of the same assessment.
- Context starts after roughly two video seconds and refreshes at least six video
  seconds apart. It does not wait for the original 45-second dwell threshold, and
  can review a scene even when the person detector misses someone.
- At most 30 sampled frames spanning approximately 30 seconds are retained in memory
  and up to five are sent per request.
  One background request and one replaceable pending request prevent an inference
  backlog. Results show the evidence timestamp, not a claim about the current frame.
- Baseline risk calculations and their weights remain unchanged. Context is a separate
  review assessment; it does not erase fall alerts or override existing incidents.
- Interpretation remains fallible, especially with blur, small objects or ambiguous
  actions. Valid JSON/evidence references do not establish factual correctness.
- Images go only to the local Ollama endpoint, not a cloud API. Timeout/model errors
  appear inline without stopping video. First GPU/model loading may be slow.

Local rollback snapshot of the previous implementation: `.local/backups/` (ignored
by Git). New unit tests: `python -m unittest test_context -v`.

## Launch

## Local demo video set

The optional `demo_videos/` folder contains 15 downloaded 320×240 surveillance clips,
balanced into five normal, five context/safety and five suspicious demonstrations.
The MP4 files are intentionally ignored by Git because they are large; the source,
original dataset labels and local file mapping are recorded in
[`demo_videos/MANIFEST.json`](demo_videos/MANIFEST.json). Review the UCF-Crime dataset
terms before redistributing the clips or publishing them with the project.

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

### Continuous summaries and saved incident reports

Analysis continues in bounded background windows while video runs; an end-of-clip
request includes the final available frame. This is sampled analysis, not every-frame
LLM inference. Wait for pending local inference to finish before replaying or changing
video/environment settings (those start a new context session).

The final view shows highest observed estimates, their video timestamps and the latest
estimate for each named hypothesis. Peaks are not a final probability or a decision rule;
early mistakes may have high scores. Similar names are not automatically merged.
Session reports (TXT and JSON) are saved under `incidents/reports/sessions/`.

Each new incident saves its original trigger and snapshot reference in
`incidents/reports/`. Matching scene-context windows are attached as they arrive,
without replacing the original reason for the alert. Matching requires the same
recording session and a window covering the incident timestamp. Some incidents may
have only trigger evidence if inference failed, was disabled, or missed their window.
Use **Incidents → View Report** for the saved explanations and peak estimates.
Incident records reload on restart. Old screenshot-only incidents cannot have their
missing LLM explanations reconstructed automatically.

### Report reliability updates

Open incident report windows now refresh automatically when matching analysis arrives.
Waiting, analysis errors and a completed run without a covering report are distinct.
Incident-triggered analysis has a separate bounded priority queue (eight requests);
queue overload is reported explicitly. Ordinary updates retain a replaceable pending
request so playback cannot build an unlimited backlog.

Hypotheses use stable task families, including inspection/maintenance and tampering,
so wording changes do not produce repeated peak entries. Legacy report names are
mapped conservatively for display; original saved interpretations remain intact.
Brightness is the mean sampled image intensity on a 0–255 scale. Dim/moderate/bright
are descriptive image thresholds, not a calibrated day/night detector. The visual
model separately estimates daylight-like, artificial/night-like or indeterminate
appearance. Actual recording time is never inferred from brightness.

See `QUALITY_PLAN.md` for acceptance checks and the held-out evaluation needed before
making accuracy or calibrated probability claims.

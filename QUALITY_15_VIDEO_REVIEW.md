# Fifteen-video evaluation — 2026-10-27

For the later ten supplemental demo clips, see [QUALITY_ADDITIONAL_10_VIDEO_REVIEW.md](QUALITY_ADDITIONAL_10_VIDEO_REVIEW.md).

## Scope and evidence

All 15 supplied MP4 files decoded successfully: 31,786 frames, approximately
17 minutes 40 seconds. Production person detection, tracking and pose estimation
ran on 551 sampled frames (every two seconds plus the last frame). This verifies
execution and coverage, not bounding-box precision/recall or full-rate tracking.

The baseline and revised 4B visual pipelines each completed 15 selected-window
assessments. Full-video overview sheets and denser event sheets were inspected.
These windows are development examples, not a held-out accuracy benchmark.
Dataset names and reviewer notes are not sent to the model. A video-level anomaly
label does not establish the action or intent in each sampled window.

Local, reproducible artifacts are indexed in
[the run comparison](evaluation/context_review/SUITE_RESULTS.md).
Individual JSON files retain observations, model output, timestamps and image IDs;
JPEG contact sheets show the exact sampled evidence. Revised runs record source
hashes, and the runner rejects attempts to resume with a changed configuration.

## Retained engineering fixes

- Vehicle scenes and vehicle collisions have explicit schema categories. A visible
  driver is not required for a traffic safety event. Ordinary traffic is not a collision.
- Large pixel changes no longer automatically reset context when distinctive
  local image features still match. This retains the lead-in during camera shake,
  zooming or a large moving foreground object. It remains a heuristic.
- Text separates image markers and binds each image to its evidence ID. The
  observation pass reviews individual frames before temporal interpretation.
- Truncated visual observations are rejected instead of becoming complete reports.
- Explicitly negated striking/kicking/impact statements cannot themselves trigger
  the physical-event review policy.
- Session reports retain earlier review flags even when the latest view is quiet.
  Saved session entries now include the context packet, excluding encoded images.
- The local Qwen3-VL 8B model is installed as a comparison option; 4B remains the
  default pending evidence of a useful accuracy/latency improvement.

Original detector weights, person-only tracking, posture rules and movement-risk
weights are unchanged. Existing incident files are not rewritten.

## Confirmed limitations

The 4B runs missed several physical interactions, and some descriptions invented
falls, entry or routine actions. Five selected normal windows remained not-suspicious
in both runs, but that is not proof of a low false-positive rate on other footage.
The repeated successful doorway label is not sufficient to establish general reliability.

The simple three-image transport diagnostic succeeded both with adjacent markers
and separated markers. The upstream image-merging issue is therefore **not
confirmed** on this installation. Separators provide explicit image/evidence binding;
they are not claimed to solve recognition.

The first 8B comparison windows cost approximately 157–263 seconds each and did not
improve the missed junction event. Full comparison and short-window experiments
are recorded separately; model size alone is not treated as a fix.

## Reproduction

```powershell
python evaluation/detector_suite.py
python evaluation/review_suite.py --run new_4b_run --windows evaluation/demo_windows.json
python evaluation/review_suite.py --run new_8b_run --windows evaluation/demo_windows.json --model qwen3-vl:8b-instruct
python evaluation/review_suite.py --run new_short_run --windows evaluation/demo_windows.json --short-window
python evaluation/summarize_suite.py
python -m unittest test_context test_incident_reports test_video_demo -q
```

Use a new run name after changing code, model or windows. The short-window option
is an evaluation experiment, not a claim about the app's current sampling policy.
The targeted regression suite passed 39 tests. Do not use broad unittest discovery:
the legacy `test_detection.py` runs a GPU/video demo at import time.

Upstream reference: [Ollama issue 17321](https://github.com/ollama/ollama/issues/17321).


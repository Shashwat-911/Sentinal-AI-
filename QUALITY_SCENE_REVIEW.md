# Scene-context quality review — 2026-10-27

For the subsequent broader checks and current limitations, see
[the 15-video evaluation](QUALITY_15_VIDEO_REVIEW.md). The successful category
outputs below were not consistently reproduced and are not accuracy guarantees.

This review addresses the three supplied demo reports. It is a targeted regression
review, not a measured accuracy benchmark or a claim of human-level understanding.

## Changes

- Installed local `qwen3-vl:4b-instruct` and selected it as the default. A first pass
  describes the chronological image sequence; a second pass builds a structured report
  from that observation record. This keeps motion context without confusing collage
  positions and retains the visual source record
  in the saved incident/session packet. Both passes use local Ollama.
- Separate daylight-like, artificial illumination, low visibility and indeterminate
  lighting. Artificial light does not imply night; image brightness never sets the
  recording clock. Full reports include the model's cited lighting cues.
- Establish visible subject scope before human task hypotheses. Animal-only and
  empty scenes cannot populate human-task probabilities or their session peaks.
- Generate observations before the verdict. Unknown intent alone is insufficient
  to classify ordinary behavior as uncertain. Falls/restraint/struggles are physical
  safety concerns even without weapons, tools or visible property damage.
- Include conflict, assistance, accidental fall and routine movement task families.
  Deduplicate repeated families by retaining the strongest estimate, without adding
  percentages; sort estimates consistently. Remove empty placeholder list entries.
- Preserve up to five clean frames, combining temporal endpoints with a burst
  around the strongest sampled image change. Static windows retain evenly spaced
  sampling. Broad view/illumination changes start a fresh evidence window; the
  boundary appears in the full report. Image change never sets risk scores.
- Preserve original detector, tracker, pose, movement alerts and incident linkage.
  Existing historical reports remain historical; replay generates new assessments.
- Evidence-linked physical events apply an explicit minimum-review policy: reported
  forceful object contact requires suspicious review; reported interpersonal restraint
  or a fall requires safety review. The explanation retains the initial model label
  and identifies this policy. It does not establish intent, permission or criminality.
  Event references must point to frames supplied to the visual model.
- Overlong display-only lists are shortened before validation; action summaries retain
  the final step. Invalid types, event references and percentages still fail validation.
- Explicitly negated events such as "No visible impact" are excluded from the review
  policy. Their presence in an event enum cannot raise an alert.

## Model comparison notes

The original Qwen3.5 4B frequently missed or misdescribed the later doorway action.
Qwen3-VL with five images and a structured report in one large request exceeded the timeout; a deeply
grammar-constrained report also produced truncated output. Longer reasoning mode
could exhaust its budget before producing JSON. These configurations were rejected.
The retained Qwen3-VL path separates the unconstrained visual observation from
schema-constrained text reporting, with the schema explicitly provided to the model
and checked locally. Independent frame descriptions were also tested but missed the
doorway motion sequence. A single collage was tested
but confused the doorway occlusion with a fall; collages are now for human inspection
only, not the production visual observation pass.

## Reproduce the visual review

```powershell
python evaluation/review_demo_context.py --assess --adaptive
python -m unittest test_context test_incident_reports test_video_demo -q
python test_controlled_scenarios.py
```

Use `--model` to compare installed models, `--case` to select a case, or omit
`--assess` to extract only contact sheets. Omit `--adaptive` to compare the original
five review timestamps. Outputs are local under `evaluation/context_review/`,
excluded from Git. Filenames and dataset category labels are not sent to the model.
No detector/pose inference is run by this manual VLM review harness.

Cases and review criteria:

| Clip/window | Visible evidence to check |
| --- | --- |
| Normal003, 3–32s | Dog in a daylight-lit passage; no invented human gate manipulation. |
| Arrest003, 7–28.5s | Standing interaction develops into physical contact and a person on the ground; do not dismiss the hazard because weapons are absent. |
| Burglary005, 16–44.8s | Person standing/turning at doorway; no invented seated posture. Sunlit lawn beyond shaded foreground. |
| Burglary005, 150–170s | Later vigorous doorway contact; denser inspection around 156–168s reveals actions omitted by sparse uniform samples. |

These windows are not interchangeable: a dataset filename does not make its entire
duration suspicious. The later doorway action must not retroactively label earlier
ordinary-looking frames. Visible daylight remains separate from verified clock time.

## Verification results

Earlier five-frame checks used the default Qwen3-VL observation/report pipeline. The early
doorway window was rerun after fixing the negated-event false alarm. These are
targeted sampled windows, not full-video detector replays or a held-out benchmark.

| Window | Final classification | Lighting appearance | Local latency |
| --- | --- | --- | --- |
| Normal003, 3–32s | Not suspicious; animal-only, no human tasks | Daylight-like | 49.55s |
| Arrest003, 7–28.5s | Safety concern | Daylight-like | 51.19s |
| Burglary005, 16–44.8s | Not suspicious; standing, no invented sitting | Daylight-like | 48.41s |
| Burglary005, 150–170s | Suspicious review for doorway lunge/contact; no invented fall | Daylight-like | 46.75s |

Validation: **34 context/report/playback regression tests passed**, plus **all six
controlled baseline movement scenarios**. Original detection/tracking/pose/scoring
modules were not changed. Raw observations, full reports, and contact sheets are in
`evaluation/context_review/qwen3-vl_4b-instruct_adaptive/` (local, ignored by Git).

Installed model digest:
`84f7ed8042fa7cdbd902f7dcb7d0370e032095047cc3244179f263b77b5bc318`.

The interaction report still misattributes some actor details (which person fell)
and can add weak environmental explanations. Correct high-level review categories
do not establish that every sentence is factually correct. This remains a specific
limitation for further evaluation, not a solved accuracy claim.

## Remaining limits

### Follow-up: Vandalism007 (2026-10-27)

The supplied report stops at 26s; the file lasts approximately 38.2s. Inspection
found a camera-view cut at about 14s. The former motion selector treated that cut
as the strongest movement and supplied 13/14/15s as its action burst. The retained
fix starts a new evidence window when more than 35% of resized grayscale pixels
change by over 25 intensity levels. This is a broad-change heuristic, not a
semantic shot detector: camera motion or exposure changes can also reset a window.
Original detector/tracker state and earlier incident reports are preserved.

A six-frame experiment combined uniform coverage and a motion pair. It still
missed the vehicle interaction and regressed the later doorway example to
not-suspicious; that sampler was rejected and the five-frame burst restored.
The six-frame run returned: normal/not-suspicious, interaction/safety-concern,
early-door/not-suspicious, later-door/not-suspicious, vehicle-26/not-suspicious,
vehicle-end/not-suspicious. Its artifacts are retained locally in
`evaluation/context_review/rejected_six_frame_20261027/` for comparison.

Do not interpret these category outputs as six successful accuracy checks. The
vehicle observation invented a door-opening/entry sequence; the interaction report
still confused actors; the final vehicle window incorrectly suggested daylight.
The lighting prompts explicitly warn that shadows can come from lamps, but that
instruction alone does not guarantee correct visual interpretation. The broad-cut
fix addresses evidence selection, not these unresolved model errors.

After restoring the five-frame sampler, all 36 context/report/playback regression
tests passed. All six controlled baseline movement scenarios also passed. These
tests verify software behavior, not visual understanding. No full-video YOLO/UI
replay was performed in this follow-up; the visual harness assesses sampled windows.

The final five-frame later-door rerun also returned not-suspicious (daylight-like,
169.05s). Therefore the earlier successful doorway result is not a reliable accuracy
claim, and the six-frame change alone cannot explain the regression. Visual event
recognition remains unresolved; no rule was added to force a dataset's expected label.
The final five-frame vehicle-end rerun returned not-suspicious with artificial-light
appearance in 39.53s, covering the post-change 32–38s window. This does not clear the
earlier vehicle interaction or verify nighttime. Results are per sampled window.

Five sampled images cannot recover every brief action. Camera movement, illumination
changes and occlusion can divert motion-based selection. Low-resolution footage can
still produce incorrect descriptions or categories. Model percentages remain
uncalibrated and overlapping; session peaks are not final probabilities.

For broader quality measurement, use held-out clips with independently annotated
time intervals and review observable actions, false alerts, missed hazards and
response latency separately. Include empty scenes, animals, shaded daylight,
artificially lit daytime rooms, night scenes, ordinary waiting, assistance and actual
physical conflict. Do not optimize solely for these three examples.

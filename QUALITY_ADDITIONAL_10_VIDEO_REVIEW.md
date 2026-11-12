# Additional 10-video review — 2026-10-28

## Scope

All ten supplied MP4s were decoded successfully: **3,528 frames** across about **130 seconds**. The local Qwen3-VL 4B report path completed ten full-window assessments; the average generation time in the saved run was about **48 seconds per clip**. Each report includes five sampled frames, the evidence timestamps, lighting, hypotheses and a short explanation. The car-item and storefront cases were rerun after their targeted fixes.

The production person detector, tracker and pose model were also exercised on **78 frames sampled about every two seconds**. All clips decoded. Person boxes appeared in 66 samples; some of the other samples were expected empty views or partially occluded subjects. This is a smoke test, not precision/recall: no ground-truth boxes or frame-level behavior labels were supplied.

These are stock/staged demo videos, not verified CCTV incidents. The Pixabay AI-generated wreck is not authentic CCTV. This review checks whether the outputs match visible evidence; it does not establish real-world accuracy.

## Final reports

| Clip | Context result | What the sampled frames support | Lighting appearance |
| --- | --- | --- | --- |
| AI car wreck | Not suspicious | The damaged car is already stationary; an emergency-lit vehicle passes. No collision is visible in this clip. | Artificial light |
| Argument | Not suspicious | Two people gesture and talk on a bridge; no physical contact or fall is visible. | Daylight-like |
| City walk | Not suspicious | Routine pedestrians in a subway station. | Artificial light |
| Crosswalk crowd | Not suspicious | Pedestrians cross while vehicles remain on the road; no impact is visible. | Daylight-like |
| Mall crowd | Not suspicious | People walk and gather; the sampled frames do not show a transfer or conflict. | Indeterminate; brightness alone was not used to claim daylight |
| Night walk | Not suspicious | One person walks along a dim road without interacting with people or objects. | Low visibility; recording time remains unknown |
| Parking man | Not suspicious | A man stands in a garage and adjusts his clothing. Clothing or waiting alone did not trigger suspicion. | Artificial light |
| Street crowd | Not suspicious | The full continuous clip shows pedestrians, a tram passing and a dog running. No collision or interaction is visible. | Indeterminate |
| Thief car | **Suspicious review** | A person remains outside the car, reaches through its window, removes a handbag and leaves. Ownership and permission are unknown; this is not a theft finding. | Daylight-like |
| Urban walk | Not suspicious | A motorcycle and a pedestrian pass through the frame without visible contact. | Daylight-like |

The “suspicious” result requests human review of the visible removal from the car. It does not claim guilt or establish that the item was taken unlawfully. The model’s task percentages remain uncalibrated estimates, not measured probabilities.

## Fixes made during review

- Resized sampled images to the same 512-pixel maximum used by the live context engine. The first harness pass sent full 1080p/4K frames and exceeded Ollama’s 8,192-token context; the production-sized pass completed all ten reports.
- Changed visual sampling to retain evenly spaced temporal anchors and substitute one anchor with the strongest sampled change. This avoids spending nearly the whole image budget on one motion burst.
- Improved scene-cut handling with local background-tile matching. Large moving foreground objects no longer reset the continuous low-angle street clip; all ten final report packets have no scene boundary.
- Tightened report validation so an ordinary grasp/pull cannot be mislabeled `forceful_object_contact`. A clear removal from a vehicle is a separate review case and does not claim force or criminal intent.
- Added wording correction for an arm reaching through a car window, so the report says the person reaches in from outside rather than claiming a full-body vehicle entry.
- Corrected lighting from visual cues: explicit fluorescent fixtures produce “artificial light,” a dark low-illumination road produces “low visibility,” and brightness/shadows alone cannot establish daylight. “Artificial light” does not mean nighttime.

## Detector and residual limits

- The person-only detector found no person in the wreck clip, where no person is visible in the sampled frames. In the close car-interior theft scene it detected a person in only **2 of 6** two-second samples because most of the visible subject is outside the frame or occluded. The context model still described the hand and item transfer. This remains a detector limitation for close-up/partial-body CCTV.
- The argument report conservatively labels the scene not suspicious because the sampled interaction has no visible physical contact or fall. Social disagreement is not itself a violence finding.
- The street crowd lighting remains indeterminate at this sample resolution. The report avoids claiming daylight from image brightness alone.
- Five frames are sparse visual coverage. Short actions between samples can be missed; the report prints timestamps and warns about gaps.
- No confusion matrix or “correct prediction” rate is justified without manually labeled ground truth. These clips are useful demos and regression examples, not a validation dataset.

## Verification and artifacts

The focused regression suite passed **49 tests** (`test_context`, `test_incident_reports`, and `test_video_demo`), and the modified Python modules passed `py_compile`.

- Final per-video text, JSON and contact-sheet reports, plus detector sample JSON: `evaluation/context_review/additional_pixabay_final_4b_r4/`
- Ten-clip detector summary: `evaluation/context_review/additional_pixabay_final_4b_r4/detector/summary.json`
- Reproducible video windows: `evaluation/additional_demo_windows.json`
- Detector smoke runner: `evaluation/additional_detector_suite.py`
- Final report pack assembler: `evaluation/finalize_additional_review.py`

The report pack records its source runs and current source hashes in `configuration.json`. The item-removal and full-storefront windows were rerun under the final report policy; the other eight use the completed full-window batch with the final deterministic report review reapplied.

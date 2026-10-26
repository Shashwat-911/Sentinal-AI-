# Quality improvement and acceptance plan

## Implement and verify now

1. Incident-report lifecycle: automatically refresh open reports, retain the original
   rule trigger, distinguish waiting/failure/no covering analysis, and persist updates.
2. Continuous review: bounded routine requests, protected incident-priority requests,
   and a final frame request. Surface queue overload instead of silently claiming success.
3. Evidence quality: pass repeated-movement alerts accurately; separate rule severity
   from model classification. Neither maintenance nor lock picking is a predetermined answer.
4. Environment: show measured mean image brightness and an explicit visual lighting
   appearance. Do not substitute brightness for the actual recording time.
5. Consistency: stable task families group hypothesis wording changes; show both latest
   and peak estimates. Percentages remain uncalibrated model estimates.
6. End-to-end check: run tre.mp4 through the real detector, pose pipeline, local 4B model,
   UI polling and incident persistence; verify final-window completion and saved reports.

## Next quality gate before accuracy claims

Use independently labelled normal, ambiguous and concerning clips, with a held-out test
set. Compare the original rules, model-only review, and combined interface. Measure false
alarms per video minute, event precision/recall, missed events, uncertain-rate, inference
latency and incident report coverage. Separate camera jitter/occlusion cases. Review
tracking displacement before changing the preserved movement weights. Do not tune only
to tre.mp4 or rank lock picking first merely because it is the expected demonstration.

Evaluate hypothesis percentage calibration only against observable, labelled outcomes;
unknown intent cannot supply ground truth. Peak estimates are audit history, not a
calibrated final score. Human review remains necessary for ambiguous footage.

## Recorded verification

See `QUALITY_VERIFICATION.md` for the completed full-video run, report coverage,
latency and limitations. The concrete lifecycle fixes pass; general accuracy and
real-time performance remain evaluation targets, not assumed achievements.

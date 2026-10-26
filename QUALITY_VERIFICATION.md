# Local quality verification — 2026-10-26

Input: C:/Users/Shashwat/Downloads/tre.mp4

- Completed all 855 frames, through 28.4723 seconds.
- 7 successful local Qwen3.5 4B assessments; no model errors.
- Two original movement incidents retained; matching review counts: [5, 3].
- Open incident report refreshed automatically: True.
- Final-frame assessment completed, classification: uncertain.
- Model response latency: 18.81–23.07 seconds.
- Full-frame CPU verification wall time: 548.29 seconds.
  The test constrained CPU threads to two; this is not an application throughput benchmark.
- 22 context/report/playback regression tests passed, plus six controlled baseline behavior scenarios.

The test used the actual detector, pose pipeline and local model. Test incidents and
reports were isolated under .local/quality_verified; user incident files were preserved.
Queue ordering, retained rule flags after detector misses, atomic session persistence
and final UI wording received targeted regression tests after the long run was started.

This validates execution and report linkage on this clip, not factual recognition
accuracy. The final model result remains uncertain. A red rule alert is not ground
truth for lock picking. Brightness-derived night claims are changed to indeterminate
when no physical lighting cue is cited. No crime label or hypothesis rank was forced.

Remaining evaluation: held-out labelled clips, false-positive/miss rates, calibrated
estimates, camera jitter cases, and a separate GPU/CPU performance benchmark. See
QUALITY_PLAN.md. No claim of human-level reasoning or real-time full-frame inference.

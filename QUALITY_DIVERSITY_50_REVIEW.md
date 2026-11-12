# Diversity 50 review — 3 November 2026

## Scope and evidence

Inspected 20 timestamped overview frames for each of 50 videos (1,000 sampled frames), and inspected selected report evidence sheets. These are independent assistant visual judgments, not continuous playback or verified per-window ground truth. The model reports were already available; this is not a blinded evaluation. All 221 successful original reports were scanned for evidence coverage and report-policy issues. Original files remain preserved.

Individual observations and comparison triage are saved in `evaluation/context_review/diversity_50_audited/visual_triage.json`. Sparse overviews cannot establish the absence of brief events or criminal intent. Dataset names were not treated as proof of activity.

## Implemented fixes

- Negation filtering now rejects `No visible force or impact` as a force event. Applied to saved reports, this corrected Arrest051 at 200s and Stealing043 at 75s from suspicious to their original not-suspicious visual verdicts. Arrest051 still warrants combined review from its rule vote.
- Added fire/smoke to the event schema and safety policy, including scenes without people. Both visual observation paths now explicitly inspect environmental hazards.
- Added one bounded larger-output retry for truncated visual observations; repeated truncation still fails explicitly.
- Reports expose lost requested coverage after a view change; a combined benign vote cannot clear an interval when its earlier footage was discarded.
- Prompt instructs the visual vote not to copy detector suspicion labels as its own evidence, and to consider a visible throw toward a building even if the landing is off-camera.

55 focused tests passed, including new regressions for denied force, fire without people, and lost temporal coverage. `git diff --check` passed.

## Actual rerun results and remaining errors

The failed RoadAccidents151 window successfully generated a fresh report. It is included in the audited bundle, giving 222 available reports. A successful generation does not establish a correct collision judgment.

Two Arson053 windows and an ordinary bowling clip were rerun with the revised main prompt. The ordinary control remained not suspicious. At 150–179s, clear flames were still described as illumination. At 200–225.8s, the sampler discarded the earlier footage and retained the black ending frame; this is a coverage failure, not evidence that the entire interval was safe.

A further per-frame visual experiment on Arson053 and Vandalism026 also returned not suspicious. The fire and throwing recognition errors therefore remain unresolved. The per-frame prompt was subsequently updated to explicitly request environmental hazards; this last wording change has unit coverage but has not had a fresh model rerun.

Overview triage identifies Arson053, Robbery001, Vandalism026 and Vandalism050 as priority missed-review candidates. Other ambiguous low-resolution interactions are marked uncertain in the independent notes, rather than forced into dataset labels. No accuracy percentage is claimed. The normal footage provides a useful control but this set was also used for development, so it is not a held-out benchmark.

## Outputs

- `evaluation/context_review/diversity_50_audited/`: 222 report records, corrected text, independent visual notes, policy changes and issue list.
- `evaluation/context_review/diversity_50_review_r1/`: fresh failed-window retry, fire windows and normal control.
- `evaluation/context_review/diversity_50_review_r2/`: targeted per-frame experiment.
- `evaluation/context_review/diversity_50_v1/`: preserved original inference outputs and overview sheets.

Most audited reports reuse saved inference with deterministic validation corrections. They must not be described as a new 50-video model run or a fully corrected classifier.

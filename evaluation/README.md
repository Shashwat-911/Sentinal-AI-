# Behaviour Evaluation Dataset

This directory is intentionally structured for behavioural evaluation, not for automated training.

The objective is to label tracked-person episodes as:

- NORMAL
- SUSPICIOUS

The evaluation is performed at the tracked-person / behavioural-episode level rather than at per-frame granularity.

## Dataset schema

The CSV file is expected to contain a row per behavioural episode:

video,track_id,ground_truth,start_time,end_time,notes,ground_truth_onset

Example rows:

normal_walk_01.mp4,PERSON-1,NORMAL,0,60,straight walking,
loitering_01.mp4,PERSON-1,SUSPICIOUS,0,60,repetitive pacing,22

Notes:

- Ground truth must be explicitly labelled by an evaluator.
- `ground_truth_onset` is optional and only used for alert latency.
- Do not fabricate labels for videos that have not been manually labelled.
- The existing repo contains validation tests and test videos, but it does not currently contain a sufficient real-world suspicious behaviour dataset for a statistically meaningful confusion matrix.
- The bundled sample video is retained only as a generic headless smoke-test clip and is not a labelled suspicious-person dataset.

## Expected directories

- `videos/` — labelled evaluation clips
- `labels.csv` — episode-level ground truth labels
- `results/` — evaluation outputs and confusion matrix
- `plots/` — optional charts if enough labelled data exists

Use the evaluation script:

python evaluate_behavior.py --labels evaluation/labels.csv --video-dir evaluation/videos --results-dir evaluation/results --plots-dir evaluation/plots

If no rows or no valid videos are present, the script will exit cleanly and print the missing-data message instead of inventing metrics.

"""Evaluation harness for SENTINEL AI behavioural classification.

This script reuses the existing detector, tracker, and behaviour pipeline without changing the
core behavioural logic. It evaluates each labelled episode at the tracked-person level and
produces a binary NORMAL vs SUSPICIOUS confusion matrix when a labelled dataset is available.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sentinel.config import SentinelConfig
from sentinel.detection import detect_persons, load_dual_models
from sentinel.tracking import match_entities
from sentinel.behavior import evaluate_person_behavior
from sentinel.risk import analyze_behaviors
from sentinel.evaluation import (
    NORMAL,
    SUSPICIOUS,
    compute_binary_metrics,
    normalize_behaviour_label,
    read_label_csv,
    save_confusion_matrix_png,
    summarize_alert_latency,
    normalise_prediction_from_track,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SENTINEL AI behaviour labels against the existing pipeline.")
    parser.add_argument("--labels", default="evaluation/labels.csv", help="Path to the evaluation CSV file.")
    parser.add_argument("--video-dir", default="evaluation/videos", help="Directory containing labelled evaluation videos.")
    parser.add_argument("--results-dir", default="evaluation/results", help="Directory where episode results are written.")
    parser.add_argument("--plots-dir", default="evaluation/plots", help="Directory where plots are written.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Person detection confidence threshold.")
    return parser.parse_args()


def resolve_video_path(video_dir, filename):
    if not filename:
        return None
    candidates = [
        os.path.join(video_dir, filename),
        os.path.join(BASE_DIR, video_dir, filename),
        os.path.abspath(filename),
        os.path.join(BASE_DIR, filename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def choose_track_for_episode(tracked_entities, desired_track_id=None):
    person_tracks = [
        entry for entry in tracked_entities.values()
        if str(entry.get("class_name", "")).lower() == "person"
    ]
    if not person_tracks:
        return None
    if desired_track_id is not None:
        for entry in person_tracks:
            track_id = entry.get("track_id")
            if str(track_id) == str(desired_track_id):
                return entry
    return max(person_tracks, key=lambda item: item.get("detection_count", 0))


def process_episode(label_row, video_dir, person_model):
    filename = str(label_row.get("video", "")).strip()
    if not filename:
        return None

    video_path = resolve_video_path(video_dir, filename)
    if video_path is None:
        return None

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None

    start_time = float(label_row.get("start_time", 0.0) or 0.0)
    end_time = float(label_row.get("end_time", 0.0) or 0.0)
    desired_track_id = label_row.get("track_id")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        capture.release()
        return None

    tracked_entities = {}
    episode_end_time = max(end_time, start_time)
    last_prediction = NORMAL
    prediction_timestamp = None
    alert_time = None

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        current_time = frame_index / fps
        if current_time < start_time:
            frame_index += 1
            continue
        if current_time > episode_end_time and end_time > start_time:
            break

        detections = detect_persons(frame, person_model, conf=0.25)
        tracked_entities = match_entities(detections, tracked_entities, current_time=current_time)
        analyze_behaviors(tracked_entities, frame.shape[1], frame.shape[0], current_time=current_time)

        target_track = choose_track_for_episode(tracked_entities, desired_track_id)
        if target_track is not None:
            last_prediction = normalise_prediction_from_track(target_track)
            prediction_timestamp = current_time
            alert_time = target_track.get("last_alert_time")
            if target_track.get("last_alert_time", 0.0) > 0:
                alert_time = float(target_track.get("last_alert_time", 0.0))
        frame_index += 1

    capture.release()

    if target_track is None:
        return None

    ground_truth = normalize_behaviour_label(label_row.get("ground_truth", NORMAL))
    if ground_truth is None:
        ground_truth = NORMAL

    prediction = last_prediction
    prediction_time = prediction_timestamp if prediction_timestamp is not None else episode_end_time

    alert_latency = None
    if ground_truth == SUSPICIOUS and row_has_ground_truth_onset(label_row):
        onset_value = label_row.get("ground_truth_onset")
        try:
            onset = float(onset_value)
        except (TypeError, ValueError):
            onset = None
        if onset is not None and alert_time is not None:
            alert_latency = float(alert_time) - onset

    return {
        "video": filename,
        "track_id": desired_track_id,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "prediction_timestamp": prediction_time,
        "alert_time": alert_time,
        "alert_latency": alert_latency,
        "correct": prediction == ground_truth,
        "notes": label_row.get("notes", ""),
    }


def row_has_ground_truth_onset(label_row):
    return label_row.get("ground_truth_onset") not in (None, "", "NA")


def ensure_directories(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def write_results_csv(path, rows):
    fieldnames = [
        "video",
        "track_id",
        "ground_truth",
        "prediction",
        "prediction_timestamp",
        "alert_time",
        "alert_latency",
        "correct",
        "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_confusion_summary(metrics):
    print("==================================================")
    print("SENTINEL AI BEHAVIOURAL EVALUATION")
    print("==================================================")
    print(f"Samples:\nNormal = {metrics['normal_count']}\nSuspicious = {metrics['suspicious_count']}\nTotal = {metrics['total_count']}")
    print()
    print("CONFUSION MATRIX")
    print("                 Pred Normal   Pred Suspicious")
    print(f"Actual Normal      {metrics['TN']}             {metrics['FP']}")
    print(f"Actual Suspicious  {metrics['FN']}             {metrics['TP']}")
    print()
    print("METRICS")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1: {metrics['f1']}")
    print(f"Specificity: {metrics['specificity']}")
    print(f"False Positive Rate: {metrics['false_positive_rate']}")
    print(f"False Negative Rate: {metrics['false_negative_rate']}")
    print("==================================================")


def main():
    args = parse_args()
    labels_path = args.labels
    video_dir = args.video_dir
    results_dir = args.results_dir
    plots_dir = args.plots_dir

    ensure_directories(results_dir, plots_dir)

    label_rows = read_label_csv(labels_path)
    if not label_rows:
        print(f"No labelled behavioural episodes were found in {labels_path}.")
        print("The repository does not currently contain a labelled real-world suspicious-behaviour dataset.")
        print("To evaluate the model properly, add labelled videos in evaluation/videos and populate evaluation/labels.csv.")
        print("Sample required columns: video,track_id,ground_truth,start_time,end_time,notes,ground_truth_onset")
        return 0

    base_dir = BASE_DIR
    bag_model, person_model, status = load_dual_models(base_dir)
    if person_model is None:
        print("Person detection model did not load; evaluation cannot proceed.")
        return 1

    valid_rows = []
    for row in label_rows:
        result = process_episode(row, video_dir, person_model)
        if result is not None:
            valid_rows.append(result)

    if not valid_rows:
        print("No usable labelled videos were found in the evaluation dataset.")
        print("Expected files in evaluation/videos and ground-truth rows in evaluation/labels.csv.")
        return 0

    output_path = os.path.join(results_dir, "behavior_evaluation.csv")
    write_results_csv(output_path, valid_rows)

    y_true = [row["ground_truth"] for row in valid_rows]
    y_pred = [row["prediction"] for row in valid_rows]
    metrics = compute_binary_metrics(y_true, y_pred)
    metrics["normal_count"] = sum(1 for label in y_true if normalize_behaviour_label(label) == NORMAL)
    metrics["suspicious_count"] = sum(1 for label in y_true if normalize_behaviour_label(label) == SUSPICIOUS)
    metrics["total_count"] = len(valid_rows)
    print_confusion_summary(metrics)

    confusion_path = os.path.join(results_dir, "confusion_matrix.png")
    save_confusion_matrix_png(y_true, y_pred, confusion_path)
    print(f"Saved evaluation results to: {output_path}")
    print(f"Saved confusion matrix figure to: {confusion_path}")

    latencies = summarize_alert_latency(valid_rows)
    if latencies.get("available"):
        print("Alert latency summary:")
        print(f"Mean: {latencies['mean']:.4f}s")
        print(f"Median: {latencies['median']:.4f}s")
        print(f"Min: {latencies['min']:.4f}s")
        print(f"Max: {latencies['max']:.4f}s")
    else:
        print(latencies.get("message"))

    false_positives = [row for row in valid_rows if row["ground_truth"] == NORMAL and row["prediction"] == SUSPICIOUS]
    false_negatives = [row for row in valid_rows if row["ground_truth"] == SUSPICIOUS and row["prediction"] == NORMAL]

    print("False positives:")
    if false_positives:
        for row in false_positives:
            print(f"  - {row['video']} / {row['track_id']}")
    else:
        print("  - None")

    print("False negatives:")
    if false_negatives:
        for row in false_negatives:
            print(f"  - {row['video']} / {row['track_id']}")
    else:
        print("  - None")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

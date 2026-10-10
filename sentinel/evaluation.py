"""Utilities for evaluating SENTINEL AI behavioural predictions against labelled episodes."""

from __future__ import annotations

import csv
import math
import os
import statistics
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import ConfusionMatrixDisplay
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

NORMAL = "NORMAL"
SUSPICIOUS = "SUSPICIOUS"
LABELS = [NORMAL, SUSPICIOUS]


def normalize_behaviour_label(value):
    """Map raw labels and states to the binary target classes used for evaluation."""
    if value is None:
        return None

    value_text = str(value).strip().upper()
    if value_text in {"NORMAL", "NO", "FALSE", "0"}:
        return NORMAL
    if value_text in {"SUSPICIOUS", "SUSPICIOUS_BEHAVIOUR", "ALERT", "CRITICAL", "HIGH", "LOITERING", "TRUE", "1"}:
        return SUSPICIOUS
    if value_text in {"OBSERVING", "LOW", "MEDIUM", "COOLING_DOWN"}:
        return NORMAL
    return value_text


def compute_confusion_counts(y_true, y_pred):
    """Return TN, FP, FN, TP for the binary NORMAL vs SUSPICIOUS classification."""
    true_labels = [normalize_behaviour_label(v) for v in y_true]
    pred_labels = [normalize_behaviour_label(v) for v in y_pred]

    valid_entries = []
    for t, p in zip(true_labels, pred_labels):
        if t is None or p is None:
            continue
        valid_entries.append((t, p))

    if not valid_entries:
        return 0, 0, 0, 0

    true_arr = [t for t, _ in valid_entries]
    pred_arr = [p for _, p in valid_entries]

    if HAS_SKLEARN:
        cm = confusion_matrix(true_arr, pred_arr, labels=LABELS)
        tn, fp, fn, tp = cm.ravel()
        return int(tn), int(fp), int(fn), int(tp)
    else:
        tn = sum(1 for t, p in zip(true_arr, pred_arr) if t == NORMAL and p == NORMAL)
        fp = sum(1 for t, p in zip(true_arr, pred_arr) if t == NORMAL and p == SUSPICIOUS)
        fn = sum(1 for t, p in zip(true_arr, pred_arr) if t == SUSPICIOUS and p == NORMAL)
        tp = sum(1 for t, p in zip(true_arr, pred_arr) if t == SUSPICIOUS and p == SUSPICIOUS)
        return int(tn), int(fp), int(fn), int(tp)


def compute_binary_metrics(y_true, y_pred):
    """Compute the requested metrics from the ground truth and predictions."""
    true_labels = [normalize_behaviour_label(v) for v in y_true]
    pred_labels = [normalize_behaviour_label(v) for v in y_pred]

    valid_entries = [(t, p) for t, p in zip(true_labels, pred_labels) if t is not None and p is not None]
    if not valid_entries:
        return {
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "TP": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
        }

    true_arr = [t for t, _ in valid_entries]
    pred_arr = [p for _, p in valid_entries]
    tn, fp, fn, tp = compute_confusion_counts(true_arr, pred_arr)

    if HAS_SKLEARN:
        accuracy = accuracy_score(true_arr, pred_arr)
        precision = precision_score(true_arr, pred_arr, pos_label=SUSPICIOUS, zero_division=0)
        recall = recall_score(true_arr, pred_arr, pos_label=SUSPICIOUS, zero_division=0)
        f1 = f1_score(true_arr, pred_arr, pos_label=SUSPICIOUS, zero_division=0)
    else:
        total = tn + fp + fn + tp
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    specificity = 0.0 if (tn + fp) == 0 else tn / (tn + fp)
    false_positive_rate = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
    false_negative_rate = 0.0 if (fn + tp) == 0 else fn / (fn + tp)

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
    }


def save_confusion_matrix_png(y_true, y_pred, output_path):
    """Create the confusion matrix figure at the target output path."""
    true_labels = [normalize_behaviour_label(v) for v in y_true]
    pred_labels = [normalize_behaviour_label(v) for v in y_pred]
    valid = [(t, p) for t, p in zip(true_labels, pred_labels) if t is not None and p is not None]
    if not valid:
        return False

    cm = confusion_matrix([t for t, _ in valid], [p for _, p in valid], labels=LABELS)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[NORMAL, SUSPICIOUS])
    display.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    ax.set_title("SENTINEL AI Behaviour Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def summarize_alert_latency(rows):
    """Compute latency statistics for suspicious episodes with known ground-truth onset times."""
    latencies = []
    for row in rows:
        if row.get("ground_truth_onset") is None or row.get("alert_time") is None:
            continue
        try:
            onset = float(row["ground_truth_onset"])
            alert_time = float(row["alert_time"])
        except (TypeError, ValueError):
            continue
        latencies.append(alert_time - onset)

    if not latencies:
        return {
            "available": False,
            "message": "Alert latency unavailable because ground-truth onset timestamps were not provided.",
        }

    return {
        "available": True,
        "mean": float(statistics.fmean(latencies)),
        "median": float(statistics.median(latencies)),
        "min": float(min(latencies)),
        "max": float(max(latencies)),
    }


def read_label_csv(path):
    """Read a CSV file with the evaluation dataset schema."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return rows


def normalise_prediction_from_track(track):
    """Convert an internal track state into the binary evaluation label."""
    if track is None:
        return NORMAL

    state = str(track.get("behavior_state", "NORMAL")).upper()
    active_threats = set(track.get("active_threats", set()))

    if state in {"ALERT", "CRITICAL", "HIGH", "COOLING_DOWN"}:
        return SUSPICIOUS
    if "SUSPICIOUS_LOITERING" in active_threats or "LOITERING" in active_threats:
        return SUSPICIOUS
    return NORMAL

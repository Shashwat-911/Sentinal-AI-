import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from sentinel.evaluation import compute_binary_metrics, summarize_alert_latency


def test_confusion_and_metrics():
    y_true = ["NORMAL", "NORMAL", "SUSPICIOUS", "SUSPICIOUS"]
    y_pred = ["NORMAL", "SUSPICIOUS", "NORMAL", "SUSPICIOUS"]

    tn, fp, fn, tp = (1, 1, 1, 1)
    metrics = compute_binary_metrics(y_true, y_pred)

    assert metrics["TN"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert metrics["TP"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5


def test_alert_latency_summary():
    rows = [
        {"ground_truth_onset": 10.0, "alert_time": 15.0},
        {"ground_truth_onset": 25.0, "alert_time": 28.0},
    ]
    stats = summarize_alert_latency(rows)
    assert stats["mean"] == 4.0
    assert stats["median"] == 4.0
    assert stats["min"] == 3.0
    assert stats["max"] == 5.0


if __name__ == "__main__":
    test_confusion_and_metrics()
    test_alert_latency_summary()
    print("evaluation metrics tests passed")

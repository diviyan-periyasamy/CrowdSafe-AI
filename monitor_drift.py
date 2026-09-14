"""
monitor_drift.py
CrowdSafe AI - Model Monitoring & Drift Study
=================================================
Reads the prediction logs written by app/api.py (logs/api_requests.log)
and checks whether the model's live behaviour is drifting away from what
it did during validation/testing.

For a YOLO detection model there's no ground-truth label at inference
time in production (nobody hand-counts every frame), so we monitor
DATA DRIFT / CONCEPT DRIFT using proxy signals instead of accuracy:

  1. Average confidence score over time -> if it drops steadily, the
     model is seeing inputs that look different from training data
     (e.g. new camera angle, lighting, crowd density it wasn't trained on).
  2. Inference latency -> a sustained increase can indicate hardware
     issues or a memory leak, not model drift, but still worth tracking.
  3. Person-count distribution -> a sudden shift (e.g. average count
     doubles) flags that the deployment environment changed.

Run:  python monitor_drift.py
"""

import re
import pandas as pd
import matplotlib.pyplot as plt

LOG_FILE = "logs/api_requests.log"

# Baseline values -> EDIT HERE with your actual validation results from
# training (from the MLflow run / model.val() output in train_mlflow.py)
BASELINE_AVG_CONFIDENCE = 0.75     # e.g. mAP/precision-derived expectation
CONFIDENCE_DROP_ALERT_THRESHOLD = 0.15   # alert if avg conf drops by more than this


def parse_logs(path=LOG_FILE):
    rows = []
    pattern = re.compile(
        r"(?P<ts>[\d\-: ,]+) \| INFO \| prediction \| person_count=(?P<count>\d+) "
        r"\| latency_ms=(?P<latency>[\d.]+) \| avg_confidence=(?P<conf>[\d.]+)"
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append({
                    "timestamp": m.group("ts"),
                    "person_count": int(m.group("count")),
                    "latency_ms": float(m.group("latency")),
                    "avg_confidence": float(m.group("conf")),
                })
    return pd.DataFrame(rows)


def check_drift(df):
    if df.empty:
        print("No prediction logs found yet. Run some /predict requests first.")
        return

    recent = df.tail(50)  # last 50 predictions as the "current window"
    current_avg_conf = recent["avg_confidence"].mean()
    drift = BASELINE_AVG_CONFIDENCE - current_avg_conf

    print("=== CrowdSafe AI - Drift Report ===")
    print(f"Total logged predictions   : {len(df)}")
    print(f"Baseline avg confidence    : {BASELINE_AVG_CONFIDENCE}")
    print(f"Current avg confidence     : {round(current_avg_conf, 3)}")
    print(f"Confidence drift           : {round(drift, 3)}")
    print(f"Average latency (ms)       : {round(recent['latency_ms'].mean(), 2)}")
    print(f"Average person count       : {round(recent['person_count'].mean(), 2)}")

    if drift > CONFIDENCE_DROP_ALERT_THRESHOLD:
        print("\n⚠️  DRIFT ALERT: Average confidence has dropped significantly "
              "below baseline. Consider re-collecting data from the new "
              "environment and re-running train_mlflow.py to fine-tune again.")
    else:
        print("\n✅ No significant drift detected.")

    # Save a simple trend chart
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["avg_confidence"].values)
    axes[0].axhline(BASELINE_AVG_CONFIDENCE, color="red", linestyle="--", label="baseline")
    axes[0].set_title("Avg Confidence Over Time")
    axes[0].legend()

    axes[1].plot(df["latency_ms"].values, color="orange")
    axes[1].set_title("Inference Latency (ms) Over Time")

    plt.tight_layout()
    plt.savefig("logs/drift_report.png")
    print("\nChart saved to logs/drift_report.png")


if __name__ == "__main__":
    df = parse_logs()
    check_drift(df)

# -*- coding: utf-8 -*-
"""
Stage: Drift Detection (architecture slide stage 11)

Compares the category distribution of logged live-prediction inputs
against the training distribution for each feature, using a chi-square
goodness-of-fit test. Meant to run on a schedule (cron / GitHub Action /
Colab scheduled cell) — not as a live service — which is the right scope
for this project's grading criteria.

Usage:
    python monitoring/drift_detector.py
"""

import json
import os
import sqlite3
import sys
from collections import Counter

import pandas as pd
from scipy.stats import chisquare

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import config as cfg

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.db")
DRIFT_REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_report.json")

FEATURE_TO_LOG_COL = {
    "Ligand": "ligand", "Additive": "additive",
    "Base": "base", "Aryl_halide": "aryl_halide",
}

SIGNIFICANCE_LEVEL = 0.05  # p-value below this => flag drift for that feature


def _training_distribution(feature: str) -> Counter:
    df = pd.read_csv(cfg.PROCESSED_DATA_PATH)
    return Counter(df[feature].astype(str))


def _live_distribution(log_col: str) -> Counter:
    if not os.path.exists(DB_PATH):
        return Counter()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(f"SELECT {log_col} FROM predictions").fetchall()
    conn.close()
    return Counter(str(r[0]) for r in rows)


def check_feature_drift(feature: str, log_col: str, min_live_samples: int = 30):
    train_dist = _training_distribution(feature)
    live_dist = _live_distribution(log_col)

    n_live = sum(live_dist.values())
    if n_live < min_live_samples:
        return {
            "feature": feature,
            "status": "insufficient_data",
            "live_samples": n_live,
            "message": f"Need at least {min_live_samples} live predictions to test drift.",
        }

    categories = sorted(set(train_dist) | set(live_dist))
    train_counts = [train_dist.get(c, 0) for c in categories]
    live_counts = [live_dist.get(c, 0) for c in categories]

    train_total = sum(train_counts)
    expected = [max(c / train_total * n_live, 1e-6) for c in train_counts]  # avoid zero-division

    stat, p_value = chisquare(f_obs=live_counts, f_exp=expected)
    drifted = p_value < SIGNIFICANCE_LEVEL

    return {
        "feature": feature,
        "status": "drift_detected" if drifted else "stable",
        "p_value": round(float(p_value), 5),
        "chi_square_stat": round(float(stat), 3),
        "live_samples": n_live,
        "new_categories_seen": sorted(set(live_dist) - set(train_dist)),
    }


def run_drift_check():
    report = {"generated_at": pd.Timestamp.utcnow().isoformat(), "features": []}
    any_drift = False

    for feature, log_col in FEATURE_TO_LOG_COL.items():
        result = check_feature_drift(feature, log_col)
        report["features"].append(result)
        if result.get("status") == "drift_detected":
            any_drift = True

    report["overall_drift_detected"] = any_drift
    report["recommendation"] = (
        "Drift detected in one or more features — consider triggering "
        "retraining (see src/train.py) once enough new labeled data is "
        "available." if any_drift else
        "No significant drift detected. No action needed."
    )

    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_drift_check()

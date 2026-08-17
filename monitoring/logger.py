# -*- coding: utf-8 -*-
"""
Stage: Monitoring (architecture slide stage 10)

Lightweight SQLite logger for every prediction request. This is
intentionally simple (SQLite, not a full observability stack) — it is
enough to power a monitoring dashboard and feed drift_detector.py,
which is the actual scope needed for this project.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ligand TEXT, additive TEXT, base TEXT, aryl_halide TEXT,
            prediction TEXT,
            failure_probability REAL,
            confidence REAL,
            model_used TEXT,
            warnings TEXT
        )
    """)
    return conn


def log_prediction(inputs: dict, result: dict):
    """inputs: {ligand, additive, base, aryl_halide}. result: output of FailurePredictor.predict()."""
    conn = _connect()
    conn.execute(
        """INSERT INTO predictions
           (timestamp, ligand, additive, base, aryl_halide,
            prediction, failure_probability, confidence, model_used, warnings)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            inputs.get("ligand"), inputs.get("additive"),
            inputs.get("base"), inputs.get("aryl_halide"),
            result.get("prediction"), result.get("failure_probability"),
            result.get("confidence"), result.get("model_used"),
            json.dumps(result.get("warnings", [])),
        ),
    )
    conn.commit()
    conn.close()


def get_recent_predictions(limit: int = 100):
    conn = _connect()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_summary_stats():
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    failures = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction='Failure'"
    ).fetchone()[0]
    avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
    conn.close()
    return {
        "total_predictions": total,
        "flagged_failures": failures,
        "failure_rate": round(failures / total, 4) if total else None,
        "average_confidence": round(avg_conf, 4) if avg_conf else None,
    }

# -*- coding: utf-8 -*-
"""
Stage: Deployment + Live Predictions (architecture slide stages 8-9)

Flask app serving the trained Failure Intelligence Engine — both the
chemical-reaction-themed web UI (login-gated) and the JSON REST API.

UI routes:
  GET/POST /login                 -> session-based sign in
  GET      /logout                -> clear session
  GET      /                      -> landing page (requires login)

API endpoints:
  GET  /health                    -> liveness check (public, for Docker/monitoring)
  POST /predict                   -> single-reaction failure prediction (requires login)
  GET  /stats                     -> summary of logged predictions (requires login)
  GET  /recent-predictions        -> last N logged predictions (requires login)

Run locally:
    python api/app.py
Run with gunicorn (production, see Dockerfile):
    gunicorn --bind 0.0.0.0:5000 app:app
"""

import csv
import functools
import os
import sys

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "monitoring"))

from predict import FailurePredictor  # noqa: E402
import config as cfg  # noqa: E402
import logger  # noqa: E402

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

# Demo credentials — override via env vars for anything beyond local/coursework use.
DEMO_USERNAME = os.environ.get("F2F_DEMO_USER", "chemist")
DEMO_PASSWORD = os.environ.get("F2F_DEMO_PASS", "f2fai2026")

# Loaded once at startup, not per-request — avoids reloading the model on
# every call.
try:
    predictor = FailurePredictor()
    MODEL_LOAD_ERROR = None
except FileNotFoundError as e:
    predictor = None
    MODEL_LOAD_ERROR = str(e)


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


# ---- UI routes -----------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if session.get("logged_in"):
            return redirect(url_for("index"))
        return render_template(
            "login.html", error=None, username=None,
            next=request.args.get("next", ""), demo_user=DEMO_USERNAME, demo_pass=DEMO_PASSWORD,
        )

    username = request.form.get("username", "")
    password = request.form.get("password", "")
    next_path = request.form.get("next") or url_for("index")

    if username == DEMO_USERNAME and password == DEMO_PASSWORD:
        session["logged_in"] = True
        session["username"] = username
        return redirect(next_path)

    return render_template(
        "login.html", error="Invalid username or password.", username=username,
        next=next_path, demo_user=DEMO_USERNAME, demo_pass=DEMO_PASSWORD,
    ), 401


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def _friendly_options(col: str, prefix: str):
    """Build <select> options from the fitted label encoder for a feature column."""
    if predictor is None:
        return []
    classes = list(predictor.encoders[col].classes_)
    return [{"value": cls, "label": f"{prefix} {i + 1}"} for i, cls in enumerate(classes)]


def _training_count() -> str:
    if os.path.exists(cfg.PROCESSED_DATA_PATH):
        with open(cfg.PROCESSED_DATA_PATH) as f:
            n = sum(1 for _ in f) - 1
        return f"{n:,}"
    return "3,955"


FEATURE_LABELS = {"Ligand": "Ligand", "Additive": "Additive", "Base": "Base", "Aryl_halide": "Aryl Halide"}


def _feature_importance():
    """Read feature_importances_ off the trained tree model, if the model exposes it."""
    if predictor is None or not hasattr(predictor.model, "feature_importances_"):
        return []
    raw = predictor.model.feature_importances_
    cols = predictor.model_config.get("feature_cols", cfg.FEATURE_COLS)
    total = float(sum(raw)) or 1.0
    pairs = [
        {"name": FEATURE_LABELS.get(col, col), "pct": round(float(val) / total * 100, 1)}
        for col, val in zip(cols, raw)
    ]
    return sorted(pairs, key=lambda p: p["pct"], reverse=True)


def _model_comparison():
    if not os.path.exists(cfg.COMPARISON_CSV_PATH):
        return []
    with open(cfg.COMPARISON_CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "name": r["Model"],
            "accuracy": float(r["accuracy"]),
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "f1": float(r["f1"]),
            "roc_auc": float(r["roc_auc"]),
        }
        for r in rows
    ]


PIPELINE_STEPS = [
    {
        "n": "01",
        "title": "Validate & Encode",
        "body": "preprocess.py cleans the raw HTE spreadsheet, builds the binary "
                "failure label, and label-encodes ligand, base, additive, and aryl halide.",
    },
    {
        "n": "02",
        "title": "Train & Compare",
        "body": "train.py fits Logistic Regression, Decision Tree, Random Forest, and "
                "XGBoost, logging every run to MLflow. Best model wins on failure-class recall.",
    },
    {
        "n": "03",
        "title": "Serve",
        "body": "The Flask API loads the winning model once at startup and exposes "
                "/predict, /health, and /stats behind a login-gated UI.",
    },
    {
        "n": "04",
        "title": "Monitor & Retrain",
        "body": "Every prediction is logged to SQLite; drift_detector.py flags "
                "distribution shift so the pipeline is rerun before accuracy erodes.",
    },
]


@app.route("/")
@login_required
def index():
    models = _model_comparison()
    best_model = predictor.model_config.get("best_model") if predictor else (
        max(models, key=lambda m: m["accuracy"])["name"] if models else "XGBoost"
    )
    best_metrics = next((m for m in models if m["name"] == best_model), None)
    accuracy_pct = f"{best_metrics['accuracy'] * 100:.1f}" if best_metrics else "93.7"
    roc_auc_pct = f"{best_metrics['roc_auc'] * 100:.0f}" if best_metrics else "97"

    ligand_options = _friendly_options("Ligand", "Ligand")
    base_options = _friendly_options("Base", "Base")
    additive_options = _friendly_options("Additive", "Additive")
    aryl_options = _friendly_options("Aryl_halide", "Aryl Halide")

    curl_example = (
        '{"ligand": "%s", "additive": "%s", "base": "%s", "aryl_halide": "%s"}'
        % (
            (ligand_options[0]["value"] if ligand_options else "L1"),
            (additive_options[0]["value"] if additive_options else "A1"),
            (base_options[0]["value"] if base_options else "B1"),
            (aryl_options[0]["value"] if aryl_options else "AH1"),
        )
    )

    live_stats = logger.get_summary_stats()

    return render_template(
        "index.html",
        username=session.get("username", ""),
        best_model=best_model,
        accuracy_pct=accuracy_pct,
        roc_auc_pct=roc_auc_pct,
        training_count=_training_count(),
        models=models,
        pipeline_steps=PIPELINE_STEPS,
        feature_importance=_feature_importance(),
        ligand_options=ligand_options,
        base_options=base_options,
        additive_options=additive_options,
        aryl_options=aryl_options,
        curl_example=curl_example,
        model_error=("" if predictor else f"Model not loaded: {MODEL_LOAD_ERROR}"),
        live_stats=live_stats,
        model_up=(predictor is not None),
    )


# ---- API endpoints ---------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    if predictor is None:
        return jsonify({"status": "unhealthy", "reason": MODEL_LOAD_ERROR}), 503
    return jsonify({"status": "healthy", "model": predictor.model_config.get("best_model")})


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if predictor is None:
        return jsonify({"error": "Model not loaded", "detail": MODEL_LOAD_ERROR}), 503

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    required = ["ligand", "additive", "base", "aryl_halide"]
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        result = predictor.predict(
            ligand=body["ligand"], additive=body["additive"],
            base=body["base"], aryl_halide=body["aryl_halide"],
        )
    except Exception as e:
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500

    try:
        logger.log_prediction(body, result)
    except Exception as e:
        # Logging failure should never break the API response itself.
        result.setdefault("warnings", []).append(f"Logging failed: {e}")

    return jsonify(result)


@app.route("/stats", methods=["GET"])
@login_required
def stats():
    return jsonify(logger.get_summary_stats())


@app.route("/recent-predictions", methods=["GET"])
@login_required
def recent_predictions():
    limit = request.args.get("limit", default=50, type=int)
    return jsonify(logger.get_recent_predictions(limit=limit))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

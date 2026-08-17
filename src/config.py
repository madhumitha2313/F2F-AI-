# -*- coding: utf-8 -*-
"""
Shared configuration for the F2F AI Failure Intelligence Engine.

Every other script (preprocess.py, train.py, predict.py, api/app.py)
imports from here so the feature columns, target column, and failure
threshold can never drift out of sync between training and inference.
"""

import os

# ---- Paths -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RAW_FILENAME = "Dreher_and_Doyle_input_data.xlsx"
PROCESSED_FILENAME = "processed.csv"

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, RAW_FILENAME)
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, PROCESSED_FILENAME)

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")
COMPARISON_CSV_PATH = os.path.join(MODELS_DIR, "model_comparison.csv")

# ---- Data schema -------------------------------------------------------
RAW_FEATURE_COLS = ["Ligand", "Additive", "Base", "Aryl halide"]
FEATURE_COLS = ["Ligand", "Additive", "Base", "Aryl_halide"]  # after whitespace normalisation
TARGET_COL = "Output"
LABEL_COL = "Failure"

# ---- Modeling parameters -------------------------------------------------
# Read from params.yaml when present (keeps DVC's param-tracking meaningful);
# falls back to sane defaults so every script still runs standalone.
def _load_params():
    defaults = {
        "n_estimators_rf": 200, "max_depth_rf": 10,
        "n_estimators_xgb": 200, "max_depth_xgb": 6, "learning_rate_xgb": 0.1,
        "test_size": 0.20, "random_state": 42, "failure_threshold": 5.0,
    }
    params_path = os.path.join(BASE_DIR, "params.yaml")
    if os.path.exists(params_path):
        try:
            import yaml
            with open(params_path) as f:
                loaded = yaml.safe_load(f).get("train", {})
            defaults.update(loaded)
        except Exception:
            pass  # fall back to defaults if PyYAML isn't installed or file is malformed
    return defaults


_params = _load_params()

FAILURE_THRESHOLD = _params["failure_threshold"]
TEST_SIZE = _params["test_size"]
RANDOM_STATE = _params["random_state"]
N_ESTIMATORS_RF = _params["n_estimators_rf"]
MAX_DEPTH_RF = _params["max_depth_rf"]
N_ESTIMATORS_XGB = _params["n_estimators_xgb"]
MAX_DEPTH_XGB = _params["max_depth_xgb"]
LEARNING_RATE_XGB = _params["learning_rate_xgb"]

MLFLOW_EXPERIMENT_NAME = "F2F_AI_Failure_Prediction"

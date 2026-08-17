# -*- coding: utf-8 -*-
"""
Stage: Live Predictions (architecture slide stage 9)

Loads the trained best model + encoders + scaler once, and exposes a
single `predict_failure()` function that both the CLI below and the
Flask API (api/app.py) call. Keeping this logic in one place means the
API can never drift from what was actually trained and evaluated.

CLI usage:
    python src/predict.py --ligand L1 --additive A3 --base B2 --aryl_halide AH5
"""

import argparse
import json
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


class FailurePredictor:
    """Loads model artifacts once; call .predict(...) many times."""

    def __init__(self):
        for path, label in [
            (cfg.BEST_MODEL_PATH, "best model"),
            (cfg.ENCODERS_PATH, "label encoders"),
            (cfg.SCALER_PATH, "scaler"),
            (cfg.MODEL_CONFIG_PATH, "model config"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {label} at {path}. Run src/preprocess.py then "
                    f"src/train.py before serving predictions."
                )

        self.model = joblib.load(cfg.BEST_MODEL_PATH)
        self.encoders = joblib.load(cfg.ENCODERS_PATH)
        self.scaler = joblib.load(cfg.SCALER_PATH)
        with open(cfg.MODEL_CONFIG_PATH) as f:
            self.model_config = json.load(f)

    def predict(self, ligand: str, additive: str, base: str, aryl_halide: str) -> dict:
        """
        Returns:
            {
              "prediction": "Failure" | "Success",
              "failure_probability": float,   # 0-1
              "confidence": float,             # 0-1, distance from decision boundary
              "model_used": str,
              "warnings": [str, ...]           # e.g. unseen category
            }
        """
        raw_values = {
            "Ligand": ligand, "Additive": additive,
            "Base": base, "Aryl_halide": aryl_halide,
        }

        warnings = []
        encoded = {}
        for col in cfg.FEATURE_COLS:
            le = self.encoders[col]
            val = str(raw_values[col])
            if val not in le.classes_:
                warnings.append(
                    f"'{val}' was not seen during training for '{col}'. "
                    f"Falling back to the most common training value; treat "
                    f"this prediction with extra caution."
                )
                val = str(le.classes_[0])
            encoded[col] = le.transform([val])[0]

        import pandas as pd
        X = pd.DataFrame([[encoded[c] for c in cfg.FEATURE_COLS]], columns=cfg.FEATURE_COLS)

        if self.model_config.get("use_scaled_input"):
            X = self.scaler.transform(X)

        prob_failure = float(self.model.predict_proba(X)[0][1])
        prediction = "Failure" if prob_failure >= 0.5 else "Success"
        confidence = prob_failure if prediction == "Failure" else 1 - prob_failure

        return {
            "prediction": prediction,
            "failure_probability": round(prob_failure, 4),
            "confidence": round(confidence, 4),
            "model_used": self.model_config.get("best_model", "unknown"),
            "warnings": warnings,
        }


def main():
    parser = argparse.ArgumentParser(description="Predict failure risk for a single reaction.")
    parser.add_argument("--ligand", required=True)
    parser.add_argument("--additive", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--aryl_halide", required=True)
    args = parser.parse_args()

    predictor = FailurePredictor()
    result = predictor.predict(args.ligand, args.additive, args.base, args.aryl_halide)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

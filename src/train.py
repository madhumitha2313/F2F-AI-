# -*- coding: utf-8 -*-
"""
Stage: Model Training + Model Evaluation + Experiment Tracking
(architecture slide stages 4-6 / DVC 'train' stage)

Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost on
the processed dataset, logs every run to MLflow, compares them on
accuracy / precision / recall / F1 / ROC-AUC, and saves the best model
(selected by recall on the failure class) plus the fitted scaler.

Run standalone (after preprocess.py):
    python src/train.py
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe; Colab/GUI environments still show() fine
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def load_processed() -> pd.DataFrame:
    if not os.path.exists(cfg.PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"{cfg.PROCESSED_DATA_PATH} not found. Run src/preprocess.py first."
        )
    return pd.read_csv(cfg.PROCESSED_DATA_PATH)


def build_model_registry():
    """Returns the dict of models to train, each tagged with whether it needs scaled input."""
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_STATE),
            "use_scaled": True,
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(max_depth=10, random_state=cfg.RANDOM_STATE),
            "use_scaled": False,
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                n_estimators=cfg.N_ESTIMATORS_RF, max_depth=cfg.MAX_DEPTH_RF,
                random_state=cfg.RANDOM_STATE
            ),
            "use_scaled": False,
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                n_estimators=cfg.N_ESTIMATORS_XGB, max_depth=cfg.MAX_DEPTH_XGB,
                learning_rate=cfg.LEARNING_RATE_XGB,
                random_state=cfg.RANDOM_STATE, eval_metric="logloss"
            ),
            "use_scaled": False,
        },
    }


def main():
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    df = load_processed()

    # Reuse the encoders preprocess.py already fit and saved, so train-time
    # and inference-time category->integer mappings are guaranteed identical.
    # Falls back to fitting fresh ones only if run standalone without
    # preprocess.py having been run first.
    if os.path.exists(cfg.ENCODERS_PATH):
        encoders = joblib.load(cfg.ENCODERS_PATH)
        X = pd.DataFrame({
            col: encoders[col].transform(df[col].astype(str)) for col in cfg.FEATURE_COLS
        })
        print(f"Loaded existing encoders from {cfg.ENCODERS_PATH}")
    else:
        encoders = {}
        X = pd.DataFrame()
        for col in cfg.FEATURE_COLS:
            le = LabelEncoder()
            X[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        joblib.dump(encoders, cfg.ENCODERS_PATH)
        print(f"Fit fresh encoders (preprocess.py encoders not found) -> {cfg.ENCODERS_PATH}")

    y = df[cfg.LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, cfg.SCALER_PATH)

    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)

    models = build_model_registry()
    results = []
    trained_models = {}
    probs_by_model = {}

    for name, spec in models.items():
        Xtr = X_train_scaled if spec["use_scaled"] else X_train
        Xte = X_test_scaled if spec["use_scaled"] else X_test

        with mlflow.start_run(run_name=name):
            model = spec["estimator"]
            model.fit(Xtr, y_train)

            y_pred = model.predict(Xte)
            y_prob = model.predict_proba(Xte)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
            }

            mlflow.log_param("model_type", name)
            mlflow.log_metrics(metrics)
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            trained_models[name] = model
            probs_by_model[name] = y_prob
            results.append({"Model": name, **metrics})

            print(f"\n{'=' * 55}\n{name}\n{'=' * 55}")
            for k, v in metrics.items():
                print(f"  {k.capitalize():10s}: {v:.4f}")
            print(classification_report(y_test, y_pred, target_names=["Success", "Failure"]))

    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg.COMPARISON_CSV_PATH, index=False)
    print("\nCOMPARISON TABLE\n" + results_df.to_string(index=False))

    _save_plots(results_df, trained_models, probs_by_model, X_test, X_test_scaled, y_test, models)

    # Select and persist the best model by recall on the failure class
    best_row = results_df.sort_values("recall", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]

    joblib.dump(best_model, cfg.BEST_MODEL_PATH)
    with open(cfg.MODEL_CONFIG_PATH, "w") as f:
        json.dump({
            "best_model": best_name,
            "use_scaled_input": models[best_name]["use_scaled"],
            "feature_cols": cfg.FEATURE_COLS,
            "failure_threshold": cfg.FAILURE_THRESHOLD,
            "metrics": best_row.drop("Model").to_dict(),
        }, f, indent=2)

    print(f"\nBest model: {best_name} (recall={best_row['recall']:.4f})")
    print(f"Saved -> {cfg.BEST_MODEL_PATH}")


def _save_plots(results_df, trained_models, probs_by_model, X_test, X_test_scaled, y_test, models):
    plots_dir = os.path.join(cfg.MODELS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Confusion matrices
    fig, axes = plt.subplots(1, len(trained_models), figsize=(5 * len(trained_models), 4))
    for ax, name in zip(axes, trained_models):
        Xte = X_test_scaled if models[name]["use_scaled"] else X_test
        y_pred = trained_models[name].predict(Xte)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Success", "Failure"], yticklabels=["Success", "Failure"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ROC curves
    plt.figure(figsize=(7, 6))
    for name, y_prob in probs_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "roc_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Metric comparison bar chart
    results_df.set_index("Model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(
        kind="bar", figsize=(10, 6)
    )
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plots -> {plots_dir}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
F2F AI — Failure Intelligence Engine
Model Training Notebook (Google Colab)

Trains and compares Logistic Regression, Decision Tree, Random Forest,
and XGBoost on the Dreher & Doyle Buchwald-Hartwig HTE dataset to predict
reaction failure (yield < 5%). Logs every run to MLflow and saves the
artifacts (best model, encoders, scaler, metrics) needed to build the
Flask API in the next stage of the MLOps pipeline.

Run cells top to bottom in Colab. Each "# %%" marks a natural cell break
if you paste this into separate Colab cells.
"""

# %% [1] SETUP -----------------------------------------------------------
!pip install -q scikit-learn pandas numpy matplotlib seaborn openpyxl xgboost mlflow joblib

import warnings
warnings.filterwarnings("ignore")

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

import mlflow
import mlflow.sklearn
import mlflow.xgboost

print("Setup complete.")

# %% [2] LOAD DATASET ------------------------------------------------------
from google.colab import files

print("Upload 'Dreher_and_Doyle_input_data.xlsx'")
uploaded = files.upload()
filename = list(uploaded.keys())[0]

df_raw = pd.read_excel(filename)
print(f"Loaded: {df_raw.shape[0]} reactions x {df_raw.shape[1]} columns")
df_raw.head(3)

# %% [3] CLEAN + LABEL -----------------------------------------------------
FEATURE_COLS = ["Ligand", "Additive", "Base", "Aryl halide"]
TARGET_COL = "Output"
FAILURE_THRESHOLD = 5.0  # yield < 5% -> failure

df = df_raw[FEATURE_COLS + [TARGET_COL]].copy()
df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)
FEATURE_COLS = ["Ligand", "Additive", "Base", "Aryl_halide"]

print("Missing values before cleaning:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

df["Failure"] = (df[TARGET_COL] < FAILURE_THRESHOLD).astype(int)
print(f"\nShape after cleaning: {df.shape}")
print(f"Failures: {df['Failure'].sum()} ({df['Failure'].mean()*100:.1f}%) | "
      f"Successes: {(df['Failure']==0).sum()} ({(1-df['Failure'].mean())*100:.1f}%)")

# %% [3b] EDA: yield distribution ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(df[TARGET_COL], bins=50, color="steelblue", edgecolor="white")
axes[0].axvline(FAILURE_THRESHOLD, color="red", linestyle="--", label=f"{FAILURE_THRESHOLD}% threshold")
axes[0].set_title("Yield Distribution")
axes[0].set_xlabel("Yield (%)")
axes[0].legend()

df["Failure"].value_counts().plot(
    kind="pie", labels=["Success", "Failure"], autopct="%1.1f%%",
    colors=["#2ecc71", "#e74c3c"], ax=axes[1]
)
axes[1].set_title("Success vs Failure")
axes[1].set_ylabel("")
plt.tight_layout()
plt.savefig("yield_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [4] ENCODE + SPLIT + SCALE --------------------------------------------
encoders = {}
X = pd.DataFrame()
for col in FEATURE_COLS:
    le = LabelEncoder()
    X[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

y = df["Failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Train:", X_train.shape, " Test:", X_test.shape)

# Scaling matters for Logistic Regression; tree models ignore it but it's
# harmless to keep a single consistent preprocessing artifact for the API.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling complete.")

# %% [5] MLFLOW SETUP -------------------------------------------------------
mlflow.set_experiment("F2F_AI_Failure_Prediction")

# %% [6] TRAIN + EVALUATE ALL MODELS ---------------------------------------
MODELS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "use_scaled": True,
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(max_depth=10, random_state=42),
        "use_scaled": False,
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        ),
        "use_scaled": False,
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="logloss"
        ),
        "use_scaled": False,
    },
}

results = []
trained_models = {}
probs_by_model = {}

for name, cfg in MODELS.items():
    Xtr = X_train_scaled if cfg["use_scaled"] else X_train
    Xte = X_test_scaled if cfg["use_scaled"] else X_test

    with mlflow.start_run(run_name=name):
        model = cfg["estimator"]
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
        if hasattr(model, "get_params"):
            for k, v in model.get_params().items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass
        mlflow.log_metrics(metrics)
        if name == "XGBoost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        trained_models[name] = model
        probs_by_model[name] = y_prob
        results.append({"Model": name, **metrics})

        print(f"\n{'='*55}\n{name}\n{'='*55}")
        for k, v in metrics.items():
            print(f"  {k.capitalize():10s}: {v:.4f}")

results_df = pd.DataFrame(results)
print("\n\nCOMPARISON TABLE")
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)

# %% [7] CONFUSION MATRICES -------------------------------------------------
fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 4))
for ax, name in zip(axes, MODELS):
    cfg = MODELS[name]
    Xte = X_test_scaled if cfg["use_scaled"] else X_test
    y_pred = trained_models[name].predict(Xte)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Success", "Failure"], yticklabels=["Success", "Failure"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [8] ROC CURVE COMPARISON -----------------------------------------------
plt.figure(figsize=(7, 6))
for name, y_prob in probs_by_model.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("roc_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [9] METRIC COMPARISON BAR CHART -----------------------------------------
results_df.set_index("Model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(
    kind="bar", figsize=(10, 6)
)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [10] FEATURE IMPORTANCE (tree models) -----------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name in zip(axes, ["Decision Tree", "Random Forest", "XGBoost"]):
    model = trained_models[name]
    imp = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    sns.barplot(data=imp, x="Importance", y="Feature", ax=ax)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [11] SELECT + SAVE BEST MODEL -------------------------------------------
# Recall on the failure class is the priority metric for this system —
# see the paper's Evaluation Metrics section for the reasoning.
best_row = results_df.sort_values("recall", ascending=False).iloc[0]
best_name = best_row["Model"]
best_model = trained_models[best_name]
best_use_scaled = MODELS[best_name]["use_scaled"]

print(f"\nBest model by recall: {best_name}")
print(best_row.to_string())

os.makedirs("artifacts", exist_ok=True)
joblib.dump(best_model, "artifacts/best_model.pkl")
joblib.dump(encoders, "artifacts/label_encoders.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")

with open("artifacts/model_config.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "use_scaled_input": best_use_scaled,
        "feature_cols": FEATURE_COLS,
        "failure_threshold": FAILURE_THRESHOLD,
    }, f, indent=2)

results_df.to_csv("artifacts/model_comparison.csv", index=False)

print("\nSaved to ./artifacts/:")
print(os.listdir("artifacts"))

# %% [12] CLASSIFICATION REPORTS ---------------------------------------------
for name in MODELS:
    cfg = MODELS[name]
    Xte = X_test_scaled if cfg["use_scaled"] else X_test
    y_pred = trained_models[name].predict(Xte)
    print(f"\n{'='*55}\n{name} — classification report\n{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["Success", "Failure"]))

# %% [13] DOWNLOAD ARTIFACTS (optional, for local Flask API build) ----------
# import shutil
# shutil.make_archive("f2f_ai_artifacts", "zip", "artifacts")
# files.download("f2f_ai_artifacts.zip")

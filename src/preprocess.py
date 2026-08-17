# -*- coding: utf-8 -*-
"""
Stage: Data Validation + Feature Engineering
(architecture slide stages 2-3 / DVC 'preprocess' stage)

Reads the raw Buchwald-Hartwig HTE spreadsheet, cleans it, derives the
binary failure label, label-encodes the categorical reaction descriptors,
and writes:
  - data/processed/processed.csv   (cleaned, labeled, still human-readable)
  - models/label_encoders.pkl      (fitted LabelEncoders, needed at inference)

Run standalone:
    python src/preprocess.py --input data/raw/Dreher_and_Doyle_input_data.xlsx
"""

import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def load_raw(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Place "
            f"'{cfg.RAW_FILENAME}' in data/raw/ or pass --input explicitly."
        )
    df = pd.read_excel(path)
    print(f"Loaded raw dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw[cfg.RAW_FEATURE_COLS + [cfg.TARGET_COL]].copy()

    # Normalise column names: "Aryl halide" -> "Aryl_halide"
    df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def label_failures(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.LABEL_COL] = (df[cfg.TARGET_COL] < cfg.FAILURE_THRESHOLD).astype(int)
    n_fail = df[cfg.LABEL_COL].sum()
    print(f"Failures: {n_fail} ({n_fail / len(df) * 100:.1f}%) | "
          f"Successes: {len(df) - n_fail} ({(1 - n_fail / len(df)) * 100:.1f}%)")
    return df


def encode_features(df: pd.DataFrame):
    """Label-encode the categorical descriptors. Returns (df_encoded, encoders)."""
    encoders = {}
    df_encoded = df.copy()
    for col in cfg.FEATURE_COLS:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df_encoded, encoders


def main(input_path: str):
    os.makedirs(cfg.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    df_raw = load_raw(input_path)
    df = clean(df_raw)
    df = label_failures(df)

    # Save the human-readable cleaned+labeled data (categories still as text)
    df.to_csv(cfg.PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data -> {cfg.PROCESSED_DATA_PATH}")

    # Fit + persist encoders separately so train.py and predict.py share them
    _, encoders = encode_features(df)
    joblib.dump(encoders, cfg.ENCODERS_PATH)
    print(f"Saved label encoders -> {cfg.ENCODERS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=cfg.RAW_DATA_PATH,
                         help="Path to the raw .xlsx dataset")
    args = parser.parse_args()
    main(args.input)

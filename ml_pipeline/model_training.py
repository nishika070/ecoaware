"""
Run this ONCE to train and save your AQI regressor:
    cd D:\ecoaware-project
    python ml_pipeline/model_training.py
"""
from __future__ import annotations
import os
import sys
import argparse
from xml.parsers.expat import model
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml_pipeline.custom_models.random_forest import RandomForest
from ml_pipeline.custom_models.metrics import rmse, r2_score
from utils.data_utils import load_daily_aqi, build_training_frame, get_feature_columns

# output paths 
TRAINING_DIR = os.path.join(os.path.dirname(__file__), "..", "training")
REGRESSOR_PATH = os.path.join(TRAINING_DIR, "aqi_regressor.pkl")
SCALER_PATH    = os.path.join(TRAINING_DIR, "data_scaler_aqi.pkl")
TARGET_SCALER_PATH = os.path.join(TRAINING_DIR, "target_scaler_aqi.pkl")


def aqi_to_policy(aqi: float) -> int:
    if aqi <= 50:   return 0
    if aqi <= 100:  return 1
    if aqi <= 200:  return 2
    if aqi <= 300:  return 3
    if aqi <= 400:  return 4
    return 5


def train_and_save(scaled=False):
    print("=" * 60)
    print("TRAINING AQI REGRESSOR (Lag-based Random Forest)")
    print("=" * 60)

    if scaled:
        print("Using scaled data from delhi25-26.csv")
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "datasets", "delhi25-26.csv"))
        X = df.drop('AQI', axis=1).values
        y = df['AQI'].values
        feature_cols = list(df.drop('AQI', axis=1).columns)
    else:
        daily = load_daily_aqi()
        if daily.empty:
            print("ERROR: No data loaded. Check your dataset path.")
            return

        frame = build_training_frame(daily)
        if frame.empty:
            print("ERROR: Training frame is empty after feature engineering.")
            return

        feature_cols = get_feature_columns()
        X = frame[feature_cols].values
        y = frame["target"].values

    print(f"Dataset shape : {X.shape}")
    print(f"Features      : {feature_cols}")

    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X)

    # scale targets (for pre-scaled data like delhi25-26.csv) 
    target_scaler = MinMaxScaler()
    y_reshaped = y.reshape(-1, 1) if len(y.shape) == 1 else y
    y_sc = target_scaler.fit_transform(y_reshaped).ravel()

    # temporal train/test split (80/20)
    split_idx  = int(len(X) * 0.8)
    X_train_sc, X_test_sc = X_sc[:split_idx], X_sc[split_idx:]
    y_train_sc, y_test_sc = y_sc[:split_idx], y_sc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]  # Keep original for eval metrics

    print(f"Train samples : {len(X_train_sc)}")
    print(f"Test samples  : {len(X_test_sc)}")

    print("\nTraining Random Forest Regressor …")
    model = RandomForest(
        n_trees=100,
        max_depth=10,
        min_samples_split=4,
        mode='regression'
    )
    model.fit(X_train_sc, y_train_sc)
    y_pred_sc = model.predict(X_test_sc)
    # Inverse transform for evaluation on original scale
    y_pred = target_scaler.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    rmse_val = rmse(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)

    print(f"\nTest RMSE : {rmse_val:.2f}")
    print(f"Test R²   : {r2_val:.3f}")

    os.makedirs(TRAINING_DIR, exist_ok=True)
    joblib.dump(model,  REGRESSOR_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)

    print(f"\n Model saved         → {REGRESSOR_PATH}")
    print(f" Feature Scaler saved → {SCALER_PATH}")
    print(f" Target Scaler saved  → {TARGET_SCALER_PATH}")
    print("\nNow restart your Flask server.")


if __name__ == "__main__":
    train_and_save()
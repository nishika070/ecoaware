"""
Run this ONCE to train and save your AQI regressor:
    cd D:\ecoaware-project
    python ml_pipeline/model_training.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from utils.data_utils import load_daily_aqi, build_training_frame, get_feature_columns

# ── output paths ─────────────────────────────────────────────────────────────
TRAINING_DIR = os.path.join(os.path.dirname(__file__), "..", "training")
REGRESSOR_PATH = os.path.join(TRAINING_DIR, "aqi_regressor.pkl")
SCALER_PATH    = os.path.join(TRAINING_DIR, "data_scaler_aqi.pkl")


def aqi_to_policy(aqi: float) -> int:
    if aqi <= 50:   return 0
    if aqi <= 100:  return 1
    if aqi <= 200:  return 2
    if aqi <= 300:  return 3
    if aqi <= 400:  return 4
    return 5


def train_and_save():
    print("=" * 60)
    print("TRAINING AQI REGRESSOR (Lag-based Random Forest)")
    print("=" * 60)

    # ── load data ─────────────────────────────────────────────
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

    # ── temporal train/test split (80/20) ─────────────────────
    split_idx  = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")

    # ── scale ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── train ─────────────────────────────────────────────────
    print("\nTraining Random Forest Regressor …")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_sc, y_train)

    # ── evaluate ──────────────────────────────────────────────
    y_pred = model.predict(X_test_sc)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    print(f"\nTest RMSE : {rmse:.2f}")
    print(f"Test R²   : {r2:.3f}")

    # ── save ──────────────────────────────────────────────────
    os.makedirs(TRAINING_DIR, exist_ok=True)
    joblib.dump(model,  REGRESSOR_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\n✅ Model saved  → {REGRESSOR_PATH}")
    print(f"✅ Scaler saved → {SCALER_PATH}")
    print("\nNow restart your Flask server.")


if __name__ == "__main__":
    train_and_save()
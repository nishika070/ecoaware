from __future__ import annotations
import os
import joblib
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), "..", "training")


def _load(name):
    path = os.path.join(_BASE, name)
    if os.path.exists(path):
        return joblib.load(path)
    return None


# Load AQI regressor (trained by model_training.py)
# Falls back to policy model only if regressor not found
_regressor = _load("aqi_regressor.pkl")        # ✅ your lag-based RF regressor
_scaler    = _load("data_scaler_aqi.pkl")       # ✅ scaler for lag features

if _regressor is None:
    # fallback: try policy model (will likely give wrong results but won't crash)
    _regressor = _load("aqi_policy_model.pkl")
    print("[aqi_model] WARNING: aqi_regressor.pkl not found. Run model_training.py to train.")


# ── predict_aqi ──────────────────────────────────────────────────────────────
def predict_aqi(features: np.ndarray) -> float:
    """
    Predict next-day AQI from lag-based features.
    features = [lag_1, lag_2, lag_3, lag_7,
                rolling_mean_3, rolling_mean_7,
                temp_lag_1, temp_rolling_mean_3,
                month, day, day_of_week, day_of_year]
    """
    try:
        if hasattr(features, "to_numpy"):
            features = features.to_numpy()
        flat_features = np.ravel(np.array(features, dtype=float))
        X = flat_features.reshape(1, -1)
        if _scaler is not None:
            X = _scaler.transform(X)
        if _regressor is not None:
            result = _regressor.predict(X)
            val = result[0]
            if hasattr(val, "__len__"):
                return float(flat_features[0]) if flat_features.size > 0 else 150.0
            return float(val)
    except Exception as e:
        print(f"[predict_aqi] fallback due to: {e}")

    flat = np.ravel(np.array(features, dtype=float)) if features is not None else np.array([])
    return float(flat[0]) if flat.size > 0 else 150.0


# ── get_policy_action ────────────────────────────────────────────────────────
def get_policy_action(aqi: float) -> str:
    if aqi <= 50:   return "No restrictions needed. Air quality is good."
    if aqi <= 100:  return "Advisory issued. Sensitive groups should limit outdoor activity."
    if aqi <= 200:  return "Moderate restrictions. Reduce vehicle usage and industrial output."
    if aqi <= 300:  return "Odd-even vehicle scheme active. Construction halted."
    if aqi <= 400:  return "Emergency measures. Schools closed, heavy industry suspended."
    return "Severe emergency. All non-essential outdoor activity banned."


# ── get_health_suggestion ────────────────────────────────────────────────────
def get_health_suggestion(aqi: float) -> str:
    if aqi <= 50:   return "Air quality is satisfactory. Enjoy outdoor activities."
    if aqi <= 100:  return "Unusually sensitive people should consider reducing prolonged exertion."
    if aqi <= 200:  return "Wear a mask outdoors. Reduce prolonged physical activity."
    if aqi <= 300:  return "N95 mask essential. Children and elderly should stay indoors."
    if aqi <= 400:  return "Avoid all outdoor exposure. Keep windows closed."
    return "Health emergency. Seek medical advice if experiencing symptoms."

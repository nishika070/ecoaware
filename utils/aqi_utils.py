from __future__ import annotations
from typing import Any


# ----------------------------------------------------------------------------
# AQI CLASSIFICATION (SINGLE SOURCE OF TRUTH)
# ----------------------------------------------------------------------------

def classify_aqi(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 150:
        return "Moderately Polluted"
    elif aqi <= 200:
        return "Poor"
    elif aqi <= 300:
        return "Very Poor"
    elif aqi <= 400:
        return "Severe"
    else:
        return "Severe+"


# ----------------------------------------------------------------------------
# HEALTH + ADVICE
# ----------------------------------------------------------------------------

def advice_for_aqi(aqi: float) -> str:
    if aqi <= 50:
        return "Outdoor activity is generally safe."
    elif aqi <= 100:
        return "Sensitive groups should reduce prolonged exertion."
    elif aqi <= 150:
        return "Limit prolonged outdoor activity for sensitive individuals."
    elif aqi <= 200:
        return "Reduce prolonged outdoor activity; consider wearing masks."
    elif aqi <= 300:
        return "Avoid intense outdoor activity, especially for children and elderly."
    elif aqi <= 400:
        return "Stay indoors when possible; use air filtration if available."
    else:
        return "Avoid outdoor exposure; follow high-risk emergency precautions."


# ----------------------------------------------------------------------------
# UI COLOR + STATUS CLASS
# ----------------------------------------------------------------------------

def get_aqi_color(aqi: float) -> str:
    if aqi <= 50:
        return "#2e9f57"
    elif aqi <= 100:
        return "#8abf2f"
    elif aqi <= 150:
        return "#d2a819"
    elif aqi <= 200:
        return "#e67e22"
    elif aqi <= 300:
        return "#d55353"
    else:
        return "#7a0019"


def get_status_class(aqi: float) -> str:
    if aqi <= 50:
        return "status-good"
    elif aqi <= 100:
        return "status-satisfactory"
    elif aqi <= 150:
        return "status-moderate"
    elif aqi <= 200:
        return "status-poor"
    elif aqi <= 300:
        return "status-very-poor"
    else:
        return "status-severe"


# ----------------------------------------------------------------------------
# RELATIVE SPREAD (MAP VISUALS)
# ----------------------------------------------------------------------------

def get_relative_spread_color(aqi: float, min_aqi: float, max_aqi: float) -> str:
    # fallback if no variation
    if max_aqi <= min_aqi:
        return "#d2a819"

    ratio = (aqi - min_aqi) / (max_aqi - min_aqi)

    if ratio <= 0.2:
        return "#2e9f57"
    elif ratio <= 0.4:
        return "#8abf2f"
    elif ratio <= 0.6:
        return "#d2a819"
    elif ratio <= 0.8:
        return "#e67e22"
    else:
        return "#d55353"


def get_relative_spread_label(aqi: float, min_aqi: float, max_aqi: float) -> str:
    # fallback if no variation
    if max_aqi <= min_aqi:
        return "Uniform spread"

    ratio = (aqi - min_aqi) / (max_aqi - min_aqi)

    if ratio <= 0.2:
        return "Low spread"
    elif ratio <= 0.4:
        return "Mild spread"
    elif ratio <= 0.6:
        return "Medium spread"
    elif ratio <= 0.8:
        return "High spread"
    else:
        return "Very high spread"


# ----------------------------------------------------------------------------
# SAFE HELPERS
# ----------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None
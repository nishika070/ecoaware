from __future__ import annotations
import time  # ← YEH ADD KARO
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import sys
import os
import pickle
import numpy as np
import json
import pandas as pd
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

from station_map import STATION_COORDINATES
from weather_service import get_station_weather_snapshot

# Add custom_models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'custom_models'))

try:
    from random_forest import RandomForest
    from decision_tree import DecisionTree
except ImportError:
    RandomForest = None
    DecisionTree = None


_live_aqi_cache: dict = {}
_live_aqi_cache_time: float = 0
_LIVE_AQI_TTL = 600
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "datasets" / "Merged_all_readable.csv"
DATASET_SCALED_PATH = BASE_DIR / "datasets" / "Merged_all_scaled.csv"
MODEL_DIR = BASE_DIR / "models"

AQI_REGRESSOR_PATH = MODEL_DIR / "aqi_regressor.pkl"
POLICY_CLASSIFIER_PATH = MODEL_DIR / "policy_classifier.pkl"
MERGER_SCALER_PATH = MODEL_DIR / "data_scaler_merger.pkl"


@dataclass
class AqiPredictionBundle:
    today: int
    tomorrow: int
    advice: str
    category: str
    policy_level: int
    policy_action: str
    health_suggestion: str
    latest_date: str
    model_name: str
    station_count: int
    history_labels: list[str] = None
    history_values: list[int] = None
    temperature: float | None = None

HEALTH_SUGGESTIONS = {
    0: "Good! Air quality is satisfactory. Enjoy outdoor activities.",
    1: "Satisfactory. Sensitive groups should limit prolonged outdoor time.",
    2: "Unhealthy for sensitive groups. Wear N95 masks outdoors.",
    3: "Unhealthy! General public advised to avoid outdoor activities.",
    4: "Very unhealthy! Stay indoors. Use air purifiers.",
    5: "Hazardous! Avoid all outdoor activities. Wear respirators.",
    6: "EMERGENCY! Remain indoors. Medical support may be needed."
}

POLICY_ACTIONS = {
    0: "No special action (AQI low / improvement expected)",
    1: "GRAP Stage-3 / Stage-4 measures (partial restrictions, e.g., stricter controls)",
    2: "Odd-even vehicle policy",
    3: "Industrial checks + fines for fire/ash/dust control",
    4: "Water sprinkler enforcement (or upgrade if already present)",
    5: "Suspend outdoor activities + shift schools/colleges/offices to online/work-from-home",
    6: "Suspend construction activities temporarily"
}

AQI_CATEGORIES = {
    "0-50": "Good",
    "51-100": "Satisfactory",
    "101-150": "Moderately Polluted",
    "151-200": "Poor",
    "201-300": "Very Poor",
    "301-400": "Severe",
    "400+": "Severe+"
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_station_name(raw_name: str) -> str:
    cleaned = raw_name.replace("_", " ")
    for suffix in [" Delhi DPCC", " Delhi CPCB", " Delhi IMD", " Delhi IITM"]:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    cleaned = cleaned.replace("(T3)", "T3")
    return " ".join(cleaned.split())


def format_number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "Unavailable"
    try:
        return f"{round(float(value), digits)}"
    except (TypeError, ValueError):
        return "Unavailable"


def format_int(value: Any) -> str:
    if value is None:
        return "Unavailable"
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return "Unavailable"


def aqi_to_policy_level(aqi: float) -> int:
    if aqi <= 100:
        return 0
    elif aqi <= 150:
        return 1
    elif aqi <= 200:
        return 2
    elif aqi <= 300:
        return 3
    elif aqi <= 400:
        return 4
    elif aqi <= 500:
        return 5
    else:
        return 6


def aqi_to_category(aqi: float) -> str:
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


def get_health_suggestion(aqi: float) -> str:
    policy_level = aqi_to_policy_level(aqi)
    health_index = min(int(policy_level), len(HEALTH_SUGGESTIONS) - 1)
    return HEALTH_SUGGESTIONS.get(health_index, "Consult health authorities for guidance.")


def get_policy_action(aqi: float) -> str:
    policy_level = aqi_to_policy_level(aqi)
    policy_index = min(int(policy_level), len(POLICY_ACTIONS) - 1)
    return POLICY_ACTIONS.get(policy_index, "No policy available")


def get_aqi_color(aqi: float) -> str:
    if aqi <= 50:
        return "#2e9f57"
    elif aqi <= 100:
        return "#8abf2f"
    elif aqi <= 200:
        return "#d2a819"
    elif aqi <= 300:
        return "#e67e22"
    elif aqi <= 400:
        return "#d55353"
    else:
        return "#7a0019"


def get_status_class(aqi: float) -> str:
    if aqi <= 50:
        return "status-good"
    elif aqi <= 100:
        return "status-satisfactory"
    elif aqi <= 200:
        return "status-moderate"
    elif aqi <= 300:
        return "status-poor"
    elif aqi <= 400:
        return "status-very-poor"
    else:
        return "status-severe"


def classify_aqi(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def advice_for_aqi(aqi: float) -> str:
    if aqi <= 50:
        return "Outdoor activity is generally safe."
    elif aqi <= 100:
        return "Sensitive groups should reduce prolonged exertion."
    elif aqi <= 200:
        return "Limit prolonged outdoor activity and keep masks ready."
    elif aqi <= 300:
        return "Avoid intense outdoor activity, especially for children and elderly."
    elif aqi <= 400:
        return "Stay indoors when possible; use air filtration if available."
    else:
        return "Avoid outdoor exposure; follow high-risk emergency precautions."


@lru_cache(maxsize=1)
def load_models() -> tuple[Any, Any, Any] | tuple[None, None, None]:
    try:
        if not AQI_REGRESSOR_PATH.exists():
            print(f"⚠ AQI Regressor not found at {AQI_REGRESSOR_PATH}")
            return None, None, None
        if not POLICY_CLASSIFIER_PATH.exists():
            print(f"⚠ Policy Classifier not found at {POLICY_CLASSIFIER_PATH}")
            return None, None, None
        if not MERGER_SCALER_PATH.exists():
            print(f"⚠ Scaler not found at {MERGER_SCALER_PATH}")
            return None, None, None

        with open(AQI_REGRESSOR_PATH, 'rb') as f:
            aqi_regressor = pickle.load(f)
        with open(POLICY_CLASSIFIER_PATH, 'rb') as f:
            policy_classifier = pickle.load(f)
        scaler = joblib.load(MERGER_SCALER_PATH)

        print("✓ Models loaded successfully!")
        return aqi_regressor, policy_classifier, scaler

    except Exception as e:
        print(f"✗ Error loading models: {str(e)}")
        return None, None, None


@lru_cache(maxsize=1)
def load_aqi_data() -> pd.DataFrame:
    try:
        return pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f"✗ Error loading AQI data: {str(e)}")
        return pd.DataFrame()


@lru_cache(maxsize=1)
def load_scaled_data() -> pd.DataFrame:
    try:
        return pd.read_csv(DATASET_SCALED_PATH)
    except Exception as e:
        print(f"✗ Error loading scaled data: {str(e)}")
        return pd.DataFrame()


def predict_aqi(X_features: np.ndarray) -> tuple[float, float]:
    aqi_regressor, policy_classifier, scaler = load_models()

    if aqi_regressor is None or policy_classifier is None:
        return 200.0, 3

    try:
        if len(X_features.shape) == 1:
            X_features = X_features.reshape(1, -1)

        aqi_pred_scaled = aqi_regressor.predict(X_features)[0]
        policy_pred = policy_classifier.predict(X_features)[0]

        X_full = X_features[0].copy()
        X_full_with_aqi = np.concatenate([X_full[:11], [aqi_pred_scaled], X_full[11:]])
        real_vals = scaler.inverse_transform(X_full_with_aqi.reshape(1, -1))[0]
        aqi_real = real_vals[11]
        aqi_real = np.clip(aqi_real, 0, 500)

        return float(aqi_real), int(policy_pred)

    except Exception as e:
        print(f"✗ Prediction error: {str(e)}")
        return 200.0, 3



def get_relative_spread_color(aqi: float, min_aqi: float, max_aqi: float) -> str:
    if max_aqi <= min_aqi:
        return "#d2a819"
    spread_ratio = (aqi - min_aqi) / (max_aqi - min_aqi)
    if spread_ratio <= 0.2:
        return "#2e9f57"
    if spread_ratio <= 0.4:
        return "#8abf2f"
    if spread_ratio <= 0.6:
        return "#d2a819"
    if spread_ratio <= 0.8:
        return "#e67e22"
    return "#d55353"


def get_relative_spread_label(aqi: float, min_aqi: float, max_aqi: float) -> str:
    if max_aqi <= min_aqi:
        return "Uniform spread"
    spread_ratio = (aqi - min_aqi) / (max_aqi - min_aqi)
    if spread_ratio <= 0.2:
        return "Lower pressure in city spread"
    if spread_ratio <= 0.4:
        return "Mild pressure in city spread"
    if spread_ratio <= 0.6:
        return "Medium pressure in city spread"
    if spread_ratio <= 0.8:
        return "High pressure in city spread"
    return "Very high pressure in city spread"


def get_available_stations() -> list[str]:
    station_daily = load_station_daily_aqi()
    return sorted(station_daily["display_station"].unique().tolist())


def resolve_station_name(selected_station: str | None = None) -> str | None:
    available_stations = get_available_stations()
    if not available_stations:
        return None
    if selected_station in available_stations:
        return selected_station
    return available_stations[0]


def get_station_series(station_name: str) -> pd.DataFrame:
    station_daily = load_station_daily_aqi()
    station_series = (
        station_daily[station_daily["display_station"] == station_name]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if station_series.empty:
        fallback_station = resolve_station_name()
        if fallback_station is None:
            return station_series
        station_series = (
            station_daily[station_daily["display_station"] == fallback_station]
            .sort_values("date")
            .reset_index(drop=True)
        )

    return station_series


def get_station_latest_table() -> list[dict[str, Any]]:
    station_daily = load_station_daily_aqi()
    latest_rows = (
        station_daily.sort_values("date")
        .groupby("display_station", as_index=False)
        .tail(1)
        .sort_values("aqi", ascending=False)
    )

    results = []
    for _, row in latest_rows.iterrows():
        aqi_value = float(row["aqi"])
        status = classify_aqi(aqi_value)
        results.append(
            {
                "station": row["display_station"],
                "aqi": int(round(aqi_value)),
                "status": status,
                "advice": advice_for_aqi(aqi_value),
                "latest_date": row["date"].strftime("%d %b %Y"),
                "color": get_aqi_color(aqi_value),
                "status_class": get_status_class(aqi_value),
            }
        )

    return results
def get_all_stations_live_aqi() -> dict[str, int | None]:
    global _live_aqi_cache, _live_aqi_cache_time
    
    if time.time() - _live_aqi_cache_time < _LIVE_AQI_TTL and _live_aqi_cache:
        return _live_aqi_cache
    
    station_rows = get_station_latest_table()
    results: dict[str, int | None] = {}
    
    def fetch_one(station_name: str):
        coords = STATION_COORDINATES.get(station_name, {})
        weather = get_station_weather_snapshot(
            station_name,
            coords.get("lat"),
            coords.get("lng"),
        )
        aqi = weather.get("air_quality", {}).get("aqi")
        return station_name, (int(aqi) if isinstance(aqi, (int, float)) else None)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_one, row["station"]): row["station"]
                   for row in station_rows}
        for future in as_completed(futures):
            try:
                name, aqi = future.result(timeout=8)
                results[name] = aqi
            except Exception:
                results[futures[future]] = None
    
    _live_aqi_cache = results
    _live_aqi_cache_time = time.time()
    return results

def get_monthly_pattern() -> list[dict[str, Any]]:
    station_daily = load_station_daily_aqi().copy()
    monthly = (
        station_daily.groupby(["date"], as_index=False)["aqi"].mean()
        .assign(month=lambda frame: frame["date"].dt.month, month_label=lambda frame: frame["date"].dt.strftime("%b"))
        .groupby(["month", "month_label"], as_index=False)["aqi"]
        .mean()
        .sort_values("month")
    )
    return [
        {"month": row["month_label"], "aqi": int(round(row["aqi"]))}
        for _, row in monthly.iterrows()
    ]


def get_home_context(selected_station: str | None = None) -> dict[str, Any]:

    available_stations = get_available_stations()
    station_name = resolve_station_name(selected_station)

    city_payload = build_prediction_payload()
    station_rows = get_station_latest_table()
    hotspot_markers = []
    map_min_aqi = min((row["aqi"] for row in station_rows), default=0)
    map_max_aqi = max((row["aqi"] for row in station_rows), default=0)


    if station_name is None:
        station_name = selected_station or "No station data"
        station_series = pd.DataFrame(columns=["date", "aqi"])
        history = station_series.tail(7)
        latest_station_aqi = float(city_payload["today"])
        latest_station_date = city_payload["latest_date"]
    else:
        station_series = get_station_series(station_name)
        history = station_series.tail(7)
        if station_series.empty:
            latest_station_aqi = float(city_payload["today"])
            latest_station_date = city_payload["latest_date"]
        else:
            latest_station = station_series.iloc[-1]
            latest_station_aqi = float(latest_station["aqi"])
            latest_station_date = latest_station["date"].strftime("%d %b %Y")

    station_coordinates = STATION_COORDINATES.get(station_name, {})
    weather = get_station_weather_snapshot(
        station_name,
        station_coordinates.get("lat"),
        station_coordinates.get("lng"),
    )
    current_weather = weather.get("current", {})
    hourly_preview = weather.get("hourly_preview", [])
    forecast_days = weather.get("forecast_days", [])
    air_quality = weather.get("air_quality", {})
    weather_error = weather.get("source_error")
    print(air_quality)
    live_us_aqi = air_quality.get("aqi")
    today_forecast = forecast_days[0] if forecast_days else {}
    live_station_aqis = get_all_stations_live_aqi()

    for row in station_rows:
        coordinates = STATION_COORDINATES.get(row["station"])
        if not coordinates:
            continue
    
        # Prefer live AQI, fall back to dataset AQI
        live_aqi = live_station_aqis.get(row["station"])
        display_aqi = live_aqi if live_aqi is not None else row["aqi"]
        display_status = classify_aqi(float(display_aqi))
    
        hotspot_markers.append({
            "name": row["station"],
            "aqi": display_aqi,
            "status": display_status,
            "advice": advice_for_aqi(display_aqi),
            "latest_date": row["latest_date"],
            "lat": coordinates["lat"],
            "lng": coordinates["lng"],
            "color": get_aqi_color(display_aqi),
            "radius": min(6 + display_aqi / 40, 22),
        })
    

    live_category = classify_aqi(float(live_us_aqi)) if live_us_aqi is not None else city_payload["category"]
    live_advice = advice_for_aqi(float(live_us_aqi)) if live_us_aqi is not None else city_payload["advice"]
    live_aqi_numeric = float(live_us_aqi) if live_us_aqi is not None else latest_station_aqi
    today_max_temp = today_forecast.get("max_temp")  # ← fixed key
    today_min_temp = today_forecast.get("min_temp")  # ← fixed key
    visibility_m = current_weather.get("visibility_m")
    
    visibility_km = round(float(visibility_m) / 1000, 1) if visibility_m is not None else None
    return {
        "stations": available_stations,
        "selected_station": station_name,
        "primary_metrics": [
            {
                "title": "Temperature",
                "value": (
                    f"{format_number(current_weather.get('temperature_c'))} C"
                    if current_weather.get("temperature_c") is not None
                    else "Unavailable"
                ),
                "note": weather_error if weather_error else f"Live temperature for {station_name}",
            },
            {
                "title": "Precipitation Chance",
                "value": (
                    f"{format_int(today_forecast.get('precip_probability'))}%"
                    if today_forecast.get("precip_probability") is not None
                    else "Unavailable"
                ),
                "note": weather_error if weather_error else "Today's maximum rain probability",
            },
            {
                "title": "AQI",
                "value": int(round(live_aqi_numeric)) if live_us_aqi is not None else int(round(latest_station_aqi)),
                "note": (
                    "Live US AQI from WAQI"
                    if live_us_aqi is not None
                    else f"Dataset AQI on {latest_station_date}"
                ),
            },
            {
                "title": "Health Advisory",
                "value": live_category,
                "note": live_advice,
            },
        ],
        "secondary_metrics": [
            {
                "title": "Delhi Prediction",
                "value": city_payload["tomorrow"],
                "note": f"{city_payload['model_name']} next-day AQI forecast",
            },
            {
                "title": "Monitored Stations",
                "value": city_payload["station_count"],
                "note": "Coverage from the latest Delhi AQI dataset snapshot",
            },
        ],
        "chart": get_station_30day_chart(station_name),
        "advisory": {
            "headline": f"{station_name}: {current_weather.get('condition', 'Current weather')}",
            "tag": live_advice,
            "summary": (
                "Live weather and pollutant feed is from Open-Meteo and WAQI. "
                f"AQI trend model is trained on dataset data through {city_payload['latest_date']}."
            ),
            "items": [
                (
                    f"Feels like: {format_number(current_weather.get('feels_like_c'))} C."
                    if current_weather.get("feels_like_c") is not None
                    else "Feels-like temperature unavailable right now."
                ),
                (
                    f"Humidity: {format_int(current_weather.get('humidity_percent'))}%."
                    if current_weather.get("humidity_percent") is not None
                    else "Humidity data unavailable right now."
                ),
                (
                    f"Today's rain probability: {format_int(today_forecast.get('precip_probability'))}%."
                    if today_forecast.get("precip_probability") is not None
                    else "Rain probability unavailable right now."
                ),
                (
                    f"Live AQI: {format_int(live_us_aqi)}."
                    if live_us_aqi is not None
                    else f"Latest dataset AQI: {int(round(latest_station_aqi))}."
                ),
            ],
        },
        "hero": {
            "eyebrow": "Delhi environmental intelligence platform",
            "title": "Integrated AQI, weather and public-health signals for daily decisions",
            "description": (
                "Track live temperature, 5-day precipitation and pollutant signals for your selected station, "
                "with model-based AQI trend insights in one dashboard."
            ),
        },
        "map": {
            "center": {"lat": 28.6139, "lng": 77.2090},
            "markers": hotspot_markers,
            "legend": [
                {"label": "Lower city spread", "color": "#2e9f57"},
                {"label": "Mild city spread", "color": "#8abf2f"},
                {"label": "Medium city spread", "color": "#d2a819"},
                {"label": "High city spread", "color": "#e67e22"},
                {"label": "Very high city spread", "color": "#d55353"},
            ],
        },
        "live_weather": {
            "station": station_name,
            "error": weather_error,
            "current": current_weather,
            "hourly_preview": hourly_preview,
            "air_quality": air_quality,
            "forecast_days": forecast_days,
            "fetched_at": weather.get("fetched_at"),
        },
        "hourly_preview": hourly_preview,
        "home_weather": {
            "temperature_c": current_weather.get("temperature_c"),
            "feels_like_c": current_weather.get("feels_like_c"),
            "condition": current_weather.get("condition"),
            "temp_max_today": today_max_temp,
            "temp_min_today": today_min_temp,
            "rain_probability": today_forecast.get("precip_probability"),
            "wind_speed_kmh": current_weather.get("wind_speed_kmh"),
            "wind_direction_deg": current_weather.get("wind_direction_deg"),
            "wind_gust_kmh": current_weather.get("wind_gust_kmh"),
            "humidity_percent": current_weather.get("humidity_percent"),
            "precip_mm": current_weather.get("precip_mm"),
            "cloud_cover_percent": current_weather.get("cloud_cover_percent"),
            "visibility_km": visibility_km,
            "pressure_hpa": current_weather.get("pressure_hpa"),  
            "uv_index": today_forecast.get("uv_index"),
            "aqi": int(round(live_aqi_numeric)),
            "aqi_label": live_category,
            "aqi_class": get_status_class(live_aqi_numeric),
            "updated_at": weather.get("fetched_at"),
        },
        "temperature_chart": {
            "labels": [item["day_label"] for item in forecast_days],
            "max_temps": [item.get("max_temp_c") for item in forecast_days],  # ← fixed key
            "min_temps": [item.get("min_temp_c") for item in forecast_days],  # ← fixed key
            "precip_chance": [item["precip_probability"] for item in forecast_days],
        },
    }

def get_aqi_page_context(selected_station: str | None = None) -> dict[str, Any]:
    csv_rows = get_station_latest_table()
    live_aqis = get_all_stations_live_aqi()

    station_rows = []
    for row in csv_rows:
        live_aqi = live_aqis.get(row["station"])
        display_aqi = int(live_aqi) if isinstance(live_aqi, (int, float)) else row["aqi"]
        station_rows.append({
            "station":      row["station"],
            "aqi":          display_aqi,
            "status":       classify_aqi(float(display_aqi)),
            "advice":       advice_for_aqi(float(display_aqi)),
            "latest_date":  row["latest_date"],
            "color":        get_aqi_color(float(display_aqi)),
            "status_class": get_status_class(float(display_aqi)),
        })
    station_rows.sort(key=lambda x: x["aqi"], reverse=True)

    if selected_station is None or selected_station == "all":
        coords = STATION_COORDINATES.get("Delhi", {"lat": 28.61, "lng": 77.23})
        weather = get_station_weather_snapshot("Delhi", coords.get("lat"), coords.get("lng"))
        air_quality = weather.get("air_quality", {})
        live_aqi = air_quality.get("aqi")
        current_aqi = int(live_aqi) if isinstance(live_aqi, (int, float)) else (
            int(round(sum(r["aqi"] for r in station_rows) / len(station_rows))) if station_rows else 0
        )
        current_status = classify_aqi(current_aqi)
        current_label = "City Average"
        selected_station = "all"
    else:
        selected = next((s for s in station_rows if s["station"] == selected_station), None)
        coords = STATION_COORDINATES.get(selected_station, {})
        weather = get_station_weather_snapshot(selected_station, coords.get("lat"), coords.get("lng"))
        air_quality = weather.get("air_quality", {})
        live_aqi = air_quality.get("aqi")
        current_aqi = int(live_aqi) if isinstance(live_aqi, (int, float)) else (selected["aqi"] if selected else 0)
        current_status = classify_aqi(current_aqi)
        current_label = selected["station"] if selected else "Unknown"

    current_pollutants = {
        "pm25": air_quality.get("pm2_5"),
        "pm10": air_quality.get("pm10"),
        "no2":  air_quality.get("nitrogen_dioxide"),
        "so2":  air_quality.get("sulphur_dioxide"),
        "co":   air_quality.get("carbon_monoxide"),
        "o3":   air_quality.get("ozone"),
    }

    _pred = build_prediction_payload()
    predicted = {
        "aqi":    _pred["tomorrow"],
        "status": classify_aqi(_pred["tomorrow"]),
    }

    return {
        "station_rows":       station_rows,
        "selected_station":   selected_station,
        "current_aqi":        current_aqi,
        "current_status":     current_status,
        "current_label":      current_label,
        "current_pollutants": current_pollutants,
        "predicted":          predicted,
        "policy_note": (
            f"{station_rows[0]['station']} currently has the highest AQI."
            if station_rows else "No AQI data available."
        ),
        "aqi_legend": [
            {"range": "0-50",    "label": "Good",         "color": "#2e9f57"},
            {"range": "51-100",  "label": "Satisfactory", "color": "#8abf2f"},
            {"range": "101-200", "label": "Moderate",     "color": "#d2a819"},
            {"range": "201-300", "label": "Poor",         "color": "#e67e22"},
            {"range": "301-400", "label": "Very Poor",    "color": "#d55353"},
            {"range": "401+",    "label": "Severe",       "color": "#7a0019"},
        ],
    }
def get_contact_page_context() -> dict[str, Any]:
    prediction = build_prediction_payload()
    return {
        "contact_cards": [
            {
                "title": "Project Email",
                "detail": "ecoaware.delhi@project.org (replace with your official college/team mail)",
            },
            {
                "title": "Mentor Showcase Focus",
                "detail": "Live AQI + weather intelligence, station map spread, and ML-backed policy suggestions for Delhi.",
            },
            {
                "title": "Model In Use",
                "detail": f"{prediction['model_name']} (fallback: Decision Tree) for next-day AQI prediction.",
            },
            {
                "title": "Current Prediction",
                "detail": f"Tomorrow AQI forecast: {prediction['tomorrow']} ({prediction['category']})",
            },
        ],
        "collaboration_points": [
            "AQI model improvement and retraining with latest CPCB records",
            "Hospital and school level health-alert automation",
            "Ward-level intervention planning for dust and traffic control",
            "Dashboard extensions for heat stress and rainfall risk response",
        ],
    }


# ============================================================================
# ACTIVE DATA PIPELINE OVERRIDES
# ============================================================================

MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


@lru_cache(maxsize=1)
def load_station_daily_aqi() -> pd.DataFrame:
    raw = pd.read_csv(DATASET_PATH)
    raw.columns = raw.columns.str.strip()
    required_columns = ["YEAR", "DOY", "T2M", "AQI", "Location"]
    available_columns = [column for column in required_columns if column in raw.columns]

    if len(available_columns) != len(required_columns):
        return pd.DataFrame(columns=["date", "aqi", "temperature", "station", "display_station"])

    frame = raw[required_columns].copy()
    frame["YEAR"] = pd.to_numeric(frame["YEAR"], errors="coerce")
    frame["DOY"] = pd.to_numeric(frame["DOY"], errors="coerce")
    frame["T2M"] = pd.to_numeric(frame["T2M"], errors="coerce")
    frame["AQI"] = pd.to_numeric(frame["AQI"], errors="coerce")
    frame = frame.dropna(subset=["YEAR", "DOY", "AQI", "Location"]).copy()

    frame["YEAR"] = frame["YEAR"].astype(int)
    frame["DOY"] = frame["DOY"].astype(int)
    frame["date"] = pd.to_datetime(
        frame["YEAR"].astype(str) + frame["DOY"].astype(str).str.zfill(3),
        format="%Y%j",
        errors="coerce",
    )
    frame = frame.dropna(subset=["date"]).copy()
    frame["station"] = frame["Location"]
    frame["display_station"] = frame["Location"].apply(format_station_name)
    frame["aqi"] = frame["AQI"].round(2)
    frame["temperature"] = frame["T2M"].round(2)
    return frame.sort_values(["display_station", "date"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def load_daily_aqi() -> pd.DataFrame:
    station_daily = load_station_daily_aqi()
    if station_daily.empty:
        return pd.DataFrame(columns=["date", "avg_aqi", "avg_temp", "station_count"])

    daily = (
        station_daily.groupby("date", as_index=False)
        .agg(
            avg_aqi=("aqi", "mean"),
            avg_temp=("temperature", "mean"),
            station_count=("display_station", "nunique"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["avg_aqi"] = daily["avg_aqi"].round(2)
    daily["avg_temp"] = daily["avg_temp"].round(2)
    return daily


def build_training_frame(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy().sort_values("date").reset_index(drop=True)
    if frame.empty:
        return frame

    frame["lag_1"] = frame["avg_aqi"].shift(1)
    frame["lag_2"] = frame["avg_aqi"].shift(2)
    frame["lag_3"] = frame["avg_aqi"].shift(3)
    frame["lag_7"] = frame["avg_aqi"].shift(7)
    frame["rolling_mean_3"] = frame["avg_aqi"].shift(1).rolling(3).mean()
    frame["rolling_mean_7"] = frame["avg_aqi"].shift(1).rolling(7).mean()
    frame["temp_lag_1"] = frame["avg_temp"].shift(1)
    frame["temp_rolling_mean_3"] = frame["avg_temp"].shift(1).rolling(3).mean()
    frame["month"] = frame["date"].dt.month
    frame["day"] = frame["date"].dt.day
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_of_year"] = frame["date"].dt.dayofyear
    frame["target"] = frame["avg_aqi"].shift(-1)
    frame["target_date"] = frame["date"].shift(-1)

    frame = frame.dropna().reset_index(drop=True)
    return frame


def get_feature_columns() -> list[str]:
    return [
        "lag_1", "lag_2", "lag_3", "lag_7",
        "rolling_mean_3", "rolling_mean_7",
        "temp_lag_1", "temp_rolling_mean_3",
        "month", "day", "day_of_week", "day_of_year",
    ]


@lru_cache(maxsize=1)
def load_or_train_model() -> tuple[Any, str]:
    training_frame = build_training_frame(load_daily_aqi())
    feature_columns = get_feature_columns()

    if len(training_frame) < 12:
        class PersistenceModel:
            def predict(self, frame: pd.DataFrame) -> np.ndarray:
                return frame["lag_1"].to_numpy(dtype=float)
        return PersistenceModel(), "Persistence baseline"

    split_index = max(int(len(training_frame) * 0.8), 1)
    X_train = training_frame.iloc[:split_index][feature_columns].to_numpy(dtype=np.float32)
    y_train = training_frame.iloc[:split_index]["target"].to_numpy(dtype=np.float32)

    if RandomForest is not None:
        model = RandomForest(n_trees=18, max_depth=8, min_samples_split=4, mode="regression")
        model.fit(X_train, y_train)
        return model, "Random Forest"

    class MeanModel:
        def __init__(self, mean_value: float):
            self.mean_value = float(mean_value)
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.full(len(frame), self.mean_value, dtype=float)

    return MeanModel(float(np.mean(y_train))), "Mean baseline"


def _build_prediction_backtest() -> dict[str, Any]:
    training_frame = build_training_frame(load_daily_aqi())
    feature_columns = get_feature_columns()

    if training_frame.empty:
        return {
            "mae": "Unavailable", "rmse": "Unavailable",
            "r2_score": "Unavailable", "classification_accuracy": "Unavailable",
            "prediction_comparisons": [], "mae_value": None,
            "training_data_period": "Unavailable", "test_data_period": "Unavailable",
            "last_updated": "Unavailable", "no_historical_predictions": True,
        }

    model, model_name = load_or_train_model()
    split_index = max(int(len(training_frame) * 0.8), 1)
    test_frame = training_frame.iloc[split_index:].copy()
    if test_frame.empty:
        test_frame = training_frame.tail(min(len(training_frame), 10)).copy()

    predictions = np.clip(np.asarray(model.predict(test_frame[feature_columns]), dtype=float), 0, 500)
    actuals = test_frame["target"].to_numpy(dtype=float)
    predicted_policy = np.array([aqi_to_policy_level(v) for v in predictions], dtype=int)
    actual_policy = np.array([aqi_to_policy_level(v) for v in actuals], dtype=int)

    absolute_errors = np.abs(actuals - predictions)
    squared_errors = np.square(actuals - predictions)
    mae_value = float(np.mean(absolute_errors)) if len(absolute_errors) else None
    rmse_value = float(np.sqrt(np.mean(squared_errors))) if len(squared_errors) else None
    actual_mean = float(np.mean(actuals)) if len(actuals) else 0.0
    total_variance = float(np.sum(np.square(actuals - actual_mean))) if len(actuals) else 0.0
    residual_variance = float(np.sum(squared_errors)) if len(squared_errors) else 0.0
    r2_value = None if total_variance == 0 else (1 - residual_variance / total_variance)
    class_accuracy = float(np.mean(predicted_policy == actual_policy)) if len(actual_policy) else None

    comparisons = []
    for (_, row), prediction, actual, error in zip(test_frame.iterrows(), predictions, actuals, absolute_errors):
        actual_date = row["target_date"]
        comparisons.append(
            {
                "date": actual_date.strftime("%d %b %Y"),
                "predicted": int(round(prediction)),
                "actual": int(round(actual)),
                "error": round(float(error), 1),
                "error_percent": round(float((error / actual) * 100), 1) if actual else 0.0,
            }
        )

    return {
        "mae": f"{mae_value:.1f}" if mae_value is not None else "Unavailable",
        "rmse": f"{rmse_value:.1f}" if rmse_value is not None else "Unavailable",
        "r2_score": f"{r2_value:.2f}" if r2_value is not None else "Unavailable",
        "classification_accuracy": f"{class_accuracy * 100:.0f}%" if class_accuracy is not None else "Unavailable",
        "prediction_comparisons": list(reversed(comparisons[-10:])),
        "mae_value": mae_value,
        "training_data_period": (
            f"{training_frame.iloc[0]['date'].strftime('%d %b %Y')} to {training_frame.iloc[split_index - 1]['date'].strftime('%d %b %Y')}"
            if split_index > 0 else "Unavailable"
        ),
        "test_data_period": (
            f"{test_frame.iloc[0]['target_date'].strftime('%d %b %Y')} to {test_frame.iloc[-1]['target_date'].strftime('%d %b %Y')}"
            if not test_frame.empty else "Unavailable"
        ),
        "last_updated": training_frame.iloc[-1]["target_date"].strftime("%d %b %Y"),
        "model_name": model_name,
        "no_historical_predictions": len(comparisons) == 0,
    }


def _month_focus(daily: pd.DataFrame) -> tuple[int, str]:
    if daily.empty:
        return 1, MONTH_LABELS[1]
    current_month = pd.Timestamp.today().month
    month_frame = daily[daily["date"].dt.month == current_month]
    if month_frame.empty:
        current_month = int(daily.iloc[-1]["date"].month)
    return current_month, MONTH_LABELS[current_month]


def _build_monthly_chart_series(daily: pd.DataFrame, value_column: str, month_number: int) -> tuple[list[str], list[float]]:
    month_frame = daily[daily["date"].dt.month == month_number].copy()
    if month_frame.empty:
        return [], []
    grouped = (
        month_frame.groupby(month_frame["date"].dt.year)[value_column]
        .mean()
        .reset_index(name="value")
        .sort_values("date")
    )
    return grouped["date"].astype(int).astype(str).tolist(), grouped["value"].round(1).tolist()


def _build_last_30_days_series(daily: pd.DataFrame, value_column: str) -> tuple[list[str], list[float]]:
    recent = daily.tail(30).copy()
    if recent.empty:
        return [], []
    return recent["date"].dt.strftime("%d %b").tolist(), recent[value_column].round(1).tolist()


def _temperature_history_looks_unreliable(daily: pd.DataFrame) -> bool:
    if daily.empty or "avg_temp" not in daily:
        return True
    temps = pd.to_numeric(daily["avg_temp"], errors="coerce").dropna()
    if temps.empty:
        return True
    return bool(temps.max() < 15 or temps.mean() < 10 or temps.quantile(0.9) < 18)


def _temperature_policy_items(current_temp: float | None, baseline_temp: float | None) -> list[str]:
    if current_temp is None:
        return [
            "Use the latest dataset temperature until live weather is available.",
            "Keep heat-readiness actions flexible because the live temperature feed is missing.",
            "Continue monitoring school, worker, and transport advisories once fresh data arrives.",
        ]
    items = []
    if current_temp >= 38:
        items.append("Activate heat-action messaging for outdoor workers, schools, and transport hubs.")
        items.append("Increase public drinking-water access and shaded waiting areas in dense corridors.")
    elif current_temp >= 32:
        items.append("Schedule field inspections and construction work earlier in the day to reduce heat stress.")
        items.append("Issue hydration and cooling advisories across public facilities.")
    else:
        items.append("Maintain routine seasonal monitoring and preserve contingency plans for hotter days ahead.")
        items.append("Use the lower heat window for outdoor maintenance and civic operations.")
    if baseline_temp is not None and current_temp - baseline_temp >= 2:
        items.append("Temperatures are above the dataset month average, so scale readiness before the next warm spell.")
    else:
        items.append("Current temperature is near the month baseline, so keep targeted actions localized.")
    return items


def _temperature_health_items(current_temp: float | None, baseline_temp: float | None) -> list[str]:
    if current_temp is None:
        return [
            "Temperature-linked health guidance will tighten automatically when live data becomes available.",
            "For now, use standard hydration, shade, and midday exposure precautions.",
        ]
    items = []
    if current_temp >= 38:
        items.append("Avoid long outdoor exposure in the afternoon and prioritize hydration every hour.")
        items.append("Watch for headache, dizziness, or fatigue, especially in children and elderly residents.")
    elif current_temp >= 32:
        items.append("Carry water, use light clothing, and reduce prolonged exertion during peak sunlight.")
        items.append("Take shaded breaks if commuting or working outdoors.")
    else:
        items.append("Weather is comparatively manageable, but keep water intake steady through the day.")
        items.append("Sensitive groups should still avoid abrupt exposure during hotter afternoon periods.")
    if baseline_temp is not None and current_temp - baseline_temp >= 2:
        items.append("Because conditions are warmer than the month norm, heat fatigue can build faster than expected.")
    return items


def _aqi_policy_items(current_aqi: float, predicted_aqi: float, top_station: str) -> list[str]:
    items = [
        f"Prioritize enforcement near {top_station}, which currently leads the station AQI ranking.",
        f"Plan next-day controls around the forecast AQI of {int(round(predicted_aqi))} to avoid delayed response.",
    ]
    policy_level = aqi_to_policy_level(max(current_aqi, predicted_aqi))
    if policy_level >= 4:
        items.append("Escalate dust suppression, construction checks, and traffic-control messaging immediately.")
    elif policy_level >= 2:
        items.append("Intensify inspections at industrial and roadside dust hotspots before conditions worsen.")
    else:
        items.append("Keep monitoring active and use targeted advisories instead of city-wide restrictions.")
    return items


def _aqi_health_items(current_aqi: float, predicted_aqi: float) -> list[str]:
    risk_aqi = max(current_aqi, predicted_aqi)
    items = [advice_for_aqi(risk_aqi)]
    if risk_aqi > 300:
        items.append("Schools, elderly residents, and people with asthma should minimize outdoor exposure.")
        items.append("Use masks and indoor air filtration where possible.")
    elif risk_aqi > 150:
        items.append("Limit heavy exercise outdoors and keep rescue medication available for sensitive groups.")
        items.append("Prefer morning travel windows if outdoor movement is necessary.")
    else:
        items.append("General outdoor activity is acceptable, but sensitive groups should still monitor symptoms.")
        items.append("Check tomorrow's forecast before planning prolonged outdoor events.")
    return items


def build_prediction_payload() -> dict[str, Any]:
    daily = load_daily_aqi()
    if daily.empty:
        return {
            "today": 0, "tomorrow": 0,
            "advice": "AQI dataset unavailable.",
            "category": "Unavailable",
            "latest_date": "Unavailable",
            "history_labels": [], "history_values": [],
            "temperature": None,
            "model_name": "Unavailable",
            "station_count": 0,
            "policy_level": 0,
            "policy_action": "Dataset unavailable",
            "health_suggestion": "Dataset unavailable",
        }

    model, model_name = load_or_train_model()
    training_frame = build_training_frame(daily)
    feature_columns = get_feature_columns()

    latest_window = training_frame.iloc[-1].copy() if not training_frame.empty else None
    if latest_window is not None:
        predicted_tomorrow = float(np.clip(model.predict(latest_window[feature_columns].to_frame().T)[0], 0, 500))
    else:
        predicted_tomorrow = float(daily.iloc[-1]["avg_aqi"])

    today_value = float(daily.iloc[-1]["avg_aqi"])
    latest_date = daily.iloc[-1]["date"]
    history = daily.tail(7).copy()
    today_temperature = _safe_float(daily.iloc[-1]["avg_temp"])
    policy_level = aqi_to_policy_level(predicted_tomorrow)

    return {
        "today": max(0, int(round(today_value))),
        "tomorrow": max(0, int(round(predicted_tomorrow))),
        "advice": advice_for_aqi(predicted_tomorrow),
        "category": classify_aqi(predicted_tomorrow),
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "history_labels": history["date"].dt.strftime("%d %b").tolist(),
        "history_values": history["avg_aqi"].round().astype(int).tolist(),
        "temperature": today_temperature,
        "model_name": model_name,
        "station_count": int(daily.iloc[-1]["station_count"]),
        "policy_level": policy_level,
        "policy_action": get_policy_action(predicted_tomorrow),
        "health_suggestion": advice_for_aqi(predicted_tomorrow),
    }

def get_station_30day_chart(station_name: str) -> dict[str, Any]:
    station_series = get_station_series(station_name)

    if station_series.empty:
        return {"labels": [], "actual": [], "smoothed": [], "forecast": [], "forecast_labels": []}

    last30 = station_series.tail(30).copy()
    actual = last30["aqi"].round(1).tolist()
    labels = last30["date"].dt.strftime("%d %b").tolist()

    # Rolling smooth (window=3, center=True)
    smoothed = last30["aqi"].rolling(window=3, center=True, min_periods=1).mean().round(1).tolist()

    # Next 4 days forecast
    daily = load_daily_aqi()
    model, _ = load_or_train_model()
    feature_columns = get_feature_columns()
    training_frame = build_training_frame(daily)

    forecast_values = []
    forecast_labels = []

    if not training_frame.empty:
        last_row = training_frame.iloc[-1].copy()
        last_date = last30["date"].iloc[-1]

        for i in range(1, 5):
            pred = float(np.clip(
                model.predict(last_row[feature_columns].to_frame().T)[0], 0, 500
            ))
            forecast_values.append(round(pred, 1))
            forecast_labels.append(
                (last_date + pd.Timedelta(days=i)).strftime("%d %b")
            )
            # Slide window forward
            last_row["lag_7"] = last_row["lag_3"]
            last_row["lag_3"] = last_row["lag_2"]
            last_row["lag_2"] = last_row["lag_1"]
            last_row["lag_1"] = pred
            last_row["rolling_mean_3"] = np.mean([pred, last_row["lag_2"], last_row["lag_3"]])
            last_row["rolling_mean_7"] = last_row["rolling_mean_7"] * 0.85 + pred * 0.15
            last_row["month"] = (last_date + pd.Timedelta(days=i)).month
            last_row["day"] = (last_date + pd.Timedelta(days=i)).day
            last_row["day_of_week"] = (last_date + pd.Timedelta(days=i)).dayofweek
            last_row["day_of_year"] = (last_date + pd.Timedelta(days=i)).timetuple().tm_yday

    return {
        "labels": labels,
        "actual": actual,
        "smoothed": smoothed,
        "forecast_labels": forecast_labels,
        "forecast": forecast_values,
    }


def load_policy_predictions():
    if not os.path.exists("predictions.json"):
        return None
    with open("predictions.json", "r") as f:
        return json.load(f)
    
def get_policies_page_context() -> dict[str, Any]:
    daily = load_daily_aqi()
    station_rows = get_station_latest_table()
    prediction = build_prediction_payload()
    accuracy_bundle = _build_prediction_backtest()

    latest_station_row = station_rows[0] if station_rows else {
        "station": "Delhi",
        "aqi": prediction["today"],
        "status": prediction["category"],
        "latest_date": prediction["latest_date"],
        "status_class": get_status_class(float(prediction["today"])),
    }

    station_name = latest_station_row["station"]
    station_coordinates = STATION_COORDINATES.get(station_name, {})
    weather = get_station_weather_snapshot(
        station_name,
        station_coordinates.get("lat"),
        station_coordinates.get("lng"),
    )
    current_weather = weather.get("current", {})

    live_temp = _safe_float(current_weather.get("temperature_c"))
    dataset_temperature = _safe_float(daily.iloc[-1]["avg_temp"]) if not daily.empty else None
    current_temperature = live_temp if live_temp is not None else dataset_temperature
    current_aqi = float(prediction["today"])
    predicted_aqi = float(prediction["tomorrow"])
    temperature_history_unreliable = _temperature_history_looks_unreliable(daily)
    if temperature_history_unreliable and live_temp is None:
        current_temperature = None

    focus_month_number, focus_month_label = _month_focus(daily)
    temp_monthly_labels, temp_monthly_data = _build_monthly_chart_series(daily, "avg_temp", focus_month_number)
    aqi_monthly_labels, aqi_monthly_data = _build_monthly_chart_series(daily, "avg_aqi", focus_month_number)
    temp_30days_labels, temp_30days_data = _build_last_30_days_series(daily, "avg_temp")
    aqi_30days_labels, aqi_30days_data = _build_last_30_days_series(daily, "avg_aqi")

    month_slice = daily[daily["date"].dt.month == focus_month_number]
    month_temp_avg = _safe_float(month_slice["avg_temp"].mean()) if not month_slice.empty else None
    month_aqi_avg = _safe_float(month_slice["avg_aqi"].mean()) if not month_slice.empty else None
    if temperature_history_unreliable:
        month_temp_avg = None

    temperature_note = (
        f"Live weather from {station_name}"
        if live_temp is not None
        else "Historical temperature dataset needs refresh"
        if temperature_history_unreliable
        else "Using latest dataset city average temperature"
    )
    temperature_insight = (
        f"Current temperature is about {current_temperature:.1f} C against a {focus_month_label} baseline of {month_temp_avg:.1f} C."
        if current_temperature is not None and month_temp_avg is not None
        else "Temperature insight will sharpen as more daily records accumulate."
    )
    if temperature_history_unreliable:
        temperature_insight = (
            f"Current live temperature is about {current_temperature:.1f} C."
            if current_temperature is not None
            else "Historical temperature data needs to be refreshed before a reliable city baseline can be shown."
        )
    aqi_insight = f"City AQI is {int(round(current_aqi))}, with next-day forecast near {int(round(predicted_aqi))}."
    policy_insight = get_policy_action(predicted_aqi)
    general_recommendation = (
        "Combine hotspot inspections with public advisories so response stays proactive rather than reactive."
    )
    temperature_chart_warning = (
        "Historical temperature charts are hidden because the imported weather dataset does not line up with Delhi conditions. "
        "Refresh the NASA POWER source with the correct Delhi coordinates and regenerate the merged dataset to show real Celsius values."
        if temperature_history_unreliable else ""
    )
    if temperature_history_unreliable:
        temp_monthly_labels, temp_monthly_data = [], []
        temp_30days_labels, temp_30days_data = [], []

    # ─────────────────────────────────────────
    # LOAD CITY-WIDE POLICY OUTPUT ... policy has been updated
    # ─────────────────────────────────────────
    policy_json_path = "predictions.json"

    majority_policy = None
    city_policy_ranking = []
    stations = {}

    if os.path.exists(policy_json_path):
        try:
            with open(policy_json_path, "r", encoding="utf-8") as f:
                policy_data = json.load(f)

            majority_policy = policy_data.get("majority_policy")
            city_policy_ranking = policy_data.get("city_policy_ranking", [])
            stations = policy_data.get("stations", {})

        except Exception as e:
            print(f"[policy] Failed to load predictions.json: {e}")

    # Convert station dict → template-friendly list
    station_insights = []
    if isinstance(stations, dict):
        for name, data in stations.items():
            station_insights.append({
                "name": name,
                "aqi": data.get("aqi"),
                "policy": data.get("policy"),
                "policy_level": data.get("policy_level"),
            })

    
    return {
        "current_temperature": format_number(current_temperature) if current_temperature is not None else "Unavailable",
        "temperature_note": temperature_note,
        "current_aqi": int(round(current_aqi)),
        "aqi_color": get_aqi_color(current_aqi),
        "aqi_category": classify_aqi(current_aqi),
        "policy_level": prediction["policy_level"],
        "policy_level_name": f"Level {prediction['policy_level']} response",
        "policy_color": get_aqi_color(predicted_aqi),
        "mae_error": accuracy_bundle["mae"],
        "forecast_aqi": int(round(predicted_aqi)),
        "forecast_model_name": accuracy_bundle.get("model_name", prediction["model_name"]),
        "month_focus_label": focus_month_label,
        "month_focus_note": (
            temperature_chart_warning if temperature_history_unreliable
            else f"Per-year average for {focus_month_label}. If that month is sparse in any year, the chart uses the available records for that year."
        ),
        "last_30_days_note": (
            temperature_chart_warning if temperature_history_unreliable
            else f"Showing the latest {len(temp_30days_labels)} daily records. This auto-expands as new data is added."
        ),
        "temperature_policy_title": "Temperature policy suggestions",
        "temperature_policy_summary": temperature_insight,
        "temperature_policy_items": _temperature_policy_items(current_temperature, month_temp_avg),
        "temperature_health_title": "Temperature health suggestions",
        "temperature_health_summary": (
            "Daily comfort and heat-risk guidance based on current conditions."
            if temperature_history_unreliable
            else "Daily comfort and heat-risk guidance based on current conditions and seasonal baseline."
        ),
        "temperature_health_items": _temperature_health_items(current_temperature, month_temp_avg),
        "aqi_policy_title": "AQI policy suggestions",
        "aqi_policy_summary": policy_insight,
        "aqi_policy_items": _aqi_policy_items(current_aqi, predicted_aqi, latest_station_row["station"]),
        "aqi_health_title": "AQI health suggestions",
        "aqi_health_summary": prediction["advice"],
        "aqi_health_items": _aqi_health_items(current_aqi, predicted_aqi),
        "temp_monthly_labels": temp_monthly_labels,
        "temp_monthly_data": temp_monthly_data,
        "temp_30days_labels": temp_30days_labels,
        "temp_30days_data": temp_30days_data,
        "aqi_monthly_labels": aqi_monthly_labels,
        "aqi_monthly_data": aqi_monthly_data,
        "aqi_30days_labels": aqi_30days_labels,
        "aqi_30days_data": aqi_30days_data,
        "mae": accuracy_bundle["mae"],
        "rmse": accuracy_bundle["rmse"],
        "r2_score": accuracy_bundle["r2_score"],
        "classification_accuracy": accuracy_bundle["classification_accuracy"],
        "training_data_period": accuracy_bundle["training_data_period"],
        "test_data_period": accuracy_bundle["test_data_period"],
        "last_updated": accuracy_bundle["last_updated"],
        "prediction_comparisons": accuracy_bundle["prediction_comparisons"],
        "no_historical_predictions": accuracy_bundle["no_historical_predictions"],
        "policy_insight": policy_insight,
        "temperature_insight": temperature_insight,
        "aqi_insight": aqi_insight,
        "general_recommendation": general_recommendation,
        "top_station_name": latest_station_row["station"],
        "top_station_aqi": latest_station_row["aqi"],
        "top_station_status": latest_station_row["status"],
        "show_temperature_charts": not temperature_history_unreliable,
        "temperature_chart_warning": temperature_chart_warning,
        # new additons here..
        "majority_policy": majority_policy,
        "city_policy_ranking": city_policy_ranking,
        "stations": stations,
    }
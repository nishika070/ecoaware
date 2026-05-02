from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.formatters import format_number, format_int
from utils.aqi_utils import (
    classify_aqi,
    advice_for_aqi,
    get_aqi_color,
    get_status_class,
)
from utils.station_config import STATION_COORDINATES
from utils.data_utils import (
    load_station_daily_aqi,
    load_daily_aqi,
    build_training_frame,
    get_feature_columns,
)
from utils.chart_utils import get_station_30day_chart
from models.aqi_model import predict_aqi, get_policy_action, get_health_suggestion
from api.weather_service import get_station_weather_snapshot


# ─────────────────────────────────────────────────────────────────────────────
# LIVE AQI CACHE
# ─────────────────────────────────────────────────────────────────────────────

_live_aqi_cache: dict = {}
_live_aqi_cache_time: float = 0
_LIVE_AQI_TTL = 600


# ─────────────────────────────────────────────────────────────────────────────
# SAFE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        num = float(val)
        return None if pd.isna(num) else num
    except Exception:
        return None


def _safe_int(val: Any) -> int | None:
    try:
        if val is None:
            return None
        num = float(val)
        return None if pd.isna(num) else int(round(num))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_available_stations() -> list[str]:
    df = load_station_daily_aqi()
    return sorted(df["display_station"].unique().tolist())


def resolve_station_name(selected_station: str | None = None) -> str:
    stations = get_available_stations()
    if not stations:
        return "Delhi"
    return selected_station if selected_station in stations else stations[0]


def get_station_series(station_name: str) -> pd.DataFrame:
    df = load_station_daily_aqi()
    series = (
        df[df["display_station"] == station_name]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if series.empty:
        fallback = resolve_station_name()
        series = (
            df[df["display_station"] == fallback]
            .sort_values("date")
            .reset_index(drop=True)
        )
    return series


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
        aqi_value = _safe_float(row["aqi"])
        if aqi_value is None:
            continue
        results.append({
            "station":      row["display_station"],
            "aqi":          int(round(aqi_value)),
            "status":       classify_aqi(aqi_value),
            "advice":       advice_for_aqi(aqi_value),
            "latest_date":  row["date"].strftime("%d %b %Y"),
            "color":        get_aqi_color(aqi_value),
            "status_class": get_status_class(aqi_value),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# LIVE AQI (threaded, cached)
# ─────────────────────────────────────────────────────────────────────────────

def get_all_stations_live_aqi() -> dict[str, int | None]:
    global _live_aqi_cache, _live_aqi_cache_time

    if time.time() - _live_aqi_cache_time < _LIVE_AQI_TTL and _live_aqi_cache:
        return _live_aqi_cache

    results: dict[str, int | None] = {}

    def _fetch(station_name: str):
        coords  = STATION_COORDINATES.get(station_name, {})
        weather = get_station_weather_snapshot(
            station_name, coords.get("lat"), coords.get("lng")
        )
        aqi = _safe_float(weather.get("air_quality", {}).get("aqi"))
        return station_name, (int(round(aqi)) if aqi is not None else None)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch, s): s for s in get_available_stations()}
        for future in futures:
            try:
                name, aqi = future.result(timeout=8)
                results[name] = aqi
            except Exception:
                results[futures[future]] = None

    _live_aqi_cache      = results
    _live_aqi_cache_time = time.time()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION PAYLOAD
# ─────────────────────────────────────────────────────────────────────────────

def _policy_level(aqi: float) -> int:
    if aqi <= 100:  return 0
    if aqi <= 150:  return 1
    if aqi <= 200:  return 2
    if aqi <= 300:  return 3
    if aqi <= 400:  return 4
    return 5


def build_prediction_payload() -> dict[str, Any]:
    daily = load_daily_aqi()

    _empty = {
        "today": 0, "tomorrow": 0,
        "category": "Unavailable", "advice": "No data available.",
        "latest_date": "Unavailable",
        "history_labels": [], "history_values": [],
        "temperature": None, "model_name": "N/A",
        "station_count": 0, "policy_level": 0,
        "policy_action": "Dataset unavailable",
        "health_suggestion": "Dataset unavailable",
    }

    if daily.empty:
        return _empty

    frame        = build_training_frame(daily)
    feature_cols = get_feature_columns()

    if frame.empty:
        today_value        = float(daily.iloc[-1]["avg_aqi"])
        predicted_tomorrow = today_value
    else:
        latest_row = frame.iloc[-1].copy()
        try:
            coords   = STATION_COORDINATES.get("Anand Vihar", {})
            weather  = get_station_weather_snapshot("Anand Vihar", coords.get("lat"), coords.get("lng"))
            live_aqi = weather.get("air_quality", {}).get("aqi")
            live_temp = weather.get("current", {}).get("temperature_c")
            if isinstance(live_aqi, (int, float)):
                latest_row["lag_1"] = float(live_aqi)
            if isinstance(live_temp, (int, float)):
                latest_row["temp_lag_1"] = float(live_temp)
        except Exception:
            pass

        model_input = latest_row.reindex(feature_cols).to_numpy(dtype=float)
        predicted_tomorrow = float(np.clip(predict_aqi(model_input), 0, 500))
        today_value = float(latest_row.get("lag_1", daily.iloc[-1]["avg_aqi"]))

    latest_date = daily.iloc[-1]["date"]
    history     = daily.tail(7).copy()

    return {
        "today":             max(0, int(round(today_value))),
        "tomorrow":          max(0, int(round(predicted_tomorrow))),
        "advice":            advice_for_aqi(predicted_tomorrow),
        "category":          classify_aqi(predicted_tomorrow),
        "latest_date":       latest_date.strftime("%Y-%m-%d"),
        "history_labels":    history["date"].dt.strftime("%d %b").tolist(),
        "history_values":    history["avg_aqi"].round().astype(int).tolist(),
        "temperature":       _safe_float(daily.iloc[-1]["avg_temp"]),
        "model_name":        "Random Forest (Lag-based)",
        "station_count":     int(daily.iloc[-1]["station_count"]),
        "policy_level":      _policy_level(predicted_tomorrow),
        "policy_action":     get_policy_action(predicted_tomorrow),
        "health_suggestion": get_health_suggestion(predicted_tomorrow),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────

def get_home_context(selected_station: str | None = None) -> dict[str, Any]:
    station_name   = resolve_station_name(selected_station)
    station_series = get_station_series(station_name)

    coords        = STATION_COORDINATES.get(station_name, {})
    weather       = get_station_weather_snapshot(station_name, coords.get("lat"), coords.get("lng"))
    current       = weather.get("current", {})
    air           = weather.get("air_quality", {})
    hourly        = weather.get("hourly_preview", [])
    forecast_days = weather.get("forecast_days", [])
    weather_error = weather.get("source_error")

    live_us_aqi = air.get("aqi")
    today_fc    = forecast_days[0] if forecast_days else {}
    pred        = build_prediction_payload()
    live_all    = get_all_stations_live_aqi()
    station_rows = get_station_latest_table()

    if station_series.empty:
        latest_station_aqi  = float(pred["today"])
        latest_station_date = pred["latest_date"]
    else:
        last = station_series.iloc[-1]
        latest_station_aqi  = float(last["aqi"])
        latest_station_date = last["date"].strftime("%d %b %Y")

    live_aqi_numeric = _safe_float(live_us_aqi)
    if live_aqi_numeric is None:
        live_aqi_numeric = latest_station_aqi
    live_category    = classify_aqi(live_aqi_numeric)
    live_advice      = advice_for_aqi(live_aqi_numeric)

    current_temperature = _safe_float(current.get("temperature_c"))
    current_feels_like = _safe_float(current.get("feels_like_c"))
    current_rain_probability = _safe_float(today_fc.get("precip_probability"))
    current_wind_speed = _safe_float(current.get("wind_speed_kmh"))
    current_wind_direction = _safe_float(current.get("wind_direction_deg"))
    current_wind_gust = _safe_float(current.get("wind_gust_kmh"))
    current_humidity = _safe_float(current.get("humidity_percent"))
    current_precip_mm = _safe_float(current.get("precip_mm"))
    current_cloud_cover = _safe_float(current.get("cloud_cover_percent"))
    current_pressure = _safe_float(current.get("pressure_hpa"))
    current_uv_index = _safe_float(today_fc.get("uv_index"))

    # visibility
    visibility_m  = _safe_float(current.get("visibility_m"))
    visibility_km = round(float(visibility_m) / 1000, 1) if visibility_m is not None else None

    # map markers
    hotspot_markers = []
    for row in station_rows:
        c = STATION_COORDINATES.get(row["station"])
        if not c:
            continue
        live_val    = live_all.get(row["station"])
        display_aqi = _safe_float(live_val)
        if display_aqi is None:
            display_aqi = _safe_float(row["aqi"]) or 0.0
        display_aqi_int = int(round(display_aqi))
        hotspot_markers.append({
            "name":        row["station"],
            "aqi":         display_aqi_int,
            "status":      classify_aqi(float(display_aqi_int)),
            "advice":      advice_for_aqi(float(display_aqi_int)),
            "latest_date": row["latest_date"],
            "lat":         c["lat"],
            "lng":         c["lng"],
            "color":       get_aqi_color(float(display_aqi_int)),
            "radius":      min(6 + display_aqi_int / 40, 22),
        })

    return {
        "stations":         get_available_stations(),
        "selected_station": station_name,
        "chart":            get_station_30day_chart(station_series),
        "map": {
            "center":  {"lat": 28.6139, "lng": 77.2090},
            "markers": hotspot_markers,
        },
        "hourly_preview": hourly,
        "home_weather": {
            # temperature
            "temperature_c":       current_temperature,
            "feels_like_c":        current_feels_like,
            "condition":           current.get("condition"),
            "temp_max_today":      today_fc.get("max_temp"),
            "temp_min_today":      today_fc.get("min_temp"),
            "rain_probability":    current_rain_probability,
            # wind
            "wind_speed_kmh":      current_wind_speed,
            "wind_direction_deg":  current_wind_direction,
            "wind_gust_kmh":       current_wind_gust,
            # atmosphere
            "humidity_percent":    current_humidity,
            "precip_mm":           current_precip_mm,
            "cloud_cover_percent": current_cloud_cover,
            "visibility_km":       visibility_km,
            "visibility_m":        visibility_m,
            "pressure_hpa":        current_pressure,
            "uv_index":            current_uv_index,
            # AQI
            "aqi":                 int(round(live_aqi_numeric)),
            "aqi_label":           live_category,
            "aqi_class":           get_status_class(live_aqi_numeric),
            "updated_at":          weather.get("fetched_at"),
        },
        "primary_metrics": [
            {
                "title": "Temperature",
                "value": (f"{format_number(current_temperature)} °C"
                          if current_temperature is not None else "Unavailable"),
                "note":  weather_error or f"Live temperature for {station_name}",
            },
            {
                "title": "Precipitation Chance",
                "value": (f"{format_int(current_rain_probability)}%"
                          if current_rain_probability is not None else "Unavailable"),
                "note":  weather_error or "Today's maximum rain probability",
            },
            {
                "title": "AQI",
                "value": int(round(live_aqi_numeric)),
                "note":  "Live AQI from WAQI" if live_us_aqi is not None else f"Dataset AQI on {latest_station_date}",
            },
            {
                "title": "Health Advisory",
                "value": live_category,
                "note":  live_advice,
            },
        ],
        "secondary_metrics": [
            {
                "title": "Delhi Prediction",
                "value": pred["tomorrow"],
                "note":  f"{pred['model_name']} next-day AQI forecast",
            },
            {
                "title": "Monitored Stations",
                "value": pred["station_count"],
                "note":  "Coverage from the latest Delhi AQI dataset snapshot",
            },
        ],
        "advisory": {
            "headline": f"{station_name}: {current.get('condition', 'Current weather')}",
            "tag":      live_advice,
            "summary":  (
                "Live weather and pollutant feed is from Open-Meteo and WAQI. "
                f"AQI trend model trained through {pred['latest_date']}."
            ),
            "items": [
                f"Feels like: {format_number(current_feels_like)} °C." if current_feels_like is not None else "Feels-like temperature unavailable.",
                f"Humidity: {format_int(current_humidity)}%." if current_humidity is not None else "Humidity data unavailable.",
                f"Today's rain probability: {format_int(current_rain_probability)}%." if current_rain_probability is not None else "Rain probability unavailable.",
                f"Live AQI: {format_int(live_us_aqi)}." if live_us_aqi is not None else f"Latest dataset AQI: {int(round(latest_station_aqi))}.",
            ],
        },
        "temperature_chart": {
            "labels":        [d["day_label"] for d in forecast_days],
            "max_temps":     [d.get("max_temp") for d in forecast_days],   # ✅ fixed key
            "min_temps":     [d.get("min_temp") for d in forecast_days],   # ✅ fixed key
            "precip_chance": [d.get("precip_probability") for d in forecast_days],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# AQI PAGE
# ─────────────────────────────────────────────────────────────────────────────

def get_aqi_page_context(selected_station: str | None = None) -> dict[str, Any]:
    csv_rows  = get_station_latest_table()
    live_aqis = get_all_stations_live_aqi()

    station_rows = []
    for row in csv_rows:
        live_val    = live_aqis.get(row["station"])
        display_aqi = _safe_float(live_val)
        if display_aqi is None:
            display_aqi = _safe_float(row["aqi"]) or 0.0
        display_aqi_int = int(round(display_aqi))
        display_status = classify_aqi(float(display_aqi_int))
        if display_status == "Moderately Polluted":
            display_status = "Moderate"
        station_rows.append({
            "station":      row["station"],
            "aqi":          display_aqi_int,
            "status":       classify_aqi(float(display_aqi_int)),
            "display_status": display_status,
            "advice":       advice_for_aqi(float(display_aqi_int)),
            "latest_date":  row["latest_date"],
            "color":        get_aqi_color(float(display_aqi_int)),
            "status_class": get_status_class(float(display_aqi_int)),
        })
    station_rows.sort(key=lambda x: x["aqi"], reverse=True)

    if not selected_station or selected_station == "all":
        coords   = STATION_COORDINATES.get("Delhi", {"lat": 28.61, "lng": 77.23})
        weather  = get_station_weather_snapshot("Delhi", coords["lat"], coords["lng"])
        live_aqi = _safe_float(weather.get("air_quality", {}).get("aqi"))
        current_aqi = (
            int(round(live_aqi)) if live_aqi is not None
            else (int(round(sum(r["aqi"] for r in station_rows) / len(station_rows))) if station_rows else 0)
        )
        current_label    = "City Average"
        selected_station = "all"
    else:
        selected = next((r for r in station_rows if r["station"] == selected_station), None)
        coords   = STATION_COORDINATES.get(selected_station, {})
        weather  = get_station_weather_snapshot(selected_station, coords.get("lat"), coords.get("lng"))
        live_aqi = _safe_float(weather.get("air_quality", {}).get("aqi"))
        current_aqi  = int(round(live_aqi)) if live_aqi is not None else (int(round(selected["aqi"])) if selected else 0)
        current_label = selected_station

    air_quality = weather.get("air_quality", {})
    pred        = build_prediction_payload()

    return {
        "station_rows":     station_rows,
        "selected_station": selected_station,
        "current_aqi":      current_aqi,
        "current_status":   classify_aqi(current_aqi),
        "current_label":    current_label,
        "current_pollutants": {
            "pm25": air_quality.get("pm2_5"),
            "pm10": air_quality.get("pm10"),
            "no2":  air_quality.get("nitrogen_dioxide"),
            "so2":  air_quality.get("sulphur_dioxide"),
            "co":   air_quality.get("carbon_monoxide"),
            "o3":   air_quality.get("ozone"),
        },
        "predicted": {
            "aqi":    pred["tomorrow"],
            "status": classify_aqi(pred["tomorrow"]),
        },
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


# ─────────────────────────────────────────────────────────────────────────────
# POLICIES PAGE
# ─────────────────────────────────────────────────────────────────────────────

def get_policies_page_context() -> dict[str, Any]:
    pred         = build_prediction_payload()
    station_rows = get_station_latest_table()

    top        = station_rows[0] if station_rows else None
    top_name   = top["station"] if top else "N/A"
    top_aqi    = _safe_float(top["aqi"]) if top else _safe_float(pred["today"])
    if top_aqi is None:
        top_aqi = float(pred["today"])
    top_status = top["status"]  if top else classify_aqi(pred["today"])

    live_temp = None
    if top:
        coords  = STATION_COORDINATES.get(top_name, {})
        weather = get_station_weather_snapshot(top_name, coords.get("lat"), coords.get("lng"))
        live_temp = _safe_float(weather.get("current", {}).get("temperature_c"))

    current_temperature = (
        f"{live_temp:.1f}" if live_temp is not None
        else format_number(pred.get("temperature"))
    )
    temperature_note = (
        f"Live weather from {top_name}"
        if live_temp is not None
        else "Using latest dataset city average"
    )

    return {
        "current_aqi":         pred["today"],
        "forecast_aqi":        pred["tomorrow"],
        "aqi_category":        classify_aqi(pred["today"]),
        "aqi_color":           get_aqi_color(pred["today"]),
        "forecast_model_name": pred["model_name"],
        "policy_level":        pred["policy_level"],
        "policy_level_name":   f"Level {pred['policy_level']} response",
        "policy_color":        get_aqi_color(pred["tomorrow"]),
        "policy_insight":      get_policy_action(pred["tomorrow"]),
        "majority_policy":     None,
        "city_policy_ranking": [],
        "station_insights":    [],
        "current_temperature": current_temperature,
        "temperature_note":    temperature_note,
        "top_station_name":    top_name,
        "top_station_aqi":     top_aqi,
        "top_station_status":  top_status,
        "general_recommendation": pred["advice"],
        "aqi_policy_title":   "AQI Policy Actions",
        "aqi_policy_summary": get_policy_action(pred["tomorrow"]),
        "aqi_policy_items": [
            get_policy_action(pred["tomorrow"]),
            f"Prioritize enforcement near {top_name}, which leads the station AQI ranking.",
            f"Plan next-day controls around the forecast AQI of {pred['tomorrow']}.",
        ],
        "aqi_health_title":   "Health Advisory",
        "aqi_health_summary": pred["health_suggestion"],
        "aqi_health_items": [
            advice_for_aqi(pred["tomorrow"]),
            get_health_suggestion(pred["tomorrow"]),
        ],
        "temperature_policy_title":   "Temperature Policy",
        "temperature_policy_summary": f"Current temperature at {top_name}: {current_temperature} °C.",
        "temperature_policy_items": [
            "Schedule outdoor inspections earlier in the day to reduce heat stress.",
            "Maintain routine monitoring and keep contingency plans ready.",
        ],
        "temperature_health_title":   "Temperature Health",
        "temperature_health_summary": "Guidance based on current conditions.",
        "temperature_health_items": [
            "Keep water intake steady through the day.",
            "Sensitive groups should avoid abrupt exposure during peak afternoon hours.",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS PAGE
# ─────────────────────────────────────────────────────────────────────────────

def get_analysis_context(selected_station: str | None = None) -> dict[str, Any]:
    available_stations = get_available_stations()
    if selected_station == "all":
        station_name = "all"
        city_daily = load_daily_aqi()
        station_series = (
            city_daily.rename(columns={"avg_aqi": "aqi"})[["date", "aqi"]]
            if not city_daily.empty
            else pd.DataFrame(columns=["date", "aqi"])
        )
    else:
        station_name = resolve_station_name(selected_station)
        station_series = get_station_series(station_name)
    pred               = build_prediction_payload()

    coords                = STATION_COORDINATES.get(
        station_name if station_name != "all" else "Delhi",
        {"lat": 28.61, "lng": 77.23},
    )
    weather               = get_station_weather_snapshot(
        station_name if station_name != "all" else "Delhi",
        coords.get("lat"),
        coords.get("lng"),
    )
    current               = weather.get("current", {})
    forecast_days_weather = weather.get("forecast_days", [])
    live_temp             = _safe_float(current.get("temperature_c"))

    last15 = station_series.tail(15).copy() if not station_series.empty else pd.DataFrame()
    if not last15.empty:
        current_aqi = int(round(float(last15.iloc[-1]["aqi"])))
    else:
        current_aqi = int(pred["today"])

    forecast_values: list[int] = []
    weather_forecast = forecast_days_weather[:5]
    if not last15.empty and len(last15) >= 5:
        recent_aqi = last15["aqi"].tail(5).tolist()
        if len(recent_aqi) >= 2:
            trend = (recent_aqi[-1] - recent_aqi[0]) / 4
        else:
            trend = 0
        for i in range(5):
            next_aqi = recent_aqi[-1] + (trend * (i + 1))
            forecast_values.append(max(0, min(500, int(round(next_aqi)))))
    else:
        forecast_values = [pred["tomorrow"]] * 5

    forecast_days_out = []
    for i in range(5):
        wd = weather_forecast[i] if i < len(weather_forecast) else {}
        aqi_val = forecast_values[i] if i < len(forecast_values) else pred["tomorrow"]
        date_str = f"{wd.get('day_label', f'Day {i+1}')} {wd.get('date_label', '')}".strip()
        forecast_days_out.append({
            "date": date_str or f"Day {i+1}",
            "aqi": aqi_val,
            "aqi_color": get_aqi_color(aqi_val),
            "category": classify_aqi(aqi_val),
            "health_advisory": advice_for_aqi(aqi_val),
            "temp_max": wd.get("max_temp", "--"),
            "temp_min": wd.get("min_temp", "--"),
            "precipitation": wd.get("precip_probability", "--"),
            "humidity": _safe_float(current.get("humidity_percent")) if current else None,
            "wind_speed": (
                f"{int(round(wd['wind_speed_kmh']))} km/h"
                if wd.get("wind_speed_kmh") is not None else "--"
            ),
        })

    while len(forecast_days_out) < 5:
        idx = len(forecast_days_out)
        aqi_val = forecast_values[idx] if idx < len(forecast_values) else pred["tomorrow"]
        forecast_days_out.append({
            "date": f"Day {idx + 1}",
            "aqi": aqi_val,
            "aqi_color": get_aqi_color(aqi_val),
            "category": classify_aqi(aqi_val),
            "health_advisory": advice_for_aqi(aqi_val),
            "temp_max": "--",
            "temp_min": "--",
            "precipitation": "--",
            "humidity": "--",
            "wind_speed": "--",
        })

    forecast_avg = sum(forecast_values) / len(forecast_values) if forecast_values else 0
    forecast_trend = "Stable"
    if len(forecast_values) >= 2:
        if forecast_values[-1] > forecast_values[0]:
            forecast_trend = "Increasing"
        elif forecast_values[-1] < forecast_values[0]:
            forecast_trend = "Decreasing"

    aqi_trend = "Stable"
    if not last15.empty and len(last15) >= 2:
        aqi_vals = last15["aqi"].values
        if aqi_vals[-1] < aqi_vals[0]:
            aqi_trend = "Improving"
        elif aqi_vals[-1] > aqi_vals[0]:
            aqi_trend = "Deteriorating"

    temp_trend = "Stable"
    weather_today_temp = _safe_float(current.get("temperature_c")) if current else None
    if weather_today_temp is not None and len(weather_forecast) > 0:
        tmr_temp = weather_forecast[0].get("max_temp")
        if tmr_temp is not None:
            if tmr_temp > weather_today_temp:
                temp_trend = "Rising"
            elif tmr_temp < weather_today_temp:
                temp_trend = "Falling"

    category_counts = {"Good": 0, "Satisfactory": 0, "Moderate": 0, "Poor": 0, "Very Poor": 0, "Severe": 0}
    if not last15.empty:
        for _, row in last15.iterrows():
            cat = classify_aqi(float(row["aqi"]))
            if cat == "Moderately Polluted":
                category_counts["Moderate"] += 1
            elif cat in category_counts:
                category_counts[cat] += 1

    for day in forecast_days_out:
        cat = day["category"]
        if cat == "Moderately Polluted":
            category_counts["Moderate"] += 1
        elif cat in category_counts:
            category_counts[cat] += 1

    category_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    category_labels = [c for c in category_order if category_counts[c] > 0]
    category_data = [category_counts[c] for c in category_labels]
    category_colors = ["#2e9f57", "#8abf2f", "#d2a819", "#e67e22", "#d55353", "#7a0019"]
    category_color_list = [category_colors[category_order.index(c)] for c in category_labels]

    hist_labels = last15["date"].dt.strftime("%d %b").tolist() if not last15.empty else []
    hist_values = last15["aqi"].round(1).tolist() if not last15.empty else []
    fc_labels = [d["date"] for d in forecast_days_out]
    fc_values = [d["aqi"] for d in forecast_days_out]

    all_labels = hist_labels + fc_labels
    aqi_historical = hist_values + [None] * len(fc_labels)
    aqi_forecast = [None] * len(hist_labels) + fc_values

    if len(hist_values) >= 2:
        x = np.arange(len(hist_values), dtype=float)
        coeffs = np.polyfit(x, hist_values, 1)
        trend_y = np.polyval(coeffs, x).round(1).tolist()
        aqi_trend_line = trend_y + [None] * len(fc_labels)
    else:
        aqi_trend_line = [None] * len(all_labels)

    weather_labels = [d.get("day_label", f"Day {i+1}") for i, d in enumerate(weather_forecast)]
    temperature_data = [d.get("max_temp") for d in weather_forecast]
    precipitation_data = [d.get("precip_probability") for d in weather_forecast]

    analysis_data = {
        "date_range": (
            f"{last15.iloc[0]['date'].strftime('%d %b')} - {last15.iloc[-1]['date'].strftime('%d %b %Y')}"
            if not last15.empty else "No data"
        ),
        "current_aqi": current_aqi,
        "current_aqi_color": get_aqi_color(float(current_aqi)),
        "current_aqi_status": classify_aqi(float(current_aqi)),
        "forecast_avg_aqi": int(round(forecast_avg)),
        "forecast_avg_color": get_aqi_color(float(forecast_avg)),
        "forecast_trend": forecast_trend,
        "current_temp": int(round(weather_today_temp)) if weather_today_temp is not None else None,
        "temp_trend": temp_trend,
        "confidence": 75,
        "model_name": pred.get("model_name", "ML Model"),
        "forecast_days": forecast_days_out,
        "tips": [
            {
                "icon": "🏃",
                "title": "Outdoor Exercise",
                "description": advice_for_aqi(pred["tomorrow"]),
            },
            {
                "icon": "🪟",
                "title": "Windows & Ventilation",
                "description": "Keep windows closed if AQI worsens.",
            },
            {
                "icon": "😷",
                "title": "Mask Guidance",
                "description": get_health_suggestion(pred["tomorrow"]),
            },
        ],
        "insights": [
            {
                "icon": "📈",
                "title": "AQI Trend",
                "description": f"AQI is {aqi_trend.lower()} over the last 15 days.",
            },
            {
                "icon": "🌡️",
                "title": "Temperature Impact",
                "description": f"Temperature is {temp_trend.lower()}.",
            },
            {
                "icon": "🎯",
                "title": "Forecast Accuracy",
                "description": f"75% accuracy based on historical model predictions using {pred.get('model_name', 'ML model')}.",
            },
        ],
        "chart_json": {
            "aqi_labels": all_labels,
            "aqi_historical": aqi_historical,
            "aqi_forecast": aqi_forecast,
            "aqi_trend": aqi_trend_line,
            "weather_labels": weather_labels,
            "temperature_data": temperature_data,
            "precipitation_data": precipitation_data,
            "category_labels": category_labels,
            "category_data": category_data,
            "category_colors": category_color_list,
        },
        "correlation_data": {},
    }

    return {
        "stations": available_stations,
        "selected_station": station_name,
        "analysis_data": analysis_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONTACT PAGE
# ─────────────────────────────────────────────────────────────────────────────

def get_contact_page_context() -> dict[str, Any]:
    pred = build_prediction_payload()
    return {
        "contact_cards": [
            {"title": "Project Email",      "detail": "ecoaware.delhi@project.org"},
            {"title": "Model In Use",       "detail": pred["model_name"]},
            {"title": "Current Prediction", "detail": f"Tomorrow AQI forecast: {pred['tomorrow']} ({pred['category']})"},
        ],
        "collaboration_points": [
            "AQI model improvement and retraining with latest CPCB records",
            "Hospital and school level health-alert automation",
            "Ward-level intervention planning for dust and traffic control",
            "Dashboard extensions for heat stress and rainfall risk response",
        ],
    }

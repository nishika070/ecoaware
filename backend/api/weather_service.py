from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
from dotenv import load_dotenv
load_dotenv(dotenv_path=r"D:\ecoaware-project\backend\api\token.env")


WAQI_TOKEN = os.getenv("WAQI_TOKEN", "")   # ← set this from config/env later

print(f"Token loaded: '{WAQI_TOKEN}'")
WAQI_GEO_BASE_URL = "https://api.waqi.info/feed/geo:{lat};{lng}/?token={token}"
 # ← set this from config/env later

FORECAST_BASE_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

_CACHE: dict[str, tuple[datetime, dict[str, Any]]] = {}
_CACHE_TTL = timedelta(minutes=15)

WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _weather_label(code: Any) -> str:
    if code is None:
        return "Unknown"
    return WEATHER_CODE_MAP.get(int(code), "Unknown")


def _fetch_json(base_url: str, query: dict[str, Any]) -> dict[str, Any]:
    url = f"{base_url}?{urlencode(query, doseq=True)}"
    with urlopen(url, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def _find_latest_value(values: list[Any]) -> Any:
    for value in reversed(values):
        if value is not None:
            return value
    return None


def _build_hourly_preview(hourly_payload: dict[str, Any], current_time: str | None) -> list[dict[str, Any]]:
    times = hourly_payload.get("time", [])
    if not times:
        return []

    start_index = 0
    if current_time and current_time in times:
        start_index = times.index(current_time)

    preview: list[dict[str, Any]] = []
    for index in range(start_index, min(start_index + 8, len(times))):
        stamp = times[index]
        label = stamp[11:16] if len(stamp) >= 16 else stamp
        preview.append(
            {
                "time": stamp,
                "label": label,
                "temp_c": (hourly_payload.get("temperature_2m") or [None])[index],
                "precip_probability": (hourly_payload.get("precipitation_probability") or [None])[index],
                "condition": _weather_label((hourly_payload.get("weather_code") or [None])[index]),
            }
        )

    return preview


def _fetch_waqi_geo(lat: float, lon: float) -> dict[str, Any]:
    """Fetch AQI data from WAQI using geo coordinates."""
    url = WAQI_GEO_BASE_URL.format(lat=lat, lng=lon, token=WAQI_TOKEN)
    try:
        with urlopen(url, timeout=12) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data
    except Exception:
        return {"status": "error"}
    


def get_station_weather_snapshot(
    station_name: str,
    lat: float | None,
    lon: float | None,
) -> dict[str, Any]:
    if lat is None or lon is None:
        return {
            "station": station_name,
            "source_error": "Station coordinates missing.",
            "current": {},
            "forecast_days": [],
            "air_quality": {},
            "fetched_at": None,
        }

    cache_key = f"{station_name}:{lat}:{lon}"
    now_utc = datetime.now(timezone.utc)
    cached = _CACHE.get(cache_key)
    if cached and now_utc - cached[0] <= _CACHE_TTL:
        return cached[1]

    # ============ 1. FETCH OPEN‑METEO WEATHER  ============
    try:
        forecast_payload = _fetch_json(
            FORECAST_BASE_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "timezone": "auto",
                "forecast_days": 5,
                "current": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation",
                    "weather_code",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "wind_gusts_10m",
                    "surface_pressure",
                    "cloud_cover",
                    "visibility",
                ],
                "hourly": [
                    "temperature_2m",
                    "precipitation_probability",
                    "weather_code",
                ],
                "daily": [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max",
                    "precipitation_sum",
                    "uv_index_max",
                ],
            },
        )
    except Exception:
        return {
            "station": station_name,
            "source_error": "Open-Meteo weather request failed.",
            "current": {},
            "forecast_days": [],
            "air_quality": {},
            "fetched_at": None,
        }

    # Build forecast_days from Open‑Meteo:
    current = forecast_payload.get("current", {})
    hourly_weather = forecast_payload.get("hourly", {})
    daily = forecast_payload.get("daily", {})

    forecast_days: list[dict[str, Any]] = []
    times = daily.get("time", [])
    for index, date_str in enumerate(times[:5]):
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
        forecast_days.append(
            {
                "date": date_str,
                "day_label": parsed_date.strftime("%a"),
                "date_label": parsed_date.strftime("%d %b"),
                "condition": _weather_label((daily.get("weather_code") or [None])[index]),
                "max_temp": (daily.get("temperature_2m_max") or [None])[index],
                "min_temp": (daily.get("temperature_2m_min") or [None])[index],
                "precip_probability": (daily.get("precipitation_probability_max") or [None])[index],
                "precip_mm": (daily.get("precipitation_sum") or [None])[index],
                "uv_index": (daily.get("uv_index_max") or [None])[index],
            }
        )

    # ============ 2. FETCH WAQI AQI (with capping) ============
    try:
        waqi_data = _fetch_waqi_geo(lat, lon)
    except Exception:
        waqi_data = {"status": "error"}

    air_quality = {}
    if waqi_data.get("status") == "ok":
        data = waqi_data["data"]
        iaqi = data.get("iaqi", {})
        obs_time = data.get("time", {}).get("s", "Unknown")
        
        # Get raw AQI and cap it to 500
        raw_aqi = data.get("aqi")
        capped_aqi = min(raw_aqi, 500) if raw_aqi is not None else None
        
        air_quality = {
            "aqi": capped_aqi,  # ← Capped to 500
            "pm2_5": (iaqi.get("pm25") or {}).get("v"),
            "pm10": (iaqi.get("pm10") or {}).get("v"),
            "nitrogen_dioxide": (iaqi.get("no2") or {}).get("v"),
            "ozone": (iaqi.get("o3") or {}).get("v"),
            "sulphur_dioxide": (iaqi.get("so2") or {}).get("v"),
            "carbon_monoxide": (iaqi.get("co") or {}).get("v"),
            "observed_at": obs_time,
        }
    else:
        air_quality = {
            "aqi": None,
            "pm2_5": None,
            "pm10": None,
            "nitrogen_dioxide": None,
            "ozone": None,
            "sulphur_dioxide": None,
            "carbon_monoxide": None,
            "observed_at": None,
        }

    # ============ 3. BUILD SNAPSHOT  ============
    snapshot = {
        "station": station_name,
        "source_error": None,
        "current": {
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "wind_direction_deg": current.get("wind_direction_10m"),
            "wind_gust_kmh": current.get("wind_gusts_10m"),
            "pressure_hpa": current.get("surface_pressure"),
            "cloud_cover_percent": current.get("cloud_cover"),
            "visibility_m": current.get("visibility"),
            "precip_mm": current.get("precipitation"),
            "condition": _weather_label(current.get("weather_code")),
        },
        "hourly_preview": _build_hourly_preview(hourly_weather, current.get("time")),
        "forecast_days": forecast_days,
        "air_quality": air_quality,
        "fetched_at": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
    }

    _CACHE[cache_key] = (now_utc, snapshot)
    return snapshot
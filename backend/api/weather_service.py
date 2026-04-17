from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


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

        air_payload = _fetch_json(
            AIR_QUALITY_BASE_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "timezone": "auto",
                "hourly": [
                    "us_aqi",
                    "pm2_5",
                    "pm10",
                    "nitrogen_dioxide",
                    "ozone",
                    "sulphur_dioxide",
                    "carbon_monoxide",
                ],
            },
        )
    except Exception:
        return {
            "station": station_name,
            "source_error": "Open-Meteo request failed.",
            "current": {},
            "forecast_days": [],
            "air_quality": {},
            "fetched_at": None,
        }

    current = forecast_payload.get("current", {})
    daily = forecast_payload.get("daily", {})
    hourly = air_payload.get("hourly", {})

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

    air_quality = {
        "us_aqi": _find_latest_value(hourly.get("us_aqi", [])),
        "pm2_5": _find_latest_value(hourly.get("pm2_5", [])),
        "pm10": _find_latest_value(hourly.get("pm10", [])),
        "nitrogen_dioxide": _find_latest_value(hourly.get("nitrogen_dioxide", [])),
        "ozone": _find_latest_value(hourly.get("ozone", [])),
        "sulphur_dioxide": _find_latest_value(hourly.get("sulphur_dioxide", [])),
        "carbon_monoxide": _find_latest_value(hourly.get("carbon_monoxide", [])),
    }

    snapshot = {
        "station": station_name,
        "source_error": None,
        "current": {
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "precip_mm": current.get("precipitation"),
            "condition": _weather_label(current.get("weather_code")),
        },
        "forecast_days": forecast_days,
        "air_quality": air_quality,
        "fetched_at": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
    }

    _CACHE[cache_key] = (now_utc, snapshot)
    return snapshot

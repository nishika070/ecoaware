from __future__ import annotations
import pandas as pd
from functools import lru_cache
from pathlib import Path
from typing import Any

from utils.formatters import format_station_name


# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "datasets" / "Merged_all_readable.csv"
DATASET_SCALED_PATH = BASE_DIR / "datasets" / "Merged_all_scaled.csv"


# ----------------------------------------------------------------------------
# BASIC LOADERS
# ----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_aqi_data() -> pd.DataFrame:
    try:
        return pd.read_csv(DATASET_PATH)
    except Exception:
        return pd.DataFrame()


# optional (only if needed later)
@lru_cache(maxsize=1)
def load_scaled_data() -> pd.DataFrame:
    try:
        return pd.read_csv(DATASET_SCALED_PATH)
    except Exception:
        return pd.DataFrame()


# ----------------------------------------------------------------------------
# STATION LEVEL DATA
# ----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_station_daily_aqi() -> pd.DataFrame:
    try:
        raw = pd.read_csv(DATASET_PATH)
    except Exception:
        return pd.DataFrame(columns=["date", "aqi", "temperature", "station", "display_station"])

    raw.columns = raw.columns.str.strip()

    required_columns = ["YEAR", "DOY", "T2M", "AQI", "Location"]
    if not all(col in raw.columns for col in required_columns):
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


# ----------------------------------------------------------------------------
# DAILY AGGREGATED DATA
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------------------------------

def build_training_frame(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty or not {"avg_aqi", "avg_temp", "date"}.issubset(daily.columns):
        return pd.DataFrame()

    frame = daily.copy()
    frame = frame.sort_values("date").reset_index(drop=True)

    # SAFETY
    frame["avg_aqi"] = pd.to_numeric(frame["avg_aqi"], errors="coerce")
    frame["avg_temp"] = pd.to_numeric(frame["avg_temp"], errors="coerce")

    if len(frame) < 10:
        return pd.DataFrame()

    # LAG FEATURES
    frame["lag_1"] = frame["avg_aqi"].shift(1)
    frame["lag_2"] = frame["avg_aqi"].shift(2)
    frame["lag_3"] = frame["avg_aqi"].shift(3)
    frame["lag_7"] = frame["avg_aqi"].shift(7)

    # ROLLING
    frame["rolling_mean_3"] = frame["avg_aqi"].shift(1).rolling(3).mean()
    frame["rolling_mean_7"] = frame["avg_aqi"].shift(1).rolling(7).mean()

    # TEMP FEATURES
    frame["temp_lag_1"] = frame["avg_temp"].shift(1)
    frame["temp_rolling_mean_3"] = frame["avg_temp"].shift(1).rolling(3).mean()

    # DATE FEATURES
    frame["month"] = frame["date"].dt.month
    frame["day"] = frame["date"].dt.day
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_of_year"] = frame["date"].dt.dayofyear

    # TARGET
    frame["target"] = frame["avg_aqi"].shift(-1)
    frame["target_date"] = frame["date"].shift(-1)

    return frame.dropna().reset_index(drop=True)

def get_feature_columns() -> list[str]:
    return [
        "lag_1", "lag_2", "lag_3", "lag_7",
        "rolling_mean_3", "rolling_mean_7",
        "temp_lag_1", "temp_rolling_mean_3",
        "month", "day", "day_of_week", "day_of_year",
    ]
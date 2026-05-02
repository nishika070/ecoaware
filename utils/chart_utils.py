from __future__ import annotations
import pandas as pd
import numpy as np

from utils.data_utils import load_daily_aqi, build_training_frame


# ----------------------------------------------------------------------------
# MONTHLY CHART
# ----------------------------------------------------------------------------

def build_monthly_chart_series(daily: pd.DataFrame, value_column: str, month_number: int):
    month_frame = daily[daily["date"].dt.month == month_number].copy()

    if month_frame.empty:
        return [], []

    grouped = (
        month_frame.groupby(month_frame["date"].dt.year)[value_column]
        .mean()
        .reset_index(name="value")
        .rename(columns={"date": "year"})
        .sort_values("year")
    )

    return (
        grouped["year"].astype(int).astype(str).tolist(),
        grouped["value"].round(1).tolist(),
    )


# ----------------------------------------------------------------------------
# LAST 30 DAYS
# ----------------------------------------------------------------------------

def build_last_30_days_series(daily: pd.DataFrame, value_column: str):
    recent = daily.tail(30).copy()

    if recent.empty:
        return [], []

    return (
        recent["date"].dt.strftime("%d %b").tolist(),
        recent[value_column].round(1).tolist(),
    )


# ----------------------------------------------------------------------------
# STATION CHART
# ----------------------------------------------------------------------------

def get_station_30day_chart(station_series: pd.DataFrame):
    if station_series.empty:
        return {
            "labels": [],
            "actual": [],
            "smoothed": [],
            "forecast": [],
            "forecast_labels": [],
        }

    last30 = station_series.tail(30).copy()

    labels = last30["date"].dt.strftime("%d %b").tolist()
    actual = last30["aqi"].round(1).tolist()

    # smoothing (moving average)
    smoothed = (
        last30["aqi"]
        .rolling(window=3, center=True, min_periods=1)
        .mean()
        .round(1)
        .tolist()
    )

    # ----------------------------------------------------------------------------
    # FORECAST (fallback rolling lag logic)
    # ----------------------------------------------------------------------------

    daily = load_daily_aqi()
    training_frame = build_training_frame(daily)

    forecast_values = []
    forecast_labels = []

    if not training_frame.empty:
        last_row = training_frame.iloc[-1].copy()
        last_date = last30["date"].iloc[-1]

        for i in range(1, 5):
            # fallback: use last known lag_1
            pred = float(np.clip(last_row["lag_1"], 0, 500))

            forecast_values.append(round(pred, 1))
            forecast_labels.append(
                (last_date + pd.Timedelta(days=i)).strftime("%d %b")
            )

            # shift window (simulate next step)
            last_row["lag_7"] = last_row["lag_3"]
            last_row["lag_3"] = last_row["lag_2"]
            last_row["lag_2"] = last_row["lag_1"]
            last_row["lag_1"] = pred

    return {
        "labels": labels,
        "actual": actual,
        "smoothed": smoothed,
        "forecast_labels": forecast_labels,
        "forecast": forecast_values,
    }
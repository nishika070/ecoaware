from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from station_climate import STATION_CLIMATE
from station_map import STATION_COORDINATES

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "datasets" / "Data_training" / "AQI_merged_all.csv"
MODEL_DIR = BASE_DIR / "backend" / "models"
RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
DECISION_TREE_MODEL_PATH = MODEL_DIR / "decision_tree_model.joblib"

MONTH_COLUMNS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

MONTH_TO_NUMBER = {month: index + 1 for index, month in enumerate(MONTH_COLUMNS)}


@dataclass
class AqiPredictionBundle:
    today: int
    tomorrow: int
    advice: str
    category: str
    latest_date: str
    history_labels: list[str]
    history_values: list[int]
    temperature: None
    model_name: str
    station_count: int


def format_station_name(raw_name: str) -> str:
    cleaned = raw_name.replace("_", " ")

    for suffix in [
        " Delhi DPCC",
        " Delhi CPCB",
        " Delhi IMD",
        " Delhi IITM",
    ]:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]

    cleaned = cleaned.replace("(T3)", "T3")
    return " ".join(cleaned.split())


@lru_cache(maxsize=1)
def load_or_train_model() -> tuple[Any, str]:
    """
    Use an imported model file when available.
    Otherwise train a fresh model from the project dataset.
    """
    if joblib and RANDOM_FOREST_MODEL_PATH.exists():
        return joblib.load(RANDOM_FOREST_MODEL_PATH), "Random Forest"

    if joblib and DECISION_TREE_MODEL_PATH.exists():
        return joblib.load(DECISION_TREE_MODEL_PATH), "Decision Tree"

    training_frame = build_training_frame(load_daily_aqi())
    feature_columns = get_feature_columns()

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=2,
    )

    try:
        model.fit(training_frame[feature_columns], training_frame["target"])
        return model, "Random Forest"
    except ValueError:
        fallback_model = DecisionTreeRegressor(random_state=42, max_depth=8)
        fallback_model.fit(training_frame[feature_columns], training_frame["target"])
        return fallback_model, "Decision Tree"


@lru_cache(maxsize=1)
def load_daily_aqi() -> pd.DataFrame:
    station_daily = load_station_daily_aqi()
    daily = (
        station_daily.groupby("date", as_index=False)
        .agg(
            avg_aqi=("aqi", "mean"),
            station_count=("station", "nunique"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily["avg_aqi"] = daily["avg_aqi"].round(2)
    return daily


@lru_cache(maxsize=1)
def load_station_daily_aqi() -> pd.DataFrame:
    raw = pd.read_csv(DATASET_PATH)

    long_frame = raw.melt(
        id_vars=["Date", "station", "year"],
        value_vars=MONTH_COLUMNS,
        var_name="month_name",
        value_name="aqi",
    )

    long_frame["aqi"] = pd.to_numeric(long_frame["aqi"], errors="coerce")
    long_frame["day"] = pd.to_numeric(long_frame["Date"], errors="coerce")
    long_frame["year"] = pd.to_numeric(long_frame["year"], errors="coerce")
    long_frame["month"] = long_frame["month_name"].map(MONTH_TO_NUMBER)

    long_frame = long_frame.dropna(subset=["aqi", "day", "month", "year"]).copy()
    long_frame["day"] = long_frame["day"].astype(int)
    long_frame["year"] = long_frame["year"].astype(int)
    long_frame["month"] = long_frame["month"].astype(int)

    long_frame["date"] = pd.to_datetime(
        {
            "year": long_frame["year"],
            "month": long_frame["month"],
            "day": long_frame["day"],
        },
        errors="coerce",
    )

    long_frame = long_frame.dropna(subset=["date"]).copy()
    long_frame["display_station"] = long_frame["station"].apply(format_station_name)
    long_frame["aqi"] = long_frame["aqi"].round(2)
    return long_frame.sort_values(["display_station", "date"]).reset_index(drop=True)


def build_training_frame(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()

    frame["lag_1"] = frame["avg_aqi"].shift(1)
    frame["lag_2"] = frame["avg_aqi"].shift(2)
    frame["lag_3"] = frame["avg_aqi"].shift(3)
    frame["lag_7"] = frame["avg_aqi"].shift(7)
    frame["rolling_mean_3"] = frame["avg_aqi"].shift(1).rolling(3).mean()
    frame["rolling_mean_7"] = frame["avg_aqi"].shift(1).rolling(7).mean()
    frame["month"] = frame["date"].dt.month
    frame["day"] = frame["date"].dt.day
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_of_year"] = frame["date"].dt.dayofyear
    frame["target"] = frame["avg_aqi"].shift(-1)

    frame = frame.dropna().reset_index(drop=True)
    return frame


def get_feature_columns() -> list[str]:
    return [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "rolling_mean_3",
        "rolling_mean_7",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
    ]


def classify_aqi(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    return "Poor"


def advice_for_aqi(aqi: float) -> str:
    if aqi <= 50:
        return "Safe to go outside"
    if aqi <= 100:
        return "Limit outdoor activity"
    return "Avoid going out"


def get_aqi_color(aqi: float) -> str:
    if aqi <= 50:
        return "#2e9f57"
    if aqi <= 100:
        return "#d2a819"
    return "#d55353"


def get_climate_snapshot(station_name: str) -> dict[str, Any]:
    return STATION_CLIMATE.get(
        station_name,
        {"temperature_c": None, "precipitation_chance": None},
    )


def get_available_stations() -> list[str]:
    station_daily = load_station_daily_aqi()
    return sorted(station_daily["display_station"].unique().tolist())


def get_station_series(station_name: str) -> pd.DataFrame:
    station_daily = load_station_daily_aqi()
    station_series = (
        station_daily[station_daily["display_station"] == station_name]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if station_series.empty:
        fallback_station = get_available_stations()[0]
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
        results.append(
            {
                "station": row["display_station"],
                "aqi": int(round(aqi_value)),
                "status": classify_aqi(aqi_value),
                "advice": advice_for_aqi(aqi_value),
                "latest_date": row["date"].strftime("%d %b %Y"),
            }
        )

    return results


def get_monthly_pattern() -> list[dict[str, Any]]:
    station_daily = load_station_daily_aqi().copy()
    station_daily["month_label"] = station_daily["date"].dt.strftime("%b")

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
    station_name = selected_station if selected_station in available_stations else available_stations[0]

    city_payload = build_prediction_payload()
    station_series = get_station_series(station_name)
    latest_station = station_series.iloc[-1]
    history = station_series.tail(7)
    climate = get_climate_snapshot(station_name)
    station_rows = get_station_latest_table()
    hotspot_markers = []

    latest_station_aqi = float(latest_station["aqi"])

    for row in station_rows:
        coordinates = STATION_COORDINATES.get(row["station"])
        if not coordinates:
            continue

        hotspot_markers.append(
            {
                "name": row["station"],
                "aqi": row["aqi"],
                "status": row["status"],
                "advice": row["advice"],
                "latest_date": row["latest_date"],
                "lat": coordinates["lat"],
                "lng": coordinates["lng"],
                "color": get_aqi_color(row["aqi"]),
            }
        )

    return {
        "stations": available_stations,
        "selected_station": station_name,
        "primary_metrics": [
            {
                "title": "Temperature",
                "value": (
                    f"{climate['temperature_c']}°C"
                    if climate["temperature_c"] is not None
                    else "Connect source"
                ),
                "note": (
                    f"Add temperature_c for {station_name} in backend/api/station_climate.py"
                    if climate["temperature_c"] is None
                    else f"Current temperature for {station_name}"
                ),
            },
            {
                "title": "Precipitation Chance",
                "value": (
                    f"{climate['precipitation_chance']}%"
                    if climate["precipitation_chance"] is not None
                    else "Connect source"
                ),
                "note": (
                    f"Add precipitation_chance for {station_name} in backend/api/station_climate.py"
                    if climate["precipitation_chance"] is None
                    else f"Current precipitation outlook for {station_name}"
                ),
            },
            {
                "title": "AQI",
                "value": int(round(latest_station_aqi)),
                "note": f"{station_name} latest reading on {latest_station['date'].strftime('%d %b %Y')}",
            },
            {
                "title": "Health Advisory",
                "value": city_payload["category"],
                "note": city_payload["advice"],
            },
        ],
        "secondary_metrics": [
            {
                "title": "Delhi Prediction",
                "value": city_payload["tomorrow"],
                "note": f"{city_payload['model_name']} next-day forecast",
            },
            {
                "title": "Monitored Stations",
                "value": city_payload["station_count"],
                "note": "Coverage from the latest Delhi AQI dataset snapshot",
            },
        ],
        "chart": {
            "labels": history["date"].dt.strftime("%d %b").tolist(),
            "aqi_values": history["aqi"].round().astype(int).tolist(),
            "prediction": city_payload["tomorrow"],
        },
        "advisory": {
            "headline": f"Delhi next-day risk: {city_payload['category']}",
            "tag": city_payload["advice"],
            "summary": f"Prediction is built from your dataset through {city_payload['latest_date']} using {city_payload['model_name']}.",
            "items": [
                city_payload["advice"],
                f"{station_name} latest AQI is {int(round(latest_station_aqi))}.",
                "Sensitive groups should reduce prolonged outdoor exposure when AQI rises.",
                "Use this page as the central dashboard before checking detailed AQI and temperature sections.",
            ],
        },
        "hero": {
            "eyebrow": "Delhi environmental intelligence platform",
            "title": "Integrated AQI, climate risk and public-health signals for decision-ready monitoring",
            "description": (
                "Track the selected Delhi station, review city-wide next-day prediction, "
                "scan current hotspot conditions and move into deeper AQI, temperature and policy pages."
            ),
        },
        "map": {
            "center": {"lat": 28.6139, "lng": 77.2090},
            "markers": hotspot_markers,
        },
    }


def get_aqi_page_context() -> dict[str, Any]:
    station_rows = get_station_latest_table()
    top_hotspots = station_rows[:4]

    return {
        "station_rows": station_rows,
        "policy_note": (
            f"{top_hotspots[0]['station']} and {top_hotspots[1]['station']} are currently the highest-pressure stations "
            "in the latest available dataset window."
            if len(top_hotspots) > 1
            else "Latest AQI records are available from your Delhi dataset."
        ),
        "top_hotspots": top_hotspots,
    }


def get_temperature_page_context(selected_station: str | None = None) -> dict[str, Any]:
    available_stations = get_available_stations()
    station_name = selected_station if selected_station in available_stations else available_stations[0]
    station_series = get_station_series(station_name)
    monthly_pattern = get_monthly_pattern()
    latest_station = station_series.iloc[-1]
    worst_month = max(monthly_pattern, key=lambda item: item["aqi"])
    best_month = min(monthly_pattern, key=lambda item: item["aqi"])

    return {
        "stations": available_stations,
        "selected_station": station_name,
        "latest_station_aqi": int(round(float(latest_station["aqi"]))),
        "latest_station_date": latest_station["date"].strftime("%d %b %Y"),
        "worst_month": worst_month,
        "best_month": best_month,
        "monthly_pattern": monthly_pattern,
        "note": (
            "Your current project dataset contains AQI records, not direct temperature readings. "
            "So this page keeps the temperature/heat-stress section structure and uses the environmental trend "
            "layer from the same Delhi dataset until you plug in your temperature model."
        ),
    }


def get_policies_page_context() -> dict[str, Any]:
    station_rows = get_station_latest_table()
    monthly_pattern = get_monthly_pattern()
    worst_month = max(monthly_pattern, key=lambda item: item["aqi"])
    top_three = station_rows[:3]

    policy_items = [
        {
            "title": "Construction intensity tracker",
            "detail": f"Prioritize dust checks near {top_three[0]['station']} because it currently shows the highest AQI pressure.",
        },
        {
            "title": "Industrial hotspot index",
            "detail": f"Latest station ranking highlights {top_three[1]['station']} as another intervention zone for emission inspections.",
        },
        {
            "title": "Seasonal burden window",
            "detail": f"{worst_month['month']} has the highest mean AQI in the dataset, so preparedness campaigns should intensify before that cycle.",
        },
        {
            "title": "Health suggestion engine",
            "detail": f"Current top-zone advice: {top_three[0]['advice']}. Use this for citizen alerts and school/outdoor planning.",
        },
    ]

    hotspot_cards = [
        {
            "name": row["station"],
            "aqi": row["aqi"],
            "status": row["status"],
            "date": row["latest_date"],
        }
        for row in top_three
    ]

    return {
        "policy_items": policy_items,
        "hotspot_cards": hotspot_cards,
    }


def get_contact_page_context() -> dict[str, Any]:
    return {
        "contact_cards": [
            {
                "title": "Email",
                "detail": "ecoaware.delhi@project.org",
            },
            {
                "title": "Project Scope",
                "detail": "AQI awareness, environmental prediction, and public-health support for Delhi.",
            },
            {
                "title": "Collaboration",
                "detail": "Open for AQI models, civic dashboards, maps, and health recommendation integration.",
            },
        ]
    }


def build_prediction_payload() -> dict[str, Any]:
    daily = load_daily_aqi()
    model, model_name = load_or_train_model()
    feature_columns = get_feature_columns()

    latest_window = build_training_frame(daily).iloc[-1].copy()
    feature_values = latest_window[feature_columns].to_frame().T
    predicted_tomorrow = float(model.predict(feature_values)[0])

    today_value = float(daily.iloc[-1]["avg_aqi"])
    latest_date = daily.iloc[-1]["date"]
    history = daily.tail(7).copy()

    bundle = AqiPredictionBundle(
        today=max(0, round(today_value)),
        tomorrow=max(0, round(predicted_tomorrow)),
        advice=advice_for_aqi(predicted_tomorrow),
        category=classify_aqi(predicted_tomorrow),
        latest_date=latest_date.strftime("%Y-%m-%d"),
        history_labels=history["date"].dt.strftime("%d %b").tolist(),
        history_values=history["avg_aqi"].round().astype(int).tolist(),
        temperature=None,
        model_name=model_name,
        station_count=int(daily.iloc[-1]["station_count"]),
    )

    return {
        "today": bundle.today,
        "tomorrow": bundle.tomorrow,
        "advice": bundle.advice,
        "category": bundle.category,
        "latest_date": bundle.latest_date,
        "history_labels": bundle.history_labels,
        "history_values": bundle.history_values,
        "temperature": bundle.temperature,
        "model_name": bundle.model_name,
        "station_count": bundle.station_count,
    }

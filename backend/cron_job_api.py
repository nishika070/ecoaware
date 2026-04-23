import csv
import datetime
import os
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv
load_dotenv(dotenv_path=r"D:\ecoaware-project\backend\api\token.env")


from api.station_map import STATION_COORDINATES  # <-- your local file
WAQI_TOKEN = os.getenv("WAQI_TOKEN", "")

WAQI_GEO_BASE_URL = "https://api.waqi.info/feed/geo:{lat};{lng}/?token={token}"
FORECAST_BASE_URL = "https://api.open-meteo.com/v1/forecast"
CSV_FILE = "data_history.csv"


def ensure_csv_header():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "source",
                "waqi_aqi",
                "waqi_o3",
                "waqi_pm25",
                "waqi_pm10",
                "waqi_no2",
                "waqi_so2",
                "waqi_temp_c",
                "waqi_humidity",
                "RH2M",
                "PRECTOTCORR",
                "WS10M",
                "WS10M_MAX",
                "WS10M_MIN",
                "lat",
                "lng",
                "station_id",
                "stationierte_name",
            ])


def fetch_waqi_data(lat, lng):
    """Fetch WAQI data for one station (single lat/lng)."""
    url = WAQI_GEO_BASE_URL.format(
        lat=lat,
        lng=lng,
        token=WAQI_TOKEN,
    )
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            print(f"WAQI failed: {resp.status_code} for {lat}, {lng}")
            return {}
        data = resp.json().get("data", {})
        iaqi = data.get("iaqi", {})
        temp = iaqi.get("t", {}).get("v")
        humidity = iaqi.get("h", {}).get("v")

        return {
            "waqi_aqi": data.get("aqi"),
            "waqi_o3": iaqi.get("o3", {}).get("v"),
            "waqi_pm25": iaqi.get("pm25", {}).get("v"),
            "waqi_pm10": iaqi.get("pm10", {}).get("v"),
            "waqi_no2": iaqi.get("no2", {}).get("v"),
            "waqi_so2": iaqi.get("so2", {}).get("v"),
            "waqi_temp_c": temp if isinstance(temp, (int, float)) else None,
            "waqi_humidity": humidity if isinstance(humidity, (int, float)) else None,
        }
    except Exception as e:
        print(f"Error fetching WAQI for {lat}, {lng}: {e}")
        return {}


def fetch_forecast_data(lat, lng):
    """Fetch only AQI‑relevant weather variables from Open‑Meteo."""
    params = {
        "latitude": lat,
        "longitude": lng,
        "current": (
            "temperature_2m,relative_humidity_2m,precipitation,"
            "wind_speed_10m,wind_speed_10m_max,wind_speed_10m_min"
        ),
    }
    try:
        resp = requests.get(FORECAST_BASE_URL, params=params, timeout=20)
        if resp.status_code != 200:
            print(f"Forecast failed: {resp.status_code} - {resp.text} for {lat}, {lng}")
            return {}

        current = resp.json().get("current", {})
        return {
            "RH2M": current.get("relative_humidity_2m"),
            "PRECTOTCORR": current.get("precipitation"),
            "WS10M": current.get("wind_speed_10m"),
            "WS10M_MAX": current.get("wind_speed_10m_max"),
            "WS10M_MIN": current.get("wind_speed_10m_min"),
            "Temp": current.get("temperature_2m"),
        }
    except Exception as e:
        print(f"Error fetching forecast for {lat}, {lng}: {e}")
        return {}


def fetch_and_write_row():
    """For each station, fetch WAQI + forecast and write 20 rows."""
    ensure_csv_header()

    stations = STATION_COORDINATES
    if len(stations) < 1:
        print("No stations returned from weather_service.")
        return

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    rows = []
    for station in stations:  # limit to 20 stations
        lat, lng = station["lat"], station["lng"]
        station_id = station.get("id")
        station_name = station.get("name", "Unknown")

        waqi = fetch_waqi_data(lat, lng)
        forecast = fetch_forecast_data(lat, lng)

        # If both calls fail, skip this station row
        if not waqi and not forecast:
            print(f"Skipping station {station_id} at {lat}, {lng} due to failures.")
            continue

        row = [
            timestamp,
            "combined_waqi_forecast",
            waqi.get("waqi_aqi"),
            waqi.get("waqi_o3"),
            waqi.get("waqi_pm25"),
            waqi.get("waqi_pm10"),
            waqi.get("waqi_no2"),
            waqi.get("waqi_so2"),
            waqi.get("waqi_temp_c"),
            waqi.get("waqi_humidity"),
            forecast.get("RH2M"),
            forecast.get("PRECTOTCORR"),
            forecast.get("WS10M"),
            forecast.get("WS10M_MAX"),
            forecast.get("WS10M_MIN"),
            lat,
            lng,
            station_id,
            station_name,
        ]
        rows.append(row)

    # Write all 20 rows at once
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Logged {len(rows)} rows at {timestamp}")


if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(
        fetch_and_write_row,
        "cron",
        hour="*",
        minute=26,
        second=0,
    )
    print("Scheduler started; will run every hour at HH:00:00 UTC.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
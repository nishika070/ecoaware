import pandas as pd
import requests

LAT = 28.5512   # Delhi
LON = 77.2736   # Delhi

#28.5512, "lng": 77.2736},
url = "https://air-quality-api.open-meteo.com/v1/air-quality"

params = {
    "latitude": LAT,
    "longitude": LON,
    "current": "us_aqi",
    "timezone": "Asia/Kolkata",
}

response = requests.get(url, params=params)
data = response.json()

# Extract current US AQI (returns array ordered by time, so take the last)
aqi = data["current"]["us_aqi"]
print("Current US AQI:", aqi)

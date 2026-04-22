import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# --- Load and filter data ---
df = pd.read_csv("..\\datasets\\Merged_All_readable.csv")

# Keep only relevant columns
cols_keep = [
    "YEAR",
    "DOY",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "PRECTOTCORR",
    "WS10M",
    "WS10M_MAX",
    "WS10M_MIN",
    "PS",
    "AQI",
    "Location",           # optional: group by station
    "hasSprinkler",
    "isIndustrial",
]
df = df[cols_keep].copy()

# Remove rows where AQI is NaN
df = df.dropna(subset=["AQI"])

# --- Create date column from YEAR and DOY ---
df["YEAR"] = df["YEAR"].astype(int)
df["DOY"] = df["DOY"].astype(int)

# DOY 1 = 1st Jan; add to Jan 1 of that year
df["date"] = df.apply(
    lambda row: datetime(row["YEAR"], 1, 1) + timedelta(days=row["DOY"] - 1),
    axis=1,
)

# --- Add lags (past AQI values) for 5‑day forecast ---
df = df.sort_values(["Location", "date"]).reset_index(drop=True)

lags = [1, 2, 3, 4, 5]  # past 5 days of AQI
for lag in lags:
    df[f"AQI_lag_{lag}"] = df.groupby("Location")["AQI"].shift(lag)

# --- Engineer additional time features ---
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear

# --- Optional: group by Location (or train one global model) ---
# Here: train one global model over all stations

# Drop NaN from lags (required for training)
df_train = df.dropna(
    subset=["AQI_lag_1", "AQI_lag_2", "AQI_lag_3", "AQI_lag_4", "AQI_lag_5"]
).copy()

# --- Features and target ---
X_cols = [
    "YEAR",
    "DOY",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "PRECTOTCORR",
    "WS10M",
    "WS10M_MAX",
    "WS10M_MIN",
    "PS",
    "hasSprinkler",
    "isIndustrial",
    "AQI_lag_1",
    "AQI_lag_2",
    "AQI_lag_3",
    "AQI_lag_4",
    "AQI_lag_5",
    "month",
    "day_of_year",
]

X = df_train[X_cols].copy()
y = df_train["AQI"]

# --- Train‑test split (temporal) ---
X["date"] = df_train["date"]  # keep date for temporal split
X.sort_values("date", inplace=True)
y = y.loc[X.index]

cutoff = X["date"].quantile(0.8)  # 80% train, 20% test by time
X_train = X[X["date"] <= cutoff].drop(columns=["date"])
X_test = X[X["date"] > cutoff].drop(columns=["date"])
y_train = y.loc[X_train.index]
y_test = y.loc[X_test.index]

# --- Scale (optional for RF, helps feature importance analysis) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train Random Forest Regressor ---
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
)
rf.fit(X_train_scaled, y_train)

# --- Predict on test set ---
y_pred = rf.predict(X_test_scaled)

print(f"R2 on test: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE on test: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

# --- Predict AQI for current day + 4 days ---
# We'll assume you have a "current_features" dict for the current day
# Example: current day = 2026‑04‑22
current_date = datetime(2026, 4, 22)

# You need to get current‑day meteorology (T2M, RH2M, etc.) from your CSV or API
# For now, take last known day for each station (or use Open‑Meteo / station data)
last_day = df[df["date"] < current_date].groupby("Location").tail(1)

pred_dfs = []
for _, row in last_day.iterrows():
    station = row["Location"]
    loc_rows = df[df["Location"] == station].tail(5).sort_values("date")  # last 5 days
    aqi_lags = loc_rows["AQI"].values[::-1][:5]  # AQI_lag_1 = most recent

    # Use current‑day weather (you would normally fetch from API)
    # Here, just reuse last known day's weather; you should overwrite with today's forecast
    current_weather = row[X_cols].drop(labels=[
        "AQI_lag_1",
        "AQI_lag_2",
        "AQI_lag_3",
        "AQI_lag_4",
        "AQI_lag_5",
        "year",
        "doy",
        "date",
        "month",
        "day_of_year",
    ], errors="ignore")

    # For 5‑day forecast: shift lags each day
    hist_aqi = aqi_lags.tolist()
    forecasts = []
    for d in range(5):  # current day + 4 days
        # Construct feature vector for this day
        feat_vec = np.array([
            current_date.year,
            current_date.timetuple().tm_yday,
            current_weather["T2M"],
            current_weather["T2M_MAX"],
            current_weather["T2M_MIN"],
            current_weather["RH2M"],
            current_weather["PRECTOTCORR"],
            current_weather["WS10M"],
            current_weather["WS10M_MAX"],
            current_weather["WS10M_MIN"],
            current_weather["PS"],
            current_weather["hasSprinkler"],
            current_weather["isIndustrial"],
            hist_aqi[0],
            hist_aqi[1],
            hist_aqi[2],
            hist_aqi[3],
            hist_aqi[4],
            current_date.month,
            current_date.timetuple().tm_yday,
        ])

        # Scale and predict
        feat_scaled = scaler.transform(feat_vec.reshape(1, -1))
        aqi_pred = rf.predict(feat_scaled)[0]
        forecasts.append(aqi_pred)

        # Roll lags: today’s prediction becomes lag 1 tomorrow
        hist_aqi = [aqi_pred] + hist_aqi[:-1]

    # Store per station forecasts
    pred_df = pd.DataFrame({
        "date": pd.date_range(start=current_date, periods=5, freq="D"),
        "Location": station,
        "AQI_forecast": forecasts,
        "day_ahead": list(range(1, 6)),
    })
    pred_dfs.append(pred_df)

final_forecast = pd.concat(pred_dfs, ignore_index=True)
print(final_forecast.head(10))

# You can save this for your HTML page:
final_forecast.to_csv("aqi_forecast_next5.csv", index=False)
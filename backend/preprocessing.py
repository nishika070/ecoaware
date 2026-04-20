import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
from sklearn.impute import KNNImputer

#load and reshape data
def load_and_merge_data(folder_path):
    dataframes = []

    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)

            try:
                # Read ONLY first 32 rows (including header), ignore everything after
                df = pd.read_excel(file_path, nrows=32)
                df.columns = df.columns.str.strip()

                # Convert wide → long format
                df = df.melt(id_vars=["Date"], var_name="Month", value_name="AQI")

                # Extract year & location from filename
                parts = file.split("_")
                df["year"] = int(parts[2]) if len(parts) > 2 else 0
                df["location"] = parts[3] if len(parts) > 3 else "unknown"

                # Remove any rows with invalid AQI values (extra junk)
                # df = df.dropna(subset=['AQI'])
                df = df[df['AQI'] >= 0]  # AQI can't be negative

                dataframes.append(df)

            except Exception as e:
                continue

    merged_df = pd.concat(dataframes, ignore_index=True)
    #print(f"Merged {len(dataframes)} files into {merged_df.shape[0]} rows")
    return merged_df

#clean data
def clean_data(df):
    imputer = KNNImputer(n_neighbors=5)
    aqi_imputed = imputer.fit_transform(df[["AQI"]]).flatten()
    df["AQI"] = aqi_imputed
    return df

#create day of year--instead of day+month 
def process_date(df):
    df["Date"] = pd.to_numeric(df["Date"], errors='coerce')
    df["Month"] = pd.to_datetime(df["Month"], format='%B').dt.month

    df["full_date"] = pd.to_datetime(
        dict(year=df["year"], month=df["Month"], day=df["Date"]),
        errors='coerce'
    )

    df["day_of_year"] = df["full_date"].dt.dayofyear
    df = df.drop(columns=["Date", "Month", "full_date"])
    return df

#encode location
def encode_data(df):
    le = LabelEncoder()
    df["location"] = le.fit_transform(df["location"].astype(str))
    joblib.dump(le, "../models/location_encoder.pkl")
    return df

#handling outliers
def handle_outliers(df):
    Q1 = df["AQI"].quantile(0.25)
    Q3 = df["AQI"].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df["AQI"] = df["AQI"].clip(lower, upper)
    return df

#data scaling
def scale_data(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "../models/data_scaler_aqi.pkl")
    return pd.DataFrame(X_scaled, columns=X.columns)

#main 
def run_preprocessing():
    folder_path = "../datasets/Data_training"
    output_path = "../datasets"

    df = load_and_merge_data(folder_path)
    df = clean_data(df)
    df = process_date(df)
    df = encode_data(df)
    df = handle_outliers(df)
    df = scale_data(df)
    df.to_csv(os.path.join(output_path, "AQI_preprocessed.csv"), index=False)


if __name__ == "__main__":
    run_preprocessing()
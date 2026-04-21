import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
from sklearn.impute import KNNImputer
import re

PATTERN = re.compile(r'^AQI_daily_(\d{4})\_(.*?)\_(\d{4})\.xlsx$')  # regex for filenames..

# encoding locations
location_map={'Anand_Vihar_Delhi_DPCC': 0, 'Ashok_Vihar_Delhi_DPCC': 1, 'Bawana_Delhi_DPCC': 2,
              'CRRI_Mathura_Road_Delhi_IMD': 3, 'DTU_Delhi_CPCB': 4,
              'IGI_Airport_(T3)_Delhi_IMD': 5, 'ITO_Delhi_CPCB': 6, 'Jahangirpuri_Delhi_DPCC': 7,
              'Jawaharlal_Nehru_Stadium_Delhi_DPCC': 8, 'Lodhi_Road_Delhi_IMD': 9,
              'Narela_Delhi_DPCC': 10, 'North_Campus_DU_Delhi_IMD': 11, 'NSIT_Dwarka_Delhi_CPCB': 12,
              'Okhla_Phase-2_Delhi_DPCC': 13, 'Patparganj_Delhi_DPCC': 14,
              'Punjabi_Bagh_Delhi_DPCC': 15, 'Rohini_Delhi_DPCC': 16, 'Sirifort_Delhi_CPCB': 17,
              'Vivek_Vihar_Delhi_DPCC': 18, 'Wazirpur_Delhi_DPCC': 19}


#load and reshape data
def load_and_merge_data(folder_path):
    dataframes = []
    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)
            match = PATTERN.match(file)
            if not match:
                continue
            year      = int(match.group(1))
            station   = match.group(2)

            try:
                df = pd.read_excel(file_path, nrows=32)
                df.columns = df.columns.str.strip()

                day_col = None
                for col in ["Date", "Day", "DATE", "DAY"]:
                    if col in df.columns:
                        day_col = col
                        break

                if day_col is None:
                    print(f"No day-like column in {file}: {list(df.columns)}")
                    continue

                df = df.melt(id_vars=[day_col], var_name="Month", value_name="AQI")

                df["Year"]      = year
                df["Location"]  = station

                df = df[df["AQI"] >= 0]

                dataframes.append(df)

            except Exception as e:
                continue

    return pd.concat(dataframes, ignore_index=True)

#clean data
def clean_data(df):
    imputer = KNNImputer(n_neighbors=5)
    aqi_imputed = imputer.fit_transform(df[["AQI"]]).flatten()
    df["AQI"] = aqi_imputed
    return df

#create day of year--instead of day+month 
def process_date(df):
    if "Date" in df.columns:
        day_col = "Date"
    elif "Day" in df.columns:
        day_col = "Day"
    else:
        raise ValueError("No day column (Date/Day) found in DataFrame")

    df["Date_numeric"] = pd.to_numeric(df[day_col], errors='coerce')
    df["Month"] = pd.to_datetime(df["Month"], format='%B').dt.month

    df["full_date"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Date_numeric"]),
        errors='coerce'
    )
    df["DOY"] = df["full_date"].dt.dayofyear
    # Drop invalid rows (no valid DOY) and cleanup
    df = df.dropna(subset=["DOY", "AQI"])
    df = df.drop(columns=["Date_numeric", "Month", "full_date"], errors="ignore")
    return df

#encode location
def encode_data(df):
    #le = LabelEncoder()
    #df["Location"] = le.fit_transform(df["Location"].astype(str))

    df["Location"] = df["Location"].map(location_map)
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
    print("Rows before drop:", df.shape)
    print("Number of rows:", df.shape)
    df = df.drop(columns=["Day","Date"], errors="ignore")
    print("Rows after drop:", df.shape)
    df = scale_data(df)
    df.to_csv(os.path.join(output_path, "AQI_preprocessed.csv"), index=False)


if __name__ == "__main__":
    run_preprocessing()
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


# LOAD + RESHAPE DATA
def load_and_merge_data(folder_path):
    dataframes = []

    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)

            try:
                df = pd.read_excel(file_path)
                df.columns = df.columns.str.strip()

                # Convert wide → long format
                df = df.melt(id_vars=["Date"], var_name="Month", value_name="AQI")

                # Add location
                parts = file.split("_")
                df["location"] = parts[3] if len(parts) > 3 else "unknown"

                dataframes.append(df)

            except:
                continue

    return pd.concat(dataframes, ignore_index=True)


# CLEAN DATA
def clean_data(df):
    df = df.dropna(subset=["AQI"])
    return df


# PROCESS DATE + MONTH
def process_date(df):
    df["Date"] = pd.to_numeric(df["Date"], errors='coerce')

    # Convert month names to numbers
    df["Month"] = pd.to_datetime(df["Month"], format='%B').dt.month

    return df


# ENCODE LOCATION
def encode_data(df):
    le = LabelEncoder()
    df["location"] = le.fit_transform(df["location"].astype(str))
    return df


# OUTLIER HANDLING
def handle_outliers(df):
    Q1 = df["AQI"].quantile(0.25)
    Q3 = df["AQI"].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df["AQI"] = df["AQI"].clip(lower, upper)

    return df


# SCALE DATA
def scale_data(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


# SPLIT DATA
def split_data(df):
    X = df[["Date", "Month", "location"]]
    y = df["AQI"]

    X = scale_data(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)


# SAVE DATA
def save_data(X_train, X_test, y_train, y_test, output_path):
    pd.DataFrame(X_train).to_csv(os.path.join(output_path, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_path, "y_test.csv"), index=False)


# MAIN
def run_preprocessing():
    folder_path = "../datasets/Data_training"
    output_path = "../datasets"

    df = load_and_merge_data(folder_path)
    df = clean_data(df)
    df = process_date(df)
    df = encode_data(df)
    df = handle_outliers(df)

    X_train, X_test, y_train, y_test = split_data(df)

    save_data(X_train, X_test, y_train, y_test, output_path)


if __name__ == "__main__":
    run_preprocessing()
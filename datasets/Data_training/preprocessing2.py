import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


df=pd.read_csv("..\\ecoaware\\datasets\\Data_training\\POWER_Point_Daily_20000101_20260415_034d16N_090d61E_LST.csv")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# instructions given that outliers -999 are actually NaNs

def missing_values(df):
    df[numeric_cols] = df[numeric_cols].replace(-999, np.nan)
    imputer=KNNImputer(n_neighbors=5)
    for col in numeric_cols:
        df[col] = imputer.fit_transform(df[[col]])
    return df

def handle_outliers(df):
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower, upper)
    return df

def normalise(df):
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def run_preprocessing():
    df = missing_values(df)
    df = handle_outliers(df)
    df = normalise(df)
    df.to_csv("..\\ecoaware\\datasets\\Data_training\\Weather_preprocessed.csv", index=False)

if __name__ == "__main__":
    run_preprocessing()
    



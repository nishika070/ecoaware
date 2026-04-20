import pandas as pd
import joblib
import numpy as np

np.set_printoptions(precision=10, suppress=True, edgeitems=10)

weather_df = pd.read_csv('..\\datasets\\Weather_preprocessed.csv')
aqi_df = pd.read_csv('..\\datasets\\AQI_preprocessed.csv')

# Load scalers and encoders
weather_scaler = joblib.load('..\\models\\data_scaler_weather.pkl')
aqi_scaler = joblib.load('..\\models\\data_scaler_aqi.pkl')
location_encoder = joblib.load('..\\models\\location_encoder.pkl')

print("Loaded scalers and encoders successfully.")

aqi_input_real = pd.DataFrame(aqi_scaler.inverse_transform(aqi_df[["AQI","year","location","day_of_year"]].values),columns=['AQI', 'YEAR', 'LOC', 'DOY'])
print("\nInverse transformed AQI values:")
print(aqi_input_real[::150])
print("\nShape of AQI input data:")
print(aqi_input_real.shape)
aqi_input_real.to_csv('..\\datasets\\AQI_temp.csv', index=False)

weather_input_real = pd.DataFrame(weather_scaler.inverse_transform(weather_df[["YEAR","DOY","T2M","T2M_MAX","T2M_MIN","RH2M","PRECTOTCORR","WS10M","WS10M_MAX","WS10M_MIN","PS"]].values), columns=["YEAR","DOY","T2M","T2M_MAX","T2M_MIN","RH2M","PRECTOTCORR","WS10M","WS10M_MAX","WS10M_MIN","PS"])
print("\nInverse transformed weather values:")
print(weather_input_real[::150]) 
print("\nShape of weather input data:")
print(weather_input_real.shape)
weather_input_real.to_csv('..\\datasets\\Weather_temp.csv', index=False)


merged_df=pd.merge(weather_input_real, aqi_input_real, on=["YEAR","DOY"], how="inner")
print("\nMerged DataFrame:")
print(merged_df[::150])
print("Shape of merged DataFrame:")
print(merged_df.shape)
merged_df.to_csv('..\\datasets\\Merged_all.csv', index=False)
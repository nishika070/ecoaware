import pandas as pd
import joblib
import numpy as np

np.set_printoptions(precision=10, suppress=True, edgeitems=10)

# encoding locations
location_map={'Anand_Vihar_Delhi_DPCC': 0, 'Ashok_Vihar_Delhi_DPCC': 1, 'Bawana_Delhi_DPCC': 2,
              'CRRI_Mathura_Road_Delhi_IMD': 3, 'DTU_Delhi_CPCB': 4,
              'IGI_Airport_(T3)_Delhi_IMD': 5, 'ITO_Delhi_CPCB': 6, 'Jahangirpuri_Delhi_DPCC': 7,
              'Jawaharlal_Nehru_Stadium_Delhi_DPCC': 8, 'Lodhi_Road_Delhi_IMD': 9,
              'Narela_Delhi_DPCC': 10, 'North_Campus_DU_Delhi_IMD': 11, 'NSIT_Dwarka_Delhi_CPCB': 12,
              'Okhla_Phase-2_Delhi_DPCC': 13, 'Patparganj_Delhi_DPCC': 14,
              'Punjabi_Bagh_Delhi_DPCC': 15, 'Rohini_Delhi_DPCC': 16, 'Sirifort_Delhi_CPCB': 17,
              'Vivek_Vihar_Delhi_DPCC': 18, 'Wazirpur_Delhi_DPCC': 19}


inv_location_map = {v: k for k, v in location_map.items()}  #inverse of above

weather_df = pd.read_csv('..\\datasets\\Weather_preprocessed.csv')
aqi_df = pd.read_csv('..\\datasets\\AQI_preprocessed.csv')

weather_scaler = joblib.load('..\\models\\data_scaler_weather.pkl')
aqi_scaler = joblib.load('..\\models\\data_scaler_aqi.pkl')
# location_encoder = joblib.load() -> not needed now cuz of manual mapping

print("Loaded scalers and encoders successfully.")

# first retreive everything inversed to merge
# later apply scaling on merged dataset
print(aqi_df.columns)
print(aqi_df.shape)
aqi_input_real = pd.DataFrame(aqi_scaler.inverse_transform(aqi_df[["AQI","Year","Location","DOY"]].values),columns=["AQI", "YEAR", "LOC", "DOY"])
print("\nInverse transformed AQI values:")
print(aqi_input_real[::150])
print("\nShape of AQI input data:")
print(aqi_input_real.shape)
aqi_input_real.to_csv('..\\datasets\\AQI_temp.csv', index=False)

weather_input_real = pd.DataFrame(weather_scaler.inverse_transform(weather_df[["YEAR","DOY","T2M","T2M_MAX","T2M_MIN","RH2M","PRECTOTCORR","WS10M","WS10M_MAX","WS10M_MIN","PS"]].values), columns=["YEAR","DOY","T2M","T2M_MAX","T2M_MIN","RH2M","PRECTOTCORR","WS10M","WS10M_MAX","WS10M_MIN","PS"])
print("\nInverse transformed weather values:")
print(weather_input_real[5800:8000:100]) 
print("\nShape of weather input data:")
print(weather_input_real.shape)
weather_input_real.to_csv('..\\datasets\\Weather_temp.csv', index=False)

merged_df=pd.merge(weather_input_real, aqi_input_real, on=["YEAR","DOY"], how="inner")
merged_df["LOC"] = merged_df["LOC"].astype(int)
merged_df["Location"] = merged_df["LOC"].map(inv_location_map)

def markSprinklers(df):
    return 1 if df["LOC"] in [1,2,7,10,13,15,16,18,19] else 0
    return df
merged_df["hasSprinkler"] = merged_df.apply(markSprinklers, axis=1)

def markIndustrial(df):
    return 1 if df["LOC"] in [1,2,7,10,14,15,16,19] else 0
merged_df["isIndustrial"] = merged_df.apply(markIndustrial, axis=1)

print("\nMerged DataFrame:")
print(merged_df[::150])
print("Shape of merged DataFrame:")
print(merged_df.shape)

merged_df.to_csv('..\\datasets\\Merged_all_readable.csv', index=False)
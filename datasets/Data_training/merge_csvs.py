import pandas as pd
import glob
import os
import re


csv_files = glob.glob("AQI_daily_*.xlsx")
pattern = re.compile(r'^AQI_daily_(\d{4})\_(.+?)\_(\d{4})\.xlsx$')
print(f"Found {len(csv_files)} files\n")

df_list=[]

# Remove limits for rows, columns, and width
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

for file in csv_files:
    filename = os.path.basename(file)
    match = pattern.search(filename)
    
    if match:
        station = match.group(2)
        year = match.group(1)
        #print(f"File: {filename} → Station: {station}, Years: {year}")

    
    df=pd.read_excel(filename)
    df["station"]=station
    df["year"]=year

    if 'Day' in df.columns:
        df = df.rename(columns={'Day': 'Date'})
        #print(f"  Renamed 'day' → 'date'")
    elif 'Date' not in df.columns:
        print(f"  Warning: Neither 'date' nor 'day' column found in {filename}")
    df_list.append(df)

df_all=pd.concat(df_list, ignore_index=True)
# Ensure consistent column order and data types
df_all['Date'] = pd.to_numeric(df_all['Date'], errors='coerce')
df_all.to_csv("AQI_merged_all.csv", index=False)
print("Merged\n\n")


print(df_all)
    

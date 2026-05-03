import pandas as pd
df = pd.read_csv('datasets/delhi25-26.csv')
print('Shape:', df.shape)
print('AQI unique values:', df['AQI'].nunique())
print('AQI min/max:', df['AQI'].min(), df['AQI'].max())
print('Any NaN in AQI:', df['AQI'].isnull().any())
print('Sample AQI:', df['AQI'].head(10).values)
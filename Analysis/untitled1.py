import pandas as pd

# import data

df = pd.read_excel('C:/Users/timon/Documents/BI-2020/Data Science/Projects/Data-Science-Project/HistoricalWeatherData.xlsx')

print(df)

# combine date and time to Date/Time

df = df.apply(lambda r : pd.datetime.combine(r['Date'],r['Time']),1)
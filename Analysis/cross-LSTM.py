# cross-LSTM

# import libraries
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False

# load dataset
df = pd.read_csv('C:/Users/timon/Documents/BI-2020/Data-Science/Projects/Data-Science-Project/Data/cross-cafe-final.csv')
df.info()

# change Datetime to datetype datetime
df['Datetime'] = pd.to_datetime(df['Datetime'], infer_datetime_format=True)
df.info()

# set Datetime as index and plot all series
df.set_index('Datetime')[['cross_cafe', 'cloud_cover', 'humidity', 'precip_dur_past1h', 'precip_past1h', 'pressure', 'temp_dew',
                          'temp_dry', 'temp_max_past1h', 'temp_mean_past1h', 'temp_min_past1h', 'wind_dir', 'wind_max_per10min_past1h',
                          'wind_speed', 'wind_speed_past1h', 'sun_last1h_glob']].plot(subplots=True)

# plot only cross_cafe
df.set_index('Datetime')['cross_cafe'].plot(subplots=False)
df['cross_cafe'].describe()

# cut of corona period and get dataframe for 2 years from 2018 to 2020
# this is the data we are going to use to test whether LSTM can yield good forecasts 
df = df[df['Datetime'] < '2020-1-2 00:00:00']
df.set_index('Datetime')['cross_cafe'].plot(subplots=False)

# create input dataframe
df_input = df[['cross_cafe', 'cloud_cover', 'humidity', 'precip_dur_past1h', 'precip_past1h', 'pressure', 'temp_dew',
               'temp_dry', 'temp_max_past1h', 'temp_mean_past1h', 'temp_min_past1h', 'wind_dir', 'wind_max_per10min_past1h',
               'wind_speed', 'wind_speed_past1h', 'sun_last1h_glob']]

pd.set_option('display.max_columns', None) 
df_input.describe() # looks alright 

# replace NA values in dataframe 
# cloud_cover
df_input['cloud_cover'] = df_input['cloud_cover'].fillna(0)

# scale input variables
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_input)
data_scaled

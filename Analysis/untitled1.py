# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:22:02 2021

@author: timon
"""

# cross LSTM model
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters

# set styles
# set seaborn style
register_matplotlib_converters()

# set seaborn style
sns.set(style='whitegrid', palette='muted', font_scale=1)

# set plotting parameters
rcParams['figure.figsize'] = 22, 10

# set random seed
random_seed = 20
np.random.seed(random_seed)

# load dataset
df = pd.read_csv('C:/Users/timon/Documents/BI-2020/Data-Science/Projects/Data-Science-Project/Data/cross-cafe-final.csv')

# change Datetime to datetype datetime
df['Datetime'] = pd.to_datetime(df['Datetime'], infer_datetime_format=True)
df.info()

# cut of corona period and get dataframe for 2 years from 2018 to 2020
# this is the data we are going to use to test whether LSTM can yield good forecasts 
df = df[df['Datetime'] < '2020-1-3 00:00:00']
# df.set_index('Datetime')['cross_cafe'].plot(subplots=False)


# create input dataframe
df_input = df[['Datetime', 'cross_cafe', 'cloud_cover', 'humidity', 'precip_dur_past1h', 'precip_past1h', 'temp_dew',
               'temp_dry', 'temp_max_past1h', 'temp_mean_past1h', 'temp_min_past1h', 'wind_dir', 'wind_max_per10min_past1h',
               'wind_speed', 'wind_speed_past1h', 'sun_last1h_glob']]

pd.set_option('display.max_columns', None) 
# df_input.describe() # looks alright 

# set Datetime as index
df_input = df_input.set_index('Datetime')
df_input.shape

# replace NA values in dataframe 
# cloud_cover
df_input.isna().sum()
# propagate last valid observation forward to next 
df_input.fillna(method='ffill', inplace = True)
# df_input.isna().sum()
# we will replace na's in cloud cover and wind_min with 0
df_input['cloud_cover'].fillna(0, inplace = True)
df_input.isna().sum()

# data exploration 
# add hour column
df_input['hour'] = df_input.index.hour
# add day of month column 0 = first day of the month
df_input['day_of_month'] = df_input.index.day
# add day of week column 0 = Monday
df_input['day_of_week'] = df_input.index.dayofweek
# add month column
df_input['month'] = df_input.index.month
# add weekend column
df_input['is_weekend'] = ((df_input.index.dayofweek) // 5 == 1).astype(float)

# create final dataframe for LSTM
df_input = df_input[['cross_cafe', 'cloud_cover', 'humidity', 'temp_dry',
                     'temp_mean_past1h', 'wind_max_per10min_past1h', 'sun_last1h_glob',
                     'hour', 'day_of_week', 'is_weekend']]

# scale variables
scaler = MinMaxScaler()
df_input_scaled = scaler.fit_transform(df_input)

#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
#In this example, the n_features is 2. We will make timesteps = 3. 
#With this, the resultant n_samples is 5 (as the input data has 9 rows).

# split data into train and test set 
training_size = int(len(df_input)*0.9)
test_size = len(df_input)-training_size
train_data, test_data = df_input[0:training_size,:], df_input[training_size:len(df_input),:1]

n_future = 24   # Number of hours we want to predict into the future
n_past = 240     # Number of lags we want to use to predict the future

def create_dataset(dataset, time_step=1):
    trainX = []
    trainY = [] 
    for i in range(n_past, len(df_input_scaled) - n_future +1):
        trainX.append(df_input_scaled[i - n_past:i, 0:df_input.shape[1]])
        trainY.append(df_input_scaled[i + n_future - 1:i + n_future, 0])
    
    return np.array(trainX), np.array(trainY)



print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# define Autoencoder model

model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# fit model
history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

# forecasting
n_future=24  #Redefining n_future to extend prediction hours beyond original n_future hours
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

forecast = model.predict(trainX[-n_future:]) #forecast 

#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform
forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]


# Convert timestamp to date
forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])


original = df[['Date', 'Open']]
original['Date']=pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2020-5-1']

sns.lineplot(original['Date'], original['Open'])
sns.lineplot(df_forecast['Date'], df_forecast['Open'])



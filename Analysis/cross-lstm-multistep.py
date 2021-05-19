# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:44:22 2021

@author: timon
"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from pylab import rcParams
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
import tensorflow as tf
from tensorflow import keras
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import gc
import sys


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
df_input = df[['Datetime','cross_cafe', 'cloud_cover', 'humidity', 'precip_dur_past1h', 'precip_past1h', 'temp_dew',
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

# # plot heatmap
# # df_input = df_input.reset_index()
# # get correlations
# df_input_corr = df_input.corr()
# # create mask
# mask = np.triu(np.ones_like(df_input_corr, dtype=np.bool))

# sns.heatmap(df_input_corr, mask=mask, annot=True, fmt=".2f", cmap='Blues',
#            vmin=-1, vmax=1, cbar_kws={"shrink": .8})

# df_input.to_csv('lstm-dataset.csv')

# data exploration
sns.lineplot(x=df_input.index, y='cross_cafe', data=df_input)

# aggregate data by month
df_by_month = df_input.resample('M').sum()
sns.lineplot(x=df_by_month.index, y='cross_cafe', data=df_by_month)

# check by hour
sns.pointplot(data=df_input, x='hour', y='cross_cafe');
# seem as if the busiest period is around 9-12 on average

# check for weekends
sns.pointplot(data=df_input, x='hour', y='cross_cafe', hue='is_weekend');

# check for weekdays
sns.pointplot(data=df_input, x='day_of_week', y='cross_cafe');

# check for day of month
sns.pointplot(data=df_input, x='day_of_month', y='cross_cafe');

# check for months
sns.pointplot(data=df_input, x='month', y='cross_cafe');


# due to high correlation between some features, the following features are dropped: temp_max_past1h, temp_min_past1h, wind_speed, wind_speed_past1h
# due to no correlation to target, the following features are dropped: precip_dur_past1h, precip_past1h, day_of month
df_input = df_input[['cross_cafe', 'cloud_cover', 'humidity', 'temp_dew',
                     'temp_dry', 'temp_max_past1h', 'temp_mean_past1h', 'wind_dir', 'wind_max_per10min_past1h',
                     'wind_speed', 'wind_speed_past1h', 'sun_last1h_glob', 'hour', 'day_of_week', 'month', 'is_weekend']]

# ----------------------------------------------------------------------------------------------------------------
# multiple regression
X = df_input[['cloud_cover', 'humidity', 'precip_dur_past1h', 'temp_dew','temp_dry', 'temp_max_past1h', 
              'temp_min_past1h', 'wind_dir', 'wind_max_per10min_past1h', 'wind_speed',
              'sun_last1h_glob', 'hour', 'day_of_month', 'day_of_week', 'month', 'is_weekend']]
y = df_input['cross_cafe']

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

model.summary()
# ----------------------------------------------------------------------------------------------------------------

# Data Loader Parameters
BATCH_SIZE = 40
BUFFER_SIZE = 1740
TRAIN_SPLIT = 15780

# LSTM Parameters
EVALUATION_INTERVAL = 200
EPOCHS = 100
PATIENCE = 5

# Reproducibility
SEED = 13
tf.random.set_seed(SEED)

# scale data input
df_input = df_input.values
data_mean = df_input[:TRAIN_SPLIT].mean(axis=0)
data_std = df_input[:TRAIN_SPLIT].std(axis=0)
df_input = (df_input-data_mean)/data_std

# create multivariate dataset for multi step forecast
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

# define model parameters
past_history = 480
future_target = 168
STEP = 1

x_train_multi, y_train_multi = multivariate_data(df_input, df_input[:, 0], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(df_input, df_input[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

print (x_train_multi.shape,
       y_train_multi.shape,
       'Single window of past history : {}'.format(x_train_multi[0].shape),
       'Target temperature to predict : {}'.format(y_train_multi[0].shape),
       sep='\n')

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

# plotting a sample data point
def create_time_steps(length):
    return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(18, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 0]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()
    

for x, y in train_data_multi.take(1):
    multi_step_plot(x[0], y[0], np.array([0]))
    
    
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(128,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(64, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(24))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
print(multi_step_model.summary()) 

for x, y in val_data_multi.take(1):
    print (multi_step_model.predict(x).shape)
    
early_stopping = EarlyStopping(monitor='val_loss', patience = 3, restore_best_weights=True)
multi_step_history = multi_step_model.fit(train_data_multi,
                                          epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=EVALUATION_INTERVAL,
                                          callbacks=[early_stopping])

multi_step_model.save('cross-multi-timesteps.h5', )

# plot results
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()
    
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

for x, y in val_data_multi.take(5):
    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
    

    
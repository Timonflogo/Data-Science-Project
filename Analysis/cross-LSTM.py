# cross-LSTM

# import libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False

# load dataset
df = pd.read_csv('C:/Users/timon/Documents/BI-2020/Data-Science/Projects/Data-Science-Project/Data/cross-cafe-final.csv')

# change Datetime to datetype datetime
df['Datetime'] = pd.to_datetime(df['Datetime'], infer_datetime_format=True)
df.info()

# set Datetime as index and plot all series
# df.set_index('Datetime')[['cross_cafe', 'cloud_cover', 'humidity', 'precip_dur_past1h', 'precip_past1h', 'pressure', 'temp_dew',
#                           'temp_dry', 'temp_max_past1h', 'temp_mean_past1h', 'temp_min_past1h', 'wind_dir', 'wind_max_per10min_past1h',
#                           'wind_speed', 'wind_speed_past1h', 'sun_last1h_glob']].plot(subplots=True)
# plt.close()

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
# df_input.describe() # looks alright 

# replace NA values in dataframe 
# cloud_cover
# df_input.isna().sum()
# propagate last valid observation forward to next 
df_input.fillna(method='ffill', inplace = True)
# df_input.isna().sum()
# we will replace na's in cloud cover and wind_min with 0
df_input['cloud_cover'].fillna(0, inplace = True)
df_input.isna().sum()
# no more missing values

# scale input variables
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_input)
data_scaled

# create feature and target array
features=data_scaled
target=data_scaled[:,0]

# split dataset
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=123, shuffle = False)


# specify time series generator
lag_length=240
batch_size=40
num_features=16
train_generator = TimeseriesGenerator(x_train, y_train, length=lag_length, sampling_rate=120, batch_size=batch_size)
test_generator = TimeseriesGenerator(x_test, y_test, length=lag_length, sampling_rate=120, batch_size=batch_size)

# inspect train and test data
x_train.shape
x_test.shape 

# train_generator[0]

# build model 
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, activation = 'relu', input_shape= (lag_length, num_features), return_sequences=True))
model.add(tf.keras.layers.Dropout(0.3)) 
model.add(tf.keras.layers.LSTM(32, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.3)) 
model.add(tf.keras.layers.Dense(1))

model.summary() 

# enable early stopping so that we are not going to have to wait too long if we no longer yield further improvement from additional epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,
                                                    mode='min')

# compile model
model.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanAbsoluteError()])

history = model.fit_generator(train_generator, epochs=20,
                    validation_data=test_generator,
                    shuffle=False,
                    callbacks=[early_stopping])

tf.keras.models.save_model(model, 'C:/Users/timon/Documents/BI-2020/Data-Science/Projects/Data-Science-Project/Analysis')

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# evaluate model performance
model.evaluate_generator(test_generator, verbose=0)

# predict test observations
predictions = model.predict(test_generator)
predictions.shape[0]
predictions

# subset test dataset and exclude first 720 observations as we have used those as our lag_length
df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:,1:][lag_length:])], axis=1)
df_pred

# inverse transform scaled values back to initial values
rev_trans = scaler.inverse_transform(df_pred)
rev_trans

# create final dataframe 
df_final=df_input[predictions.shape[0]*-1:]
df_final
df_final.count()
df_final['cross_pred']=rev_trans[:,0]

# plot the forecast and actual values
df_final[['cross_cafe','cross_pred']].plot()
forecast_diff = df_final['cross_cafe'] - df_final['cross_pred'] 



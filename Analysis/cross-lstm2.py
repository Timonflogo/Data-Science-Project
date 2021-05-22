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

# plot heatmap
# df_input = df_input.reset_index()
# get correlations
df_input_corr = df_input.corr()
# create mask
mask = np.triu(np.ones_like(df_input_corr, dtype=np.bool))

sns.heatmap(df_input_corr, mask=mask, annot=True, fmt=".2f", cmap='Blues',
            vmin=-1, vmax=1, cbar_kws={"shrink": .8})

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

# ---------------------------------------------------------------------------------------------------------------

# we are going to use 90% of the data as train data and the remaining as test data
train_size = int(len(df_input) * 0.9)
test_size =  len(df_input) - train_size
train, test = df_input.iloc[0:train_size], df_input.iloc[train_size:len(df_input)]

# scale train data
f_columns = ['cloud_cover', 'humidity', 'temp_dew','temp_dry', 'temp_max_past1h', 'temp_mean_past1h', 'wind_dir', 'wind_max_per10min_past1h',
             'wind_speed', 'wind_speed_past1h', 'sun_last1h_glob', 'hour', 'day_of_week', 'month', 'is_weekend']

f_transformer = MinMaxScaler()
cross_transformer = MinMaxScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
cross_transformer = cross_transformer.fit(train[['cross_cafe']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['cross_cafe'] = cross_transformer.transform(train[['cross_cafe']])

# scale test data
test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['cross_cafe'] = cross_transformer.transform(test[['cross_cafe']])

# create dataset for cutting time series data in subsequences
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i: (i + time_steps)].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)

# define number of lags
time_steps = 48

# create actual train and test dataset
X_train, y_train = create_dataset(train, train.cross_cafe, time_steps=time_steps)
X_test, y_test = create_dataset(test, test.cross_cafe, time_steps=time_steps)

# # check for shapes
# print(X_train.shape, y_train.shape)
# # (15765, 24, 20) 
# # (samples, time_steps, n_features) 
# print(X_test.shape, y_test.shape)
# X_test[0].shape 

# Build LSTM model 
model = keras.Sequential()
model.add(keras.layers.LSTM(40, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))

# define early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=2,
                                               mode='min')

          
model.compile(loss='mse',
              optimizer = 'adam',
              metrics=['mae'])
model.summary()

# specify parameters
epochs = 100
batch_size= 32

history = model.fit(
    X_train, y_train,
    epochs = epochs,
    batch_size=batch_size,
    validation_split = 0.1,
    shuffle=False
)

# visualise  results
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend();

# get predictions from model
y_pred = model.predict(X_test)

# inverse scaling to get forecasting results
y_train_inv = cross_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv =  cross_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = cross_transformer.inverse_transform(y_pred)

X_train_inv = cross_transformer.inverse_transform(X_train.reshape(1, -1))
X_test_inv =  cross_transformer.inverse_transform(X_test.reshape(1, -1))

# plot results
plt.plot(y_test_inv.flatten(), marker= '.', label='true')
plt.plot(y_pred_inv.flatten(), 'r' , marker = '.', label='predicted')
plt.legend();

# evalute model 
model.evaluate(X_test, y_test)

plt.plot(history.history['mae'], label='train')
plt.plot(history.history['val_mae'],label='validation')
plt.legend();

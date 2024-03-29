import os
import datetime

import IPython
import IPython.display
from pylab import rcParams
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
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
df.isna().sum()
# df.set_index('Datetime')['cross_cafe'].plot(subplots=False)


# create input dataframe
df_input = df[['Datetime', 'cross_cafe', 'cloud_cover', 'humidity', 'precip_dur_past1h', 'precip_past1h', 'temp_dew',
               'temp_dry', 'temp_max_past1h', 'temp_mean_past1h', 'temp_min_past1h', 'wind_dir', 'wind_max_per10min_past1h',
               'wind_speed', 'wind_speed_past1h', 'sun_last1h_glob', 'pressure']]
date_time = df_input.pop('Datetime')

# outlier detection 
#df_input.describe()
# boxplot 
#sns.boxplot(data=df_input)


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


# =========================== Exploratory Data Analysis ==============================
# plot heatmap
# df_input = df_input.reset_index()
# get correlations
# =============================================================================
# df_input_corr = df_input.corr()
# # create mask
# mask = np.triu(np.ones_like(df_input_corr, dtype=np.bool))
# 
# sns.heatmap(df_input_corr, mask=mask, annot=True, fmt=".2f", cmap='Blues',
#             vmin=-1, vmax=1, cbar_kws={"shrink": .8})
# =============================================================================

# drop values 
df_input = df_input[['cross_cafe', 'cloud_cover', 'humidity', 'pressure',
                     'temp_mean_past1h', 'wind_max_per10min_past1h', 'sun_last1h_glob', 
                     'hour', 'day_of_month', 'day_of_week', 'month', 'is_weekend']]
df_input.info()


# =============================================================================
# # Regression
# X = df_input[['cross_cafe', 'cloud_cover', 'humidity',
#               'temp_mean_past1h', 'wind_max_per10min_past1h', 'sun_last1h_glob',
#               'hour', 'day_of_week', 'is_weekend']]
# y = df_input['cross_cafe']
# 
# # encode categorical values
# X['hour'] = X['hour'].astype("category")
# X['day_of_week'] = X['day_of_week'].astype("category")
# X['is_weekend'] = X['is_weekend'].astype("category")
# X.info()
# 
# model = sm.OLS(y, X).fit()
# predictions = model.predict(X)
# 
# model.summary()
# =============================================================================


# create final dataframe for LSTM
df_input = df_input[['cross_cafe', 'cloud_cover', 'humidity', 'pressure',
                     'temp_mean_past1h', 'wind_max_per10min_past1h', 'sun_last1h_glob']]
df_input.info()
    
# create daily, weekly, and yearly signals 
timestamp_s = date_time.map(datetime.datetime.timestamp)
day = 24*60*60
week = 24*60*60*7
year = (365.2425)*day

df_input['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df_input['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))


plt.plot(np.array(df_input['Day sin'])[:25])
plt.plot(np.array(df_input['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')


# df_input.to_csv('final_dataset.csv')

# =============================================================================
# ax = sns.violinplot(x='Column', y='Normalized', data=df_input)
# _ = ax.set_xticklabels(df_input.keys(), rotation=90)
# =============================================================================


# dataframe for univariate 
df_input = df_input['cross_cafe']

# ================================== Multi Step single shot forecasting models ========================

# ---------------------------------------------- split data -----------------------------
# split the data
# we will use a 70/20/10 split for training, validation, and test sets
column_indices = {name: i for i, name in enumerate(df_input.columns)}

n = len(df_input)
train_df = df_input[0:int(n*0.7)]
val_df = df_input[int(n*0.7):int(n*0.9)]
test_df = df_input[int(n*0.9):]

num_features = df_input.shape[1]
# ------------------------------------------ normalize data --------------------------

# normalize the data before training
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# look at features after normalization
df_std = (df_input - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df_input.keys(), rotation=90)
# looks fine, but there are still some extreme values in our target variable

# ---------------------------- create window class for indexing and offsetting ---------------------
# create window class for 
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

def __repr__(self):
  return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

# --------------------- create function to handle label columns ----------------

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

# =============================================================================
# # =============================================================================
# # # ------------------- example of how the window generator function works ------------------
# # example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
# #                             np.array(train_df[100:100+w1.total_window_size]),
# #                             np.array(train_df[200:200+w1.total_window_size])])
# # 
# # 
# # example_inputs, example_labels = w1.split_window(example_window)
# # 
# # print('All shapes are: (batch, time, features)')
# # print(f'Window shape: {example_window.shape}')
# # print(f'Inputs shape: {example_inputs.shape}')
# # print(f'labels shape: {example_labels.shape}')
# # 
# # =============================================================================
# w1.example = example_inputs, example_labels
# =============================================================================

# -------------------------- define plot function for evaluation -------------

# plot window
def plot(self, model=None, plot_col='cross_cafe', max_subplots=3, title = 'plot'):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.tight_layout(pad=1.0)
    plt.ylabel(f'{plot_col} [norm]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)
    plt.title(title)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')

WindowGenerator.plot = plot

# --------------- convert input dataframe to tf.data.dataset ---------

# create tf.data.dataset
def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

# ---------------- add properties for accessing tf.data.dataset -----------

# The WindowGenerator object holds training, validation and test data. 
# Add properties for accessing them as tf.data.Datasets using the above make_dataset method. 
# Also add a standard example batch for easy access and plotting:
@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

# =============================================================================
# # Each element is an (inputs, label) pair
# w1.train.element_spec
# 
# for example_inputs, example_labels in w1.train.take(1):
#   print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
#   print(f'Labels shape (batch, time, features): {example_labels.shape}')
# =============================================================================
  
# ----------------------- define forecasting horizon --------------------------  
  
# define forecasting length
lags = 24
OUT_STEPS = 24
multi_window = WindowGenerator(input_width=lags,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               label_columns=['cross_cafe'])
# multi_window.plot()
# multi_window

# ------------------- create compile and fit -------------------------------
MAX_EPOCHS = 100

def compile_and_fit(model, window, patience=10):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping],
                      shuffle=False)
  return history


# =================== create multi-step baseline model ===========================
class MultiStepLastBaseline(tf.keras.Model):
  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}

multi_val_performance['Last_obs'] = last_baseline.evaluate(multi_window.val)
multi_performance['Last_obs'] = last_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot(last_baseline, title= 'Last observation (multivariate) 24 hours')

# ----------------- instantiate and evaluate baseline model ------------------
class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
    return inputs

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance['Repeat_win'] = repeat_baseline.evaluate(multi_window.val)
multi_performance['Repeat_win'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot(repeat_baseline, title = 'Previous window (multivariate) 24 hours')

# ===================== Dense Neural Network =======================

multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(256, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_dense_model, multi_window)

IPython.display.clear_output()
multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_dense_model, title= 'Dense Neural Network (multivariate) 24 hours')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title("Dense Loss 24 hours")
plt.legend();

# ================== Conv NN model =============================
CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(128, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_conv_model, multi_window)

IPython.display.clear_output()

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_conv_model, title = "Convolutional Neural Network 24 hours")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title("Conv loss 24 hours")
plt.legend();

# =================== run LSTm model ===========================
multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, activation = 'relu', return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model, title = 'LSTM (multivariate) 24 hours')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title("LSTM loss 24 hours")
plt.legend();

# -------------------plot evaluation metrics --------------

# =============================================================================
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='validation')
# plt.title("loss")
# plt.legend();
# 
# plt.plot(history.history['mean_absolute_error'], label='train')
# plt.plot(history.history['val_mean_absolute_error'],label='validation')
# plt.title('MAE')
# plt.legend();
# 
# =============================================================================

x = np.arange(len(multi_performance))
width = 0.3


metric_name = 'mean_absolute_error'
metric_index = multi_lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.title('MAE (multivariate) 24 hours')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()

for name, value in multi_performance.items():
  print(f'{name:8s}: {value[1]:0.4f}')

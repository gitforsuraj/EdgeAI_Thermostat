import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

df = pd.read_csv('/Users/suraj/Documents/EdgeAI_Thermostat/Dataset_V2.csv')
df.head()

df.isna().sum()

train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

sns.pairplot(train_dataset[['San Francisco Relative Humidity', 'San Francisco Temperature']], diag_kind='kde')
train_dataset.describe().transpose()

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('San Francisco Temperature')
test_labels = test_features.pop('San Francisco Temperature')

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

humidity = np.array(train_features['San Francisco Relative Humidity'])

humidity_normalizer = preprocessing.Normalization(input_shape=[1,])
humidity_normalizer.adapt(humidity)

# humidity_model = tf.keras.Sequential([humidity_normalizer, layers.Dense(units=1)])
# humidity_model.summary()
humidity_model = tf.keras.Sequential([normalizer, layers.Dense(units=1)])
humidity_model.summary()

humidity_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')

# history = humidity_model.fit(train_features['San Francisco Relative Humidity'], train_labels, epochs=100, verbose=0, validation_split = 0.2)

history = humidity_model.fit(train_features, train_labels, epochs=100, verbose=0, validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [Temperature]')
  plt.legend()
  plt.grid(True)

plot_loss(history)

#test_results = {}
#test_results['humidity_model'] = humidity_model.evaluate(test_features['San Francisco Relative Humidity'], test_labels, verbose=0)

test_results = {}
test_results['humidity_model'] = humidity_model.evaluate(test_features, test_labels, verbose=0)

x = tf.linspace(0.0, 120, 251)
y = humidity_model.predict(x)

def plot_humidity(x, y):
  plt.scatter(train_features['San Francisco Relative Humidity'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('San Francisco Relative Humidity')
  plt.ylabel('Temperature')
  plt.legend()
  
plot_humidity(x, y)

pd.DataFrame(test_results, index=['Mean absolute error [Temperature]']).T

test_predictions = humidity_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Temperature]')
plt.ylabel('Predictions [Temperature]')
lims = [30, 90]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [Temperature]')
_ = plt.ylabel('Count')
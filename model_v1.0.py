#data printing out things
import csv
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision=3, suppress=True)

#initialize tensorflow
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import os
os.environ['IF_XLA_FLAGS']='–tf_xla_enable_xla_devices'

print("--------------------------------------")
print("Tensorflow version: " + tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print()
plotnum = 1
#load data
def load_data():
    print("loading humidity data...")
    raw_dataH = pd.read_csv('humidity.csv', header=0, na_values='',
        sep=',', skipinitialspace=False)
    print("Finished loading humidity data!")
    print()

    print("loading temp data...")
    raw_dataT = pd.read_csv('temperature.csv', header=0, na_values='',
        sep=',', skipinitialspace=False)
    print("Finished loading temp data!")
    print("--------------------------------------")
    return raw_dataH, raw_dataT

def plot_loss(history):
    global plotnum
    plot = pyplot.figure(plotnum)
    plotnum += 1
    print("Loading loss plot...")
    pyplot.plot(history.history['loss'], label='loss')
    pyplot.plot(history.history['val_loss'], label='val_loss')
    pyplot.ylim([0,10])
    pyplot.xlabel('epoch')
    pyplot.ylabel('Error [Temp]')
    pyplot.legend()
    pyplot.grid(True)
    print("Finished loading loss plot!")

def plot_humidity(x, y, train_features, train_labels):
    global plotnum
    plot = pyplot.figure(plotnum)
    plotnum += 1
    print("Loading humidity plot...")
    pyplot.scatter(train_features['VancouverH'], train_labels, label='Data')
    pyplot.plot(x, y, color='k', label='Predictions')
    pyplot.xlabel('Humidity')
    pyplot.ylabel('Temp (Kelvin)')
    pyplot.legend()
    print("Finished loading humidity plot!")

#main
def main():
    raw_dataH, raw_dataT = load_data()
    datasetT = raw_dataT.copy()
    datasetH = raw_dataH.copy()
    
    datetime = datasetT.pop('datetime')
    datasetH.pop('datetime')
    datetime = pd.to_datetime(datetime)

    columnnames = {'VancouverT', 'VancouverH', 'PortlandT', 'PortlandH'}
    #dataset = pd.DataFrame(columns=datetime)
    datasetP = pd.DataFrame(data={'PortlandT': datasetT['Portland'], 
        'PortlandH': datasetH['Portland']})

    datasetV = pd.DataFrame(data={'VancouverT': datasetT['Vancouver'],
        'VancouverH': datasetH['Vancouver']})
    
    datasetP = datasetP.rename(index=lambda s: datetime[s])
    datasetP = datasetP.dropna()
    datasetP = pd.get_dummies(datasetP)
    train_datasetP = datasetP.sample(frac=0.8, random_state=0)
    test_datasetP = datasetP.drop(train_datasetP.index)

    datasetV = datasetV.rename(index=lambda s: datetime[s])
    datasetV = datasetV.dropna()
    datasetV = pd.get_dummies(datasetV)
    train_datasetV = datasetV.sample(frac=0.8, random_state=0)
    test_datasetV = datasetV.drop(train_datasetV.index)

    #sns.pairplot(train_datasetV[['VancouverT', 'VancouverH']], diag_kind='kde')
    #sns.pairplot(train_datasetP[['PortlandT', 'PortlandH']], diag_kind='kde')
    #pyplot.show()

    train_featuresV = train_datasetV.copy()
    test_featuresV = test_datasetV.copy()
    train_labelsV = train_featuresV.pop('VancouverT')
    test_labelsV = test_featuresV.pop('VancouverT')

    print(train_featuresV)
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_featuresV))

    print(normalizer.mean.numpy())

    first = np.array(train_featuresV[:1])
    with np.printoptions(precision=2, suppress=True):
        print('First example: ', first)
        print()
        print('Normalized: ', normalizer(first).numpy())
    

    humidity = np.array(train_featuresV['VancouverH'])
    
    humidity_normalizer = preprocessing.Normalization(input_shape=[1,])
    humidity_normalizer.adapt(humidity)

    humidity_model = tf.keras.Sequential([humidity_normalizer, layers.Dense(units=1)])
    humidity_model.summary()
    humidity_model.predict(humidity[:10])

    humidity_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
    
    history = humidity_model.fit(train_featuresV['VancouverH'], train_labelsV, epochs=100, verbose=0, validation_split=0.2)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    plot_loss(history)

    test_results = {}
    test_results['humidity_model'] = humidity_model.evaluate(test_featuresV['VancouverH'], test_labelsV, verbose=0)

    x=tf.linspace(0.0, 100, 101)
    y=humidity_model.predict(x)

    plot_humidity(x, y, train_featuresV, train_labelsV)
    pyplot.show()

    # linear_model = tf.keras.Sequential([normalizer, layers.Dense(units=1)])
    # linear_model.predict(train_features[:10])
    # linear_model.layers[1].kernel
    # linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
    # history = linear_model.fit(train_features, train_labels, epochs=100, verbose=0, validation_split = 0.2)
    # plot_loss(history)
    # test_results['linear_model'] = linear_model.evaluate(test_features, test_labels, verbose=0)

main()
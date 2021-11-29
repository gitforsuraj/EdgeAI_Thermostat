import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import base64
import math
import uuid
import os.path
import os
from os import walk
from tensorflow.keras import layers


# print versions
print('Numpy ' + np.__version__)
print('Pandas ' + pd.__version__)
print('TensorFlow ' + tf.__version__)
print('Keras ' + tf.keras.__version__)


# initializes features and labels into the data we want and returns 2 dictionaries filled with
# "train" "test" and "validate"
def data_init(features, labels):
    inputs = []
    outputs = []

    # converts pd array to np array
    inputs = np.array(features)
    outputs = np.array(labels)

    # set up shuffle
    num_inputs = len(inputs)
    randomize = np.arange(num_inputs)
    np.random.shuffle(randomize)

    # randomize the data
    inputs = inputs[randomize]
    outputs = outputs[randomize]

    # split data into training and testing
    train_split = int(0.6 * num_inputs)
    test_split = int(0.2 * num_inputs + train_split)

    # setup the output dictionaries each holds 3 things - train, test, validate
    input_dict = {}
    output_dict = {}

    # split data
    input_dict["train"], input_dict["test"], input_dict["validate"] = np.split(inputs, [train_split, test_split])
    output_dict["train"], output_dict["test"], output_dict["validate"] = np.split(outputs, [train_split, test_split])

    return input_dict, output_dict


# input parameters is the dictionaries from data_init
# returns the model and history graph.
def init_model(inputs, outputs, epochs):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])
    history = model.fit(inputs["train"], outputs["train"],
                        epochs=epochs, batch_size=1,
                        validation_data=(inputs["validate"], outputs["validate"]))
    return model, history


# points is the "history" return from init_model
def graph_loss(points):
    loss = points.history['loss']
    val_loss = points.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'g.', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    path = os.path.join('static', 'loss.png')
    plt.savefig(path)


# points is the "history" from the init_model
def graph_MAE(points, SKIP):
    loss = points.history['loss']
    epochs = range(1, len(loss) + 1)
    mae = points.history['mae']
    val_mae = points.history['val_mae']
    plt.plot(epochs[SKIP:], mae[SKIP:], 'g.', label='Training MAE')
    plt.plot(epochs[SKIP:], val_mae[SKIP:], 'b.', label='Validation MAE')
    plt.title('Training and validation mean absolute error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    path = os.path.join('static', 'MAE.png')
    plt.savefig(path)


def graph_predictions(model, inputs_test, outputs_test):
    predictions = model.predict(inputs_test)

    plt.clf()
    plt.title("Comparison of predictions to actual values")
    plt.plot(inputs_test, outputs_test, 'b.', label='Actual')
    plt.plot(inputs_test, predictions, 'r.', label='Prediction')
    plt.legend()
    path = os.path.join('static', 'pred.png')
    plt.savefig(path)


# model = Keras model
# inputs = inputs dictionary from init_data
# points = history from init_model
# type = string to denote what type of graph you want to make

def graph_model(model, inputs, outputs, points, typeG):
    if (typeG == 'loss'):
        graph_loss(points)
    elif (typeG == 'MAE'):
        graph_MAE(points, 100)
    elif (typeG == 'pred'):
        graph_predictions(model, inputs['test'], outputs['test'])


# Convert Keras Model to a tflite model
def TF2Keras(name, model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    open(name + '.tflite', 'wb').write(tflite_model)
    return tflite_model


# csv is the pandas array of the CSV not the actual path
def updateCSV(df1, df2):
    for df2_index in df2.index:
        mask_month = df1['Month'] == df2.at[df2_index, 'Month']
        mask_day = df1['Day'] == df2.at[df2_index, 'Day']
        mask_time = df1['Time of Day'] == df2.at[df2_index, 'Time of Day']

        # range, since humidity won't be exact
        mask_humidl = 0.8 * df2.at[df2_index, 'Relative Humidity'] <= df1['Relative Humidity']
        mask_humidu = df1['Relative Humidity'] <= 1.2 * df2.at[df2_index, 'Relative Humidity']

        rows_affected = df1.index[mask_month & mask_day & mask_time & mask_humidl & mask_humidu].to_list()
        print(len(rows_affected))

        if len(rows_affected) == 0:
            df1.loc[len(df1.index)] = df2.loc[df2_index]
        else:
            for index in rows_affected:
                print(df1.loc[[index]], df2.loc[[df2_index]] ) 
                if not (1.2 * df2.at[df2_index, 'Target Temperature'] >= df1.at[index, 'Target Temperature'] >= 0.8 * df2.at[df2_index, 'Target Temperature']):
                    df1.at[index, 'Target Temperature'] = df2.at[df2_index, 'Target Temperature']

    return df1


###################################################
# This section converts tflite model into C array #
###################################################

# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):
    c_str = ''

    # Create header guard
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'

    # Add array length at top of file
    c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

    # Declare C variable
    c_str += 'unsigned char ' + var_name + '[] = {'
    hex_array = []
    for i, val in enumerate(hex_data):

        # Construct string from hex
        hex_str = format(val, '#04x')

        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n '
        hex_array.append(hex_str)

    # Add closing brace
    c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

    # Close out header guard
    c_str += '#endif //' + var_name.upper() + '_H'

    return c_str

def C2H(name, Carray, tflite_model):
    with open(name + '.h', 'w') as file:
        file.write(hex_to_c_array(tflite_model, name))


def create_model(file_path, debug_mode = False):
    if(os.path.exists('model1.h5')):
        return update_model(file_path, debug_mode)
    else:
        # set up features and labels
        df = pd.read_csv(file_path)

        # modify names when we make the export from nano
        features = df.drop('Target Temperature', 1)
        labels = df['Target Temperature']
        print(features)
        print(labels)
        inputs, outputs = data_init(features, labels)

        model, history = init_model(inputs, outputs, 20)

        if debug_mode:
            graph_model(model, inputs, outputs, history, 'loss')
            graph_model(model, inputs, outputs, history, 'MAE')

        tflite_model = TF2Keras('thermo_model', model)

        c_str = hex_to_c_array(tflite_model, 'thermo_model')
        C2H('thermo_model', c_str, tflite_model)
        return model, inputs, outputs, history


def update_model(file_path, debug_mode = False):
    # get current working directory 
    current_directory = os.getcwd()

    # get number of dataset files in directory
    path, dirs, files = next(os.walk(current_directory + '/Datasets'))
    file_count = len(files)

    if(file_count > 1):
        # set up features and labels
        df = pd.read_csv(file_path)

        # modify names when we make the export from nano
        features = df.drop('Target Temperature', 1)
        labels = df['Target Temperature']
        inputs, outputs = data_init(features, labels)

        model = tf.keras.models.load_model('model1.h5')
        history = model.fit(inputs["train"], outputs["train"],
                            epochs=20, batch_size=1,
                            validation_data=(inputs["validate"], outputs["validate"]))

        if debug_mode:
            graph_model(model, inputs, outputs, history, 'loss')
            graph_model(model, inputs, outputs, history, 'MAE')

        tflite_model = TF2Keras('thermo_model', model)

        c_str = hex_to_c_array(tflite_model, 'thermo_model')
        C2H('thermo_model', c_str, tflite_model)
        return model, inputs, outputs, history
    else: 
        sys.exit()


if __name__ == '__main__':
    # get current working directory 
    current_directory = os.getcwd()

    # counts number of files in the dirctory
    path, dirs, files = next(os.walk(current_directory + '/Datasets'))
    file_count = len(files)

    # checks if there is a csv file ready to combine
    if (file_count == 1):
  
        find_file_name = [os.path.join(current_directory + '/Datasets', x) 
                for x in os.listdir(current_directory + '/Datasets') 
                if x.endswith(".csv")]

        filename = max(find_file_name , key = os.path.getctime)

        # creates model with base csv file        
        model, inputs, outputs, history = create_model(filename)
        model.save("model1.h5")
        model.summary()

        # build arduino file
        os.system("arduino-cli compile -p /dev/cu.usbmodem14301 --fqbn arduino:mbed_nano:nano33ble /Users/suraj/Documents/Arduino/thermo/edgeaithermostat.ino")
        os.system("arduino-cli upload -p /dev/cu.usbmodem14301 --fqbn arduino:mbed_nano:nano33ble /Users/suraj/Documents/Arduino/thermo/edgeaithermostat.ino")

    else: 
        # find newest and oldest files in directory 
        old_new_files = [os.path.join(current_directory + '/Datasets', x) 
                for x in os.listdir(current_directory + '/Datasets') 
                if x.endswith(".csv")]

        # store datapath of oldest and newest csv files based on time of creation
        newest_path = max(old_new_files , key = os.path.getctime)
        oldest_path = min(old_new_files , key = os.path.getctime)

        # new empty csv file is named after a randomly generated string 
        filename = str(uuid.uuid4())
        generated_file = open(current_directory + '/Datasets/' + filename + '.csv',"w+")
        generated_file_path = current_directory + '/Datasets/' + filename + '.csv'

        # read csv files as data frames
        df1 = pd.read_csv(oldest_path)
        df2 = pd.read_csv(newest_path)

        # modify datapoints and store in new csv file
        new_df = updateCSV(df1, df2)
        new_df.to_csv(generated_file_path, index=False)

        # remove old csv files to avoid file name collision
        os.remove(oldest_path)
        os.remove(newest_path)
        
        # creates model with modified csv file        
        model, inputs, outputs, history = create_model(generated_file_path)
        model.save("model1.h5")
        model.summary()

        # build arduino file
        os.system("arduino-cli compile -p /dev/cu.usbmodem14301 --fqbn arduino:mbed_nano:nano33ble /Users/suraj/Documents/Arduino/thermo/edgeaithermostat.ino")
        os.system("arduino-cli upload -p /dev/cu.usbmodem14301 --fqbn arduino:mbed_nano:nano33ble /Users/suraj/Documents/Arduino/thermo/edgeaithermostat.ino")


        # COMMAND USED TO RUN SCRIPT ON A TIMER
        # while true; do python3 model_v1.2.py; sleep <# of seconds>; done

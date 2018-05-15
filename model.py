from setup_data import get_data

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers import Cropping2D, Lambda
from keras.models import load_model

from sklearn.utils import shuffle

### Directories and files
data_dir = './data'
datasets_file = data_dir + '/datasets.h5'
model_file = './model.h5'

### Define the model used to train the dataset
def model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3)))
    model.add(Conv2D(6, (5, 5), strides=2, activation='relu'))
    model.add(Conv2D(12, (5, 5), strides=2, activation='relu'))
    model.add(Conv2D(24, (5, 5), strides=2, activation='relu'))
    model.add(Conv2D(48, (3, 3)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model

### Experimental model based on NVIDIA's model (with Dropout regularization)
def model1():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3)))
    model.add(Conv2D(24, (5, 5), strides=2, activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=2, activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model

### LeNet model
def LeNet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(((70, 25), (0, 0))))
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model

### NVIDIA autonomous vehicle training model
def nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3)))
    model.add(Conv2D(24, (5, 5), strides=2, activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=2, activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model

### Main program
if __name__ == '__main__':
    ### Setup dataset if it's not saved previously
    if not os.path.exists(datasets_file):
        print('Setting up data')
        X, Y = get_data() # Use center camera
        # X, Y = get_data(True) # Use all 3 cameras
    ### Load dataset if it's saved previously
    else:
        print('Loading data from', datasets_file)
        h5_data = h5py.File(datasets_file, 'r')
        datasets = {}
        for dataset_name in h5_data:
            datasets.update({dataset_name: np.array(h5_data[dataset_name])})
        X = datasets['X']
        Y = datasets['Y']

    X, Y = shuffle(X, Y, random_state=42)

    ### Try various models
    # model = nvidia()
    # model = LeNet()

    if not os.path.exists(model_file):
        model = model()
    else:
        ### Use transfer learning if a pre-trained model already exists
        print('Loading model from', model_file)
        model = load_model(model_file)

    ### Save model after every epoch, if it performs better than before
    checkpoint = ModelCheckpoint(model_file, save_best_only=True, period=1)
    training_history = model.fit(X, Y, validation_split = 0.2, epochs = 5, callbacks=[checkpoint])

    ### print the keys contained in the history object
    print(training_history.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('Model Performance')
    plt.ylabel('Mean squared error loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper right')
    plt.show()

import os
import io
import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from os import path
from tensorflow import keras
from tensorflow.keras import layers

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def DataCreator(directory, steps, data_train_len, stride=1):
    data_train = []
    # first file is the merged one - 485,650 characters - pick 550 steps -> so 883 is the batch size
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with io.open(directory + '/' + filename, "r", encoding="latin1") as my_file:
                text = my_file.read()
            data_train.append(SamplesCreator(text, steps, stride))
        if data_train_len <= len(data_train):
            break
    return data_train


def SamplesCreator(txt, steps, stride=1):
    """@:returns x_train and y_train of a specific song (which is a string)"""
    # creating list of ASCIIs
    ASCII = [ord(char) for char in txt]
    # changing it to unit vector
    one_hot = tf.one_hot(ASCII, 256, dtype=tf.uint8)
    sample = one_hot.numpy()
    # splitting it into steps
    stripes = [sample[i:i + steps] for i in range(0, len(sample) - steps + 1, stride)]
    # x and y from the list will be just extra char different in every side of the list
    x_train = stripes[:-1]
    y_train = stripes[1:]
    # TODO: Check what happens for steps that are larger than the file
    return np.array(x_train), np.array(y_train)


def Model():
    # --------------------MODEL - COMPILE & FIT--------------------
    # Creating RNN model and fit it:
    model_RNN = keras.Sequential()
    # Add an LSTM layer with 256 internal units.
    model_RNN.add(layers.LSTM(256, input_shape=(None, 256), return_sequences=True,
                              stateful=False))  # time steps = 'dynamic', dim = 256
    # Add a Dense layer with 256 units - output is unit vector
    model_RNN.add(layers.Dense(256, activation='softmax'))
    model_RNN.summary()

    model_RNN.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=['accuracy'],
    )
    return model_RNN


def ModelFit(model, data, batch_size, epochs=50, amount=None):
    if amount is None:
        amount = len(data)
    for i in range(amount):
        model.fit(data[i][0], data[i][1], batch_size=batch_size, epochs=epochs,
                  verbose=1)  # , callbacks=[tensorboard_callback])


def ModelPredict(model, data, steps, rounds):
    x_test, _ = SamplesCreator(data, 1)
    x_test = x_test.reshape(1, -1, 256)
    prediction = np.argmax(model.predict(x_test), axis=-1)
    one_hot = tf.one_hot(prediction[0][-1], 256, dtype=tf.uint8)
    newvec = one_hot.numpy().reshape(1, -1, 256)
    out = np.append(x_test, newvec).reshape(1, -1, 256)
    for i in range(rounds):
        prediction = np.argmax(model.predict(out), axis=-1)
        one_hot = tf.one_hot(prediction[0][-1], 256, dtype=tf.uint8)
        newvec = one_hot.numpy().reshape(1, -1, 256)
        out = np.append(out, newvec).reshape(1, -1, 256)
    out = np.argmax(out.reshape(-1, 256), axis=1)
    return out


if __name__ == "__main__":
    steps_train = 200  # steps to split the train value
    data_train_len = 1  # num of songs to get from the list, and train on them
    batch_size = 500  # @TODO: For stateful.. ask Cameron
    epochs = 10
    extra_chars = 500  # test prediction of 'extra_chars' characters
    directory = "songs"         # use this directory for eminem songs!
    # directory = "books"       # use this directory for books!

    # Creating the dataset to train on:
    data_train = DataCreator(directory, steps_train, data_train_len, stride=1)

    # Creating the model
    model_RNN = Model()

    # Fitting the model on #(data_train_len) songs
    ModelFit(model_RNN, data_train, batch_size, epochs, amount=data_train_len)  # amount = number of songs

    # Testing the model
    data_test = """ What should I do when I see it first thing in the morning"""

    out = ModelPredict(model_RNN, data_test, steps=steps_train, # len(data_test) - 1,
                       rounds=extra_chars)  # rounds of prediction

    NEWSONG = ''.join(chr(i) for i in out)
    print("THIS IS MY NEW SONG:\n\n" + NEWSONG)
    f = open('results/' + 'mysong_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt', 'w')
    f.write(NEWSONG)
    f.close()

    a = 0

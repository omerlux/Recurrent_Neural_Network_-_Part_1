import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def plot(x, y, title, legends=None):  # Legends should be list. Title is string
    plt.title("HMM - " + title)
    for j in range(len(y)):
        if j == 1:
            plt.plot(x, y[j], linestyle=':', marker='.')
        else:
            plt.plot(x, y[j], linestyle='--', marker='o')
    if legends is not None:
        plt.legend(legends)
    plt.grid(True)
    # plt.savefig("imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_HMM-" + title)
    plt.show()


class Sequence:
    """ @Class Sequence is a class of the process of 0s and 1s
        parity is the parity condition sequence w.r.t this process"""

    def __init__(self, m):
        """process - random sequence of 0s and 1s."""
        self.process = np.random.choice(2, m)
        """parity - the parity of the 'process'."""
        self.parity = np.array([1 if np.mod(cs, 2) else 0
                                for cs in np.cumsum(self.process)])

    def getXY(self):
        """@:returns trainable parameters.
        @:param _x is sequence of the last 'steps' element of the process.
        @:param _y is the current state of the parity, until the last element from the process. (parity index i is for the
        sequence of [:i] include i."""
        # _y = self.parity[steps - 1:]
        # _x = np.array([self.process[i - steps + 1:i + 1] for i in range(len(self.parity))][steps - 1:])
        # return [_x, _y]
        return [self.process, self.parity]


if __name__ == "__main__":
    N = 10 ** 3  # Samples train
    M = 10 ** 2  # Samples test
    steps = 100  # Sequence

    data_train = [Sequence(m=steps) for i in range(N)]
    data_test = [Sequence(m=steps) for i in range(M)]
    x_train, y_train = np.array([z.getXY()[0] for z in data_train]).reshape(-1, steps, 1), \
                       np.array([z.getXY()[1] for z in data_train])
    x_test, y_test = np.array([z.getXY()[0] for z in data_test]).reshape(-1, steps, 1), \
                       np.array([z.getXY()[1] for z in data_test])

    # --------------------MODEL - COMPILE & FIT--------------------
    # Creating RNN model and fit it:
    model_RNN = keras.Sequential()
    # Add a LSTM layer with 64 internal units.
    model_RNN.add(layers.LSTM(64, input_shape=(steps, 1), return_sequences=True))  # time steps = 'steps', dim = 1
    # Add a Dense layer with 1 units - output is only 1 or 0
    model_RNN.add(layers.Dense(1))
    model_RNN.summary()
    # keras.utils.plot_model(model_RNN, "imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_Markov_Model_RNN.png', show_shapes=True) # Model scheme

    model_RNN.compile(
        loss="binary_crossentropy",
        optimizer="RMSprop",
        metrics=['accuracy'],
    )
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_Parity"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model_RNN.fit(
        x_train, y_train, batch_size=32, epochs=100, verbose=1, callbacks=[tensorboard_callback]
    )

    # Prediction:
    # We will check out all the different 5 consecutive numbers from the process

    output_model = model_RNN.predict(x_test)
    results = model_RNN.evaluate(x_test, y_test, batch_size=32)
    print("Test loss:\t\t%f \n"
          "Test accuracy:\t%.2f%%" % (results[0], results[1] * 100))

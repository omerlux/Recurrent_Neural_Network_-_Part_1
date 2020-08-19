import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
        self.parity = np.sum(self.process)

    def getXY(self):
        _x = np.array(self.process)
        _y = [0, 1] if np.mod(self.parity, 2) else [1, 0]  # [0, 1] = 1,  [1, 0] = 0
        return [_x, _y]


if __name__ == "__main__":
    N = 10 ** 4  # Samples train
    M = 10 ** 3  # Samples test
    steps = 100  # Sequence

    data, test= [Sequence(m=steps).getXY() for i in range(N)],  [Sequence(m=steps).getXY() for i in range(M)]

    x_train, y_train = np.array([x for x, y in data]).reshape(-1, 100, 1), \
                       np.array([y for x, y in data]).reshape(-1, 1)

    x_test, y_test = np.array([x for x, y in data]).reshape(-1, 100, 1), \
                     np.array([y for x, y in data]).reshape(-1, 1)

    # --------------------MODEL - COMPILE & FIT--------------------
    # Creating RNN model and fit it:
    model_RNN = keras.Sequential()
    # Add a LSTM layer with 32 internal units.
    model_RNN.add(layers.LSTM(32, input_shape=(steps, 1), return_sequences=False))  # time steps = 3, dim = 1
    # Add a Dense layer with 2 units - output is only 1 or 0 (as A or B/C)
    model_RNN.add(layers.Dense(2, activation='softmax'))
    model_RNN.summary()
    # keras.utils.plot_model(model_RNN, "imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_Markov_Model_RNN.png', show_shapes=True) # Model scheme

    model_RNN.compile(
        loss="categorical_crossentropy",
        optimizer="RMSprop",
        metrics=['accuracy'],
    )
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_Parity"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model_RNN.fit(
        x_train, y_train, batch_size=32, epochs=10, verbose=1, callbacks=[tensorboard_callback]
    )

    # Prediction:
    # We will check out all the different 5 consecutive numbers from the process

    output_model = model_RNN.predict(x_test)
    results = model_RNN.evaluate(x_test, y_test, batch_size=32)
    print("Test loss:\t\t%f \n"
          "Test accuracy:\t%.2f%%" % (results[0], results[1] * 100))

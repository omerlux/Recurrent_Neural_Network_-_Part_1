import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def plot(x, y, title, legends=None):  # Legends should be list. Title is string
    plt.title("Markov Model - " + title)
    for j in range(len(y)):
        if j == 1:
            plt.plot(x, y[j], linestyle=':', marker='.')
        else:
            plt.plot(x, y[j], linestyle='--', marker='o')
    if legends is not None:
        plt.legend(legends)
    plt.grid(True)
    # plt.savefig("imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_Markov_model-" + title)
    plt.show()


def markov_next(prev):
    """@:returns the next markov state in the markov process given prev."""
    prob = np.random.uniform(0, 1)
    if prev == 'A' and prob <= 0.2:
        return 'B'
    if prev == 'A':
        return 'A'
    if prev == 'B':
        return 'C'
    if prev == 'C' and prob <= 0.2:
        return 'C'
    if prev == 'C':
        return 'A'


def markov_converter(markov_process):
    """@:returns a markov process of numbers - not letters"""
    return [[1, 0, 0] if x == 'A' else ([0, 1, 0] if x == 'B' else [0, 0, 1]) for x in markov_process]


def markov_inverter(markov_process):
    """@:returns a markov process of numbers - not letters"""
    # TODO: Maybe change to 0, 1, 2
    return ['A' if x == 0 else ('B' if x == 1 else 'C') for x in markov_process]


def arg_vec(vector, axis):
    """@:returns 0 1 or 2 by the argmax of the unit vectors"""
    return np.argmax(vector, axis=axis)


# Defining the process:
N = 10 ** 5
samples = [random.choice(['A', 'B', 'C'])]
for i in range(N):
    samples.append(markov_next(samples[i]))

# Section (a):
# By calculating analytically:
#     'A' = 5/7.25 = ~0.689
#     'B' = 1/7.25 = ~0.137
#     'C' = 1.25/7.25 = ~172
# Which is as the @stationary_distribution in Section (b).
print("The stationary distribution is:\n" +
        "\tA = " + '%.5f' % (5/7.25) + "\n"
        "\tB = " + '%.5f' % (1/7.25) + "\n"
        "\tC = " + '%.5f' % (1.25/7.25) + "\n")

# Section(b) - Generating the stationary distribution:
unique, counts = np.unique(samples, return_counts=True)
counts = counts / (N + 1)
stationary_distribution = dict(zip(unique, counts))
print("The approximately stationary distribution is:\n" +
        "\tA = " + '%.5f' % stationary_distribution['A'] + "\n"
        "\tB = " + '%.5f' % stationary_distribution['B'] + "\n"
        "\tC = " + '%.5f' % stationary_distribution['B'] + "\n")

# Section(c) -
# Generating train and test:
tests = [random.choice(['A', 'B', 'C'])]
for i in range(N):
    tests.append(markov_next(samples[i]))
y_train = markov_converter(samples)
x_train = []
y_test = markov_converter(tests)
x_test = []
for i in range(1, N + 1):
    x_train.append(np.argmax([y_train[i - 1]], axis=1))
    x_test.append(np.argmax([y_test[i - 1]], axis=1))
x_train = np.array(x_train).reshape(-1, 1, 1)
y_train = np.array(y_train[1:N + 1]).reshape(-1, 1, 3)
x_test = np.array(x_test).reshape(-1, 1, 1)
y_test = np.array(x_test[1:N + 1]).reshape(-1, 1, 3)

# 50 samples of the process:
x_axis = np.transpose(np.array(range(50)))
plot(x_axis, np.argmax(y_train[:50], axis=2).reshape(1, -1), "Markov process of 50 samples")

# --------------------MODEL - COMPILE & FIT--------------------

# Creating RNN model and fit it:
model_RNN = keras.Sequential()
# Add a LSTM layer with 64 internal units.
model_RNN.add(layers.LSTM(32, input_shape=(1, 1), return_sequences=True))  # time steps = 3, dim = 1
# Add a LSTM layer with 64 internal units.
model_RNN.add(layers.LSTM(16, return_sequences=False))
# Add a Dense layer with 1 units.
model_RNN.add(layers.Dense(3))
model_RNN.summary()
# keras.utils.plot_model(model_RNN, "imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_Markov_Model_RNN.png', show_shapes=True) # Model scheme

model_RNN.compile(
    loss='binary_crossentropy',
    optimizer="RMSprop",
    metrics=['accuracy'],
)
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_Markov_Model_RNN"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_RNN.fit(
    x_train, y_train, batch_size=32, epochs=10, verbose=2  # , callbacks=[tensorboard_callback]
)

# Prediction:
out_vec = model_RNN.predict(x_test)
out_as_dist = sum(out_vec) / len(out_vec)
model_distribution = dict(zip(['A', 'B', 'C'], out_as_dist))
print("The model distribution is:\n" +
        "\tA = " + '%.5f' % out_as_dist[0] + "\n"
        "\tB = " + '%.5f' % out_as_dist[1] + "\n"
        "\tC = " + '%.5f' % out_as_dist[2] + "\n")
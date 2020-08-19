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
        plt.legend(legends, loc='upper center', bbox_to_anchor=(0.5, -0.1))
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
      "\tA = " + '%.5f' % (5 / 7.25) + "\n"
                                       "\tB = " + '%.5f' % (1 / 7.25) + "\n"
                                                                        "\tC = " + '%.5f' % (1.25 / 7.25) + "\n")

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
    tests.append(markov_next(tests[i]))
y_train = markov_converter(samples)
x_train = []
y_test = markov_converter(tests)
x_test = []
for i in range(3, N + 1):
    x_train.append(np.argmax([y_train[i - 1], y_train[i - 2], y_train[i - 3]], axis=1))
    x_test.append(np.argmax([y_test[i - 1], y_test[i - 2], y_test[i - 3]], axis=1))
x_train = (np.array(x_train) / 2.0).reshape(-1, 3, 1)
y_train = np.array(y_train[3:N + 1]).reshape(-1, 3, )
x_test = (np.array(x_test) / 2.0).reshape(-1, 3, 1)
y_test = np.array(y_test[3:N + 1]).reshape(-1, 3, )

# 50 samples of the process:
x_axis = np.transpose(np.array(range(50)))
plot(x_axis, np.argmax(y_train[:50], axis=1).reshape(1, -1), "Markov process of 50 samples", ["A=0, B=0.5, C=1"])

# --------------------MODEL - COMPILE & FIT--------------------

# Creating RNN model and fit it:
model_RNN = keras.Sequential()
# Add a LSTM layer with 32 internal units.
model_RNN.add(layers.LSTM(32, input_shape=(3, 1), return_sequences=False, stateful=True))  # time steps = 3, dim = 1
""" NOTE: We can change the input time step to 1, because of markovity, and get the same result."""
# Add a Dense layer with 3 units - output is 0 1 or 2 (as A B or C)
model_RNN.add(layers.Dense(3, activation='softmax'))
model_RNN.summary()
# keras.utils.plot_model(model_RNN, "imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_Markov_Model_RNN.png', show_shapes=True) # Model scheme

model_RNN.compile(
    loss='categorical_crossentropy',
    optimizer="RMSprop",
    metrics=['accuracy'],
)
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_Markov_Model_RNN"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_RNN.fit(
    x_train, y_train, batch_size=32, epochs=3, verbose=2  # , callbacks=[tensorboard_callback]
)

# Prediction:
out_vec = model_RNN.predict(x_test)
PA = out_vec[np.squeeze([x[0] == 0 for x in x_test])]
PB = out_vec[np.squeeze([x[0] == 0.5 for x in x_test])]
PC = out_vec[np.squeeze([x[0] == 1 for x in x_test])]
PA = np.transpose(sum(PA) / len(PA))
PB = np.transpose(sum(PB) / len(PB))
PC = np.transpose(sum(PC) / len(PC))
print("The model transition matrix is:")
np.set_printoptions(suppress=True, precision=5)
print("P(x|A) -> ", end=""), print(PA)
print("P(x|B) -> ", end=""), print(PB)
print("P(x|C) -> ", end=""), print(PC)

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
    return [[0, 1] if x == 'A' else ([1, 0] if x == 'B' else [1, 0]) for x in markov_process]


# Defining the process:
N = 10 ** 5 + 2
samples = [random.choice(['A', 'B', 'C'])]
for i in range(N):
    samples.append(markov_next(samples[i]))

# Section (a):
# This is a Hidden Markov model - we only observe '0' or '1' but obliviously there are 3 different states
print("The stationary distribution is:\n" +
       "\tA = " + '%.5f' % (5 / 7.25) + "\n"
       "\tB = " + '%.5f' % (1 / 7.25) + "\n"
       "\tC = " + '%.5f' % (1.25 / 7.25) + "\n")

# Section(b) -
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
x_train = (np.array(x_train)).reshape(-1, 3, 1)
y_train = np.array(y_train[3:N + 1]).reshape(-1, 2, )   # y will be only 0 or 1...
x_test = (np.array(x_test)).reshape(-1, 3, 1)
y_test = np.array(y_test[3:N + 1]).reshape(-1, 2, )     # y will be only 0 or 1...

# 50 samples of the process:
x_axis = np.transpose(np.array(range(50)))
plot(x_axis, np.argmax(y_train[:50], axis=1).reshape(1, -1), "Markov process of 50 samples")

# --------------------MODEL - COMPILE & FIT--------------------

# Creating RNN model and fit it:
model_RNN = keras.Sequential()
# Add a LSTM layer with 32 internal units.
model_RNN.add(layers.LSTM(32, input_shape=(3, 1), batch_size=32, return_sequences=False, stateful=True))  # time steps = 3, dim = 1
# Add a Dense layer with 2 units - output is only 1 or 0 (as A or B/C)
model_RNN.add(layers.Dense(2, activation='softmax'))
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
# We will check out all the different 2 consecutive numbers' transition probability of the process,
# if we'll get the same transitions as the HMM transition matrix, the network predicted the hidden model perfectly
out_vec = model_RNN.predict(x_test)
# P (t-1), (t-2)
P11 = out_vec[np.squeeze([(x[0] == 1 and x[1] == 1) for x in x_test])]
P10 = out_vec[np.squeeze([(x[0] == 1 and x[1] == 0) for x in x_test])]
P01 = out_vec[np.squeeze([(x[0] == 0 and x[1] == 1) for x in x_test])]
P00 = out_vec[np.squeeze([(x[0] == 0 and x[1] == 0) for x in x_test])]
P11 = np.transpose(sum(P11) / len(P11))
P10 = np.transpose(sum(P10) / len(P10))
P01 = np.transpose(sum(P01) / len(P01))
P00 = np.transpose(sum(P00) / len(P00))


print("The model transition matrix is:")
print("Trnasition | out=0 | out =1 |")
print("-----------+-------+--------+")
np.set_printoptions(suppress=True, precision=5)
print("P(x|1←1):  ", end=""), print(P11),
print("P(x|1←0):  ", end=""), print(P10)
print("P(x|0←1):  ", end=""), print(P01)
print("P(x|0←0):  ", end=""), print(P00)
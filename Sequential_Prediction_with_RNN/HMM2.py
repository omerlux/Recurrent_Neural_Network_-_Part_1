import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


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
    if prev == 'C' and prob <= 0.8:
        return 'C'
    if prev == 'C':
        return 'A'


def hmm(markov, p):
    prob = np.random.uniform(0, 1)
    if markov == 'A' and prob <= p:
        return 1
    if markov == 'A':
        return 0
    if markov == 'B' and prob <= p:
        return 0
    if markov == 'B':
        return 1
    if markov == 'C' and prob <= p:
        return 0
    if markov == 'C':
        return 1


def markov_converter(markov_process):
    """@:returns a markov process of numbers - not letters"""
    # Note: may change to sparse categorical cross-entropy
    return [[1, 0, 0] if x == 'A' else ([0, 1, 0] if x == 'B' else [0, 0, 1]) for x in markov_process]


# Defining the process:
save = True
epochs = 30
past = 30
probability = 0.5
N = 10 ** 5 + past
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

samples_hmm_train = []
samples_hmm_test = []
# creating the samples from the Bernoulli probabilities
for i in range(N):
    samples_hmm_train.append(hmm(samples[i], p=probability))
    samples_hmm_test.append(hmm(tests[i], p=probability))

# new y's for return seq = True at the output
y_train_new = []
y_test_new = []
for i in range(past, N, past // 2):
    x_train.append(samples_hmm_train[i - past:i])
    x_test.append(samples_hmm_test[i - past:i])
    y_train_new.append(y_train[i-past:i])
    y_test_new.append(y_test[i - past:i])

x_train = (np.array(x_train)).reshape(-1, past, 1)
y_train = np.array(y_train_new).reshape(-1, past, 3)  # y will be only 0 or 1...
x_test = (np.array(x_test)).reshape(-1, past, 1)
y_test = np.array(y_test_new).reshape(-1, past, 3)  # y will be only 0 or 1...

# # 50 samples of the process:
# x_axis = np.transpose(np.array(range(50)))
# plot(x_axis, np.argmax(y_train[:50], axis=1).reshape(1, -1), "Markov process of 50 samples")

# --------------------MODEL - COMPILE & FIT--------------------

# Creating RNN model and fit it:
model_RNN = keras.Sequential()
# Add a LSTM layer with 32 internal units.
model_RNN.add(layers.LSTM(32, input_shape=(past, 1), return_sequences=True))  # time steps = 3, dim = 1
# NOTE: We can change the input time step to 1, because of markovity, and get the same result
model_RNN.add(layers.LSTM(16, return_sequences=True))
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
history = model_RNN.fit(
    x_train, y_train, batch_size=32, epochs=epochs, verbose=2, validation_data=(x_test, y_test)  # , callbacks=[tensorboard_callback]
)

# Graphs
x = range(epochs)
train_loss = history.history['loss']
valid_loss = history.history['val_loss']
plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x, train_loss, linewidth=1, label='LSTM training')
plt.plot(x, valid_loss, linewidth=1, label='LSTM testing')
plt.grid(True, which='both', axis='both')
plt.title('HMM - Training CE of LSTM - probability {}'.format(probability))
plt.xlabel('Epochs')
plt.ylabel('CE')
plt.legend()
if save:
    plt.savefig("./imgs/HMM - Training CE p={}.png".format(probability), dpi=800)
plt.show()

# Prediction:
# We will check out all the different 2 consecutive numbers' transition probability of the process,
# if we'll get the same transitions as the HMM transition matrix, the network predicted the hidden model perfectly
out_vec = model_RNN.predict(x_test)
P1 = out_vec[np.squeeze([(x[0] == 1) for x in x_test])]
P0 = out_vec[np.squeeze([(x[0] == 0) for x in x_test])]
P1 = np.transpose(sum(P1) / len(P1))
P0 = np.transpose(sum(P0) / len(P0))

print("The model as viewed transition matrix is:")
print("Trnasition | out=0 | out =1 |")
print("-----------+-------+--------+")
np.set_printoptions(suppress=True, precision=5)
print("P(x|1):    ", end=""), print(P1),
print("P(x|0):    ", end=""), print(P0)

print("==============================")

# P (t-1), (t-2)
P11 = out_vec[np.squeeze([(x[0] == 1 and x[1] == 1) for x in x_test])]
P10 = out_vec[np.squeeze([(x[0] == 1 and x[1] == 0) for x in x_test])]
P01 = out_vec[np.squeeze([(x[0] == 0 and x[1] == 1) for x in x_test])]
P00 = out_vec[np.squeeze([(x[0] == 0 and x[1] == 0) for x in x_test])]
P11 = np.transpose(sum(P11) / len(P11))
P10 = np.transpose(sum(P10) / len(P10))
P01 = np.transpose(sum(P01) / len(P01))
P00 = np.transpose(sum(P00) / len(P00))

print("The model hidden transition matrix is:")
print("Trnasition | out=0 | out =1 |")
print("-----------+-------+--------+")
np.set_printoptions(suppress=True, precision=5)
print("P(x|1←1):  ", end=""), print(P11),
print("P(x|1←0):  ", end=""), print(P10)
print("P(x|0←1):  ", end=""), print(P01)
print("P(x|0←0):  ", end=""), print(P00)

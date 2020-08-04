import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def plot(x, y, title, legends=None):  # Legends should be list. Title is string
    plt.title(title)
    for i in range(len(y)):
        plt.plot(x, y[i])
    if legends is not None:
        plt.legend(legends)
    plt.grid(True)
    # TODO: Uncomment
    # plt.savefig("imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_MA_model-" + title)
    plt.show()

# Defining the process:
N = 10005
U_train = np.random.normal(0, 1, N)  # for Fit
a1 = 5
a2 = a3 = a4 = a5 = -1
x_train = []
y_train = []
# Making the test:
U_test = np.random.normal(0, 1, N)   # for Predition
x_test = []
y_test = []

for i in range (5):
    y_train.append(U_train[i])
    y_test.append(U_test[i])

# Gather samples:
x_axis = np.transpose(np.array(range(5, N)))
for i in range(5, N):
    y_train = np.append(y_train, [U_train[i] + a1 * U_train[i-1] + a2 * U_train[i-2]
                                  + a3 * U_train[i-3] + a4 * U_train[i-4] + a5 * U_train[i-5]], axis=0)
    y_test = np.append(y_test, [U_test[i] + a1 * U_test[i-1] + a2 * U_test[i-2]
                                  + a3 * U_test[i-3] + a4 * U_test[i-4] + a5 * U_test[i-5]], axis=0)
    x_train.append(y_train[i-5:i])
    x_test.append(y_test[i - 5:i])

x_train = np.array(x_train).reshape(-1, 5, 1)
y_train = y_train[5:N].reshape(-1, 1, 1)
x_test = np.array(x_test).reshape(-1, 5, 1)
y_test = y_test[5:N].reshape(-1, 1, 1)

# 10,000 samples of the process:
plot(x_axis, [y_train.reshape(10000,)], "Process of 10k samples")
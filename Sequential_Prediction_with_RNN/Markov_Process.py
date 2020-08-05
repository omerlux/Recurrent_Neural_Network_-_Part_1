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
            plt.plot(x, y[j], '1')
        else:
            plt.plot(x, y[j], '1')
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
    return [0 if x == 'A' else (0.5 if x == 'B' else 1) for x in markov_process]


def markov_inverter(markov_process):
    """@:returns a markov process of numbers - not letters"""
    return ['A' if x == 0 else ('B' if x == 0.5 else 'C') for x in markov_process]


# Defining the process:
N = 10 ** 5
samples = [random.choice(['A', 'B', 'C'])]
for i in range(N):
    samples.append(markov_next(samples[i]))

""" Section (a):
By calculating analytically:
    'A' = 5/7.25 = ~0.689
    'B' = 1/7.25 = ~0.137
    'C' = 1.25/7.25 = ~172
Which is as the @stationary_distribution in Section (b).
"""
# Section(b) - Generating the stationary distribution:
unique, counts = np.unique(samples, return_counts=True)
counts = counts / (N + 1)
stationary_distribution = dict(zip(unique, counts))

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
    x_train.append([y_train[i-1], y_train[i]])
    x_test.append([y_test[i - 1], y_test[i]])
x_train = np.array(x_train).reshape(-1, 2,)
y_train = np.array(y_train[1:N + 1]).reshape(-1, 1,)
x_test = np.array(x_test).reshape(-1, 2,)
x_test = np.array(x_test[1:N + 1]).reshape(-1, 1,)

# 50 samples of the process:
x_axis = np.transpose(np.array(range(1, 51)))
plot(x_axis, [y_train[:50]], "Markov process of 500 samples")
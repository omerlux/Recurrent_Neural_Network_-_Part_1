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
    plt.title("AR Model - " + title)
    for j in range(len(y)):
        if j == 1:
            plt.plot(x, y[j], '--')
        else:
            plt.plot(x, y[j])
    if legends is not None:
        plt.legend(legends)
    plt.grid(True)
    plt.savefig("imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_AR_model-" + title)
    plt.show()

# Defining the process:
f = np.random.uniform(0, 0.1, (3, 1))   # for Fit
a1 = 0.6
a2 = -0.5
a3 = -0.2
x_train = []
y_train = []
# Making the test:
p = np.random.uniform(0, 0.1, (3, 1))   # for Predition
x_test = []
y_test = []


# Gather samples:
N = 10003
x_axis = np.transpose(np.array(range(3, N)))
for i in range(3, N):
    f = np.append(f, [a1 * f[i-1] + a2 * f[i-2] + a3 * f[i-3] + np.random.uniform(0, 0.1)], axis=0)
    p = np.append(p, [a1 * p[i-1] + a2 * p[i-2] + a3 * p[i-3] + np.random.uniform(0, 0.1)], axis=0)
    x_train.append(f[i-3:i, 0])
    x_test.append(p[i-3:i, 0])

x_train = np.array(x_train)
y_train = f[3:N]
x_test = np.array(x_test)
y_test = p[3:N]

x_train = x_train.reshape(-1, 3, 1)
y_train = y_train.reshape(-1, 1, 1)
x_test = x_test.reshape(-1, 3, 1)
y_test = y_test.reshape(-1, 1, 1)

# 10,000 samples of the process:
plot(x_axis, [y_train.reshape(10000,)], "Process of 10k samples")

# --------------------MODEL - COMPILE & FIT--------------------

# Creating RNN model and fit it:
model_RNN = keras.Sequential()
# Add a LSTM layer with 64 internal units.
model_RNN.add(layers.LSTM(64, input_shape=(3, 1), return_sequences=True))   # time steps = 3, dim = 1
# Add a LSTM layer with 16 internal units.
model_RNN.add(layers.LSTM(64, return_sequences=True))
# Add a LSTM layer with 64 internal units.
model_RNN.add(layers.LSTM(64, return_sequences=False))
# Add a Dense layer with 1 units.
model_RNN.add(layers.Dense(1))
model_RNN.summary()
# keras.utils.plot_model(model_RNN, "imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_AR_Model_RNN.png', show_shapes=True) # Model scheme

model_RNN.compile(
    loss='mse',
    optimizer="adam",
    metrics=['mae'],
)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_AR_Model_RNN"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_RNN.fit(
    x_train, y_train, batch_size=32, epochs=10, callbacks=[tensorboard_callback], verbose=2
)

# Prediction:
predict_test = model_RNN.predict(x_test)
plot(x_axis[:100], [predict_test[:100, 0].reshape(100,), y_test[:100].reshape(100,)], "RNN - Process Prediction of 100 samples",
     ["Predictions", "Ground Truth"])

# ----------- Model2 without ANN ---------------

# Creating RNN model and fit it:
model_ANN = keras.Sequential()
# Add a Dense layer with 64 internal units.
model_ANN.add(layers.Dense(64, input_shape=(3, 1), activation='relu'))   # time steps = 3, dim = 1
# Add a Dense layer with 64 internal units.
model_ANN.add(layers.Dense(64, activation='relu'))
# Add a Dense layer with 64 internal units.
model_ANN.add(layers.Dense(64, activation='relu'))
# Add a Dense layer with 1 units.
model_ANN.add(layers.Dense(1))
model_ANN.summary()
#keras.utils.plot_model(model_ANN, "imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_AR_Model_ANN.png', show_shapes=True) # Model scheme

model_ANN.compile(
    loss='mse',
    optimizer="adam",
    metrics=['mae'],
)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_AR_Model_ANN"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_ANN.fit(
    x_train, y_train, batch_size=32, epochs=10, callbacks=[tensorboard_callback], verbose=2
)

# Prediction:
predict_test_ANN = model_ANN.predict(x_test)
plot(x_axis[:100], [predict_test_ANN[:100, 0].reshape(100,), y_test[:100].reshape(100,)], "ANN - Process Prediction of 100 samples",
     ["Predictions", "Ground Truth"])

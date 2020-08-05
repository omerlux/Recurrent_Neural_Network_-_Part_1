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
    plt.title("MA Model - " + title)
    for j in range(len(y)):
        if j == 1:
            plt.plot(x, y[j], '--')
        else:
            plt.plot(x, y[j])
    if legends is not None:
        plt.legend(legends)
    plt.grid(True)
    plt.savefig("imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_MA_model-" + title)
    plt.show()


# Defining the process:
N = 10005
U_train = np.random.normal(0, 1, N)  # for Fit
a1 = 5
a2 = a3 = a4 = a5 = -1
x_train = []
y_train = []
# Making the test:
U_test = np.random.normal(0, 1, N)  # for Predition
x_test = []
y_test = []

for i in range(5):
    y_train.append(U_train[i])
    y_test.append(U_test[i])

# Gather samples:
x_axis = np.transpose(np.array(range(5, N)))
for i in range(5, N):
    y_train = np.append(y_train, [U_train[i] + a1 * U_train[i - 1] + a2 * U_train[i - 2]
                                  + a3 * U_train[i - 3] + a4 * U_train[i - 4] + a5 * U_train[i - 5]], axis=0)
    y_test = np.append(y_test, [U_test[i] + a1 * U_test[i - 1] + a2 * U_test[i - 2]
                                + a3 * U_test[i - 3] + a4 * U_test[i - 4] + a5 * U_test[i - 5]], axis=0)
    x_train.append(y_train[i - 5:i])
    x_test.append(y_test[i - 5:i])

x_train = np.array(x_train).reshape(-1, 5, 1)
y_train = y_train[5:N].reshape(-1, 1, )
x_test = np.array(x_test).reshape(-1, 5, 1)
y_test = y_test[5:N].reshape(-1, 1, )

# 10,000 samples of the process:
plot(x_axis, [y_train.reshape(10000, )], "Process of 10k samples")

# Normalize data:
add = min(np.min(y_test), np.min(y_train))  # data = data + add
max = max(np.max(y_train), np.max(y_test)) - min(np.min(y_train), np.min(y_test))  # max = max(data + add)
x_train = (x_train - add) / max
y_train = (y_train - add) / max
x_test = (x_test - add) / max
y_test = (y_test - add) / max
plot(x_axis, [y_train.reshape(10000, )], "Process of 10k samples - normalized")

# --------------------MODEL - COMPILE & FIT--------------------

# Creating RNN model and fit it:
model_RNN = keras.Sequential()
# Add a LSTM layer with 64 internal units.
model_RNN.add(layers.LSTM(64, input_shape=(5, 1), return_sequences=True))  # time steps = 3, dim = 1
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
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_MA_Model_RNN"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_RNN.fit(
    x_train, y_train, batch_size=32, epochs=10, callbacks=[tensorboard_callback], verbose=2
)

# Prediction:
predict_test = model_RNN.predict(x_test)
plot(x_axis[:100], [predict_test[:100, 0], y_test[:100, 0]], "RNN - Process Prediction of 100 samples",
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
# keras.utils.plot_model(model_ANN, "imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_AR_Model_ANN.png',
# show_shapes=True) # Model scheme

model_ANN.compile(
    loss='mse',
    optimizer="adam",
    metrics=['mae'],
)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_MA_Model_ANN"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_ANN.fit(
    x_train, y_train, batch_size=32, epochs=10, callbacks=[tensorboard_callback], verbose=2
)

# Prediction:
predict_test_ANN = model_ANN.predict(x_test)
plot(x_axis[:100], [predict_test_ANN[:100, 0], y_test[:100, 0]], "ANN - Process Prediction of 100 samples",
     ["Predictions", "Ground Truth"])


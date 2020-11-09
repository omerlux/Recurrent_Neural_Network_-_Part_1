import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from statsmodels.regression.linear_model import yule_walker
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


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


save = True
epochs = 50
past = 5

# Defining the process:
f = np.random.uniform(0, 0.1, (3, 1))  # for Fit
a1 = 0.6
a2 = -0.5
a3 = -0.2
x_train = []
y_train = []
# Making the test:
p = np.random.uniform(0, 0.1, (3, 1))  # for Predition
x_test = []
y_test = []

# Gather samples:
N = 10000 + past
x_axis = np.transpose(np.array(range(past, N)))
for i in range(3, N):
    f = np.append(f, [a1 * f[i - 1] + a2 * f[i - 2] + a3 * f[i - 3] + np.random.uniform(0, 0.1)], axis=0)
    p = np.append(p, [a1 * p[i - 1] + a2 * p[i - 2] + a3 * p[i - 3] + np.random.uniform(0, 0.1)], axis=0)
    if i >= past:
        x_train.append(f[i - past:i, 0])
        x_test.append(p[i - past:i, 0])

min = min(min(np.array([x_train]).flat), min(np.array([x_test]).flat))
max = max(max(np.array([x_train]).flat), max(np.array([x_test]).flat))

x_train = np.array(x_train)
y_train = f[past:N]
x_test = np.array(x_test)
y_test = p[past:N]

x_train = x_train.reshape(-1, past, 1)
y_train = y_train.reshape(-1, 1, 1)
x_test = x_test.reshape(-1, past, 1)
y_test = y_test.reshape(-1, 1, 1)

# 10,000 samples of the process:
plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x_axis, y_train.reshape(10000, ), linewidth=1)
plt.grid(True, which='both', axis='both')
plt.title('AR Model - Process of 10k samples')
plt.xlabel('Time')
plt.ylabel('Value')
if save:
    plt.savefig("./imgs/AR Model - Process of 10k samples.png", dpi=800)
plt.show()

# Scaling to normalize values:
# x_train_scaled = (np.array(x_train) - min) / (max - min)
# y_train_scaled = (np.array(y_train) - min) / (max - min)
# y_test_scaled =  (np.array(y_test) - min) / (max - min)


# --------------------MODEL - COMPILE & FIT--------------------

# Creating RNN model and fit it:
model_RNN = keras.Sequential()
# Add a LSTM layer with 16 internal units.
model_RNN.add(layers.LSTM(16, input_shape=(past, 1), return_sequences=True))  # time steps = 3, dim = 1
# Add a LSTM layer with 16 internal units.
model_RNN.add(layers.LSTM(16, return_sequences=False))
# Add a Dense layer with 1 units.
model_RNN.add(layers.Dense(1))
model_RNN.summary()
# keras.utils.plot_model(model_RNN, "imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_AR_Model_RNN.png', show_shapes=True) # Model scheme

model_RNN.compile(
    loss='mse',
    optimizer="adam",
    metrics=['mae'],
)
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_AR_Model_RNN"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model_RNN.fit(
    x_train, y_train, batch_size=32, epochs=epochs, callbacks=[], verbose=2
)

# Prediction:
predict_test = model_RNN.predict(x_test)

# un-normalizing back...
# predict_test = np.array(predict_test) * (max - min) + min


plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x_axis[:100], predict_test[:100, 0].reshape(100, ), linewidth=1, label='Predictions')
plt.plot(x_axis[:100], y_test[:100].reshape(100, ), linewidth=1, label='Ground Truth', linestyle='dashed')
plt.grid(True, which='both', axis='both')
plt.title('AR Model - LSTM Process Prediction of 100 samples')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
if save:
    plt.savefig("./imgs/AR Model - LSTM Process Prediction.png", dpi=800)
plt.show()

# ----------- Model2 without ANN ---------------

# Creating RNN model and fit it:
model_ANN = keras.Sequential()
model_ANN.add(keras.Input(shape=(past,)))
# Add a Dense layer with 64 internal units.
model_ANN.add(layers.Dense(64, activation='tanh'))  # time steps = 3, dim = 1
# Add a Dense layer with 64 internal units.
model_ANN.add(layers.Dense(64, activation='tanh'))
# Add a Dense layer with 1 units.
model_ANN.add(layers.Dense(1, activation='linear'))
model_ANN.summary()
# keras.utils.plot_model(model_ANN, "imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_AR_Model_ANN.png', show_shapes=True) # Model scheme

model_ANN.compile(
    loss='mse',
    optimizer="adam",
    metrics=['mae'],
)
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_AR_Model_ANN"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history2 = model_ANN.fit(
    x_train.reshape(-1, past,), y_train, batch_size=32, epochs=epochs, callbacks=[], verbose=2
)

# Prediction:
predict_test_ANN = model_ANN.predict(x_test.reshape(-1, past,))

# un-normalizing back...
# predict_test_ANN = np.array(predict_test_ANN) * (max - min) + min


plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x_axis[:100], predict_test_ANN[:100, 0].reshape(100, ), linewidth=1, label='Predictions')
plt.plot(x_axis[:100], y_test[:100].reshape(100, ), linewidth=1, label='Ground Truth', linestyle='dashed')
plt.grid(True, which='both', axis='both')
plt.title('AR Model - ANN Process Prediction of 100 samples')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
if save:
    plt.savefig("./imgs/AR Model - ANN Process Prediction.png", dpi=800)
plt.show()

# Graphs - TRAINING MSE
x = range(epochs)
train_loss = history.history['loss']
train_loss2 = history2.history['loss']
plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x, train_loss, linewidth=1, label='LSTM training')
plt.plot(x, train_loss2, linewidth=1, label='ANN training')
plt.grid(True, which='both', axis='both')
plt.title('AR Model - MSE of ANN vs LSTM')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
if save:
    plt.savefig("./imgs/AR Model - Training MSE.png", dpi=800)
plt.show()

# Yule-Walker
rho, sigma = yule_walker(y_train, order=3, method="mle")


yw_pred = np.ndarray.flatten(y_test)[:3]
for i in range(3,100):
    yw_pred = np.append(yw_pred, [rho[0] * yw_pred[i - 1] + rho[1] * yw_pred[i - 2] + rho[2] * yw_pred[i - 3] + np.random.uniform(0, 0.1)], axis=0)

plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x_axis[:100], yw_pred, linewidth=1, label='Predictions')
plt.plot(x_axis[:100], y_test[:100].reshape(100, ), linewidth=1, label='Ground Truth', linestyle='dashed')
plt.grid(True, which='both', axis='both')
plt.title('AR Model - Yule-Walker Prediction of 100 samples')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
if save:
    plt.savefig("./imgs/AR Model - Yule-Walker Prediction.png", dpi=800)
plt.show()


# Graphs - min MSE vs LSTM
x = range(epochs)
train_loss = history.history['loss']
plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x, train_loss, linewidth=1, label='LSTM training MSE')
plt.plot(x, [(0.1**2)/12]*epochs, linewidth=1, label='min(MSE)')
plt.grid(True, which='both', axis='both')
plt.title('AR Model - MSE of LSTM vs min(MSE)')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
if save:
    plt.savefig("./imgs/AR Model - MSE and min.png", dpi=800)
plt.show()
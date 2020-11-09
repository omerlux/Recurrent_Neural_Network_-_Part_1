import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from statsmodels.regression.linear_model import yule_walker
import tensorflow as tf

np.random.seed(0)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


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


save = True
only_yw = True
epochs = 300
past = 10

# Defining the process:
N = 10000 + past
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
    if i >= past:
        x_train.append(y_train[i - past:i])
        x_test.append(y_test[i - past:i])

x_train = np.array(x_train).reshape(-1, past, 1)
y_train = y_train[past:N].reshape(-1, 1, )
x_test = np.array(x_test).reshape(-1, past, 1)
y_test = y_test[past:N].reshape(-1, 1, )

# 10,000 samples of the process:
plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x_axis[past - 5:], y_train.reshape(10000, ), linewidth=1)
plt.grid(True, which='both', axis='both')
plt.title('MA Model - Process of 10k samples')
plt.xlabel('Time')
plt.ylabel('Value')
if save and not only_yw:
    plt.savefig("./imgs/MA Model - Process of 10k samples.png", dpi=800)
plt.show()

# # Normalize data:
# max = max(max(np.max(y_train), np.max(y_test)), -min(np.min(y_train), np.min(y_test)))
# # x_train = (x_train) / max
# # y_train = (y_train) / max
# # x_test = (x_test) / max
# # y_test = (y_test) / max
# plt.rcParams['axes.facecolor'] = 'white'
# plt.plot(x_axis[past-5:], y_train.reshape(10000, ), linewidth=1)
# plt.grid(True, which='both', axis='both')
# plt.title('MA Model - Process of 10k samples - Scaled')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.show()

# --------------------MODEL - COMPILE & FIT--------------------

# Creating RNN model and fit it:

model_RNN = keras.Sequential()
model_RNN.add(layers.LSTM(256, input_shape=(past, 1), return_sequences=True))  # time steps = 3, dim = 1
model_RNN.add(layers.Dropout(rate=0.3))
model_RNN.add(layers.LSTM(128, return_sequences=True))
model_RNN.add(layers.Dropout(rate=0.3))
model_RNN.add(layers.LSTM(64, return_sequences=False))
model_RNN.add(layers.Dropout(rate=0.3))
model_RNN.add(layers.Dense(1))
model_RNN.summary()
# keras.utils.plot_model(model_RNN, "imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_AR_Model_RNN.png', show_shapes=True) # Model scheme

model_RNN.compile(
    loss='mse',
    optimizer='RMSProp',  # keras.optimizers.SGD(learning_rate=0.15),
    metrics=['mae'],
)
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_MA_Model_RNN"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model_RNN.fit(
    x_train, y_train, batch_size=32, epochs=epochs, callbacks=[], verbose=2
)

# Prediction:
predict_test = model_RNN.predict(x_test)
# plot(x_axis[:100], [predict_test[:100, 0], y_test[:100, 0]], "RNN - Process Prediction of 100 samples",
#      ["Predictions", "Ground Truth"])

plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x_axis[:100], predict_test[:100, 0].reshape(100, ), linewidth=1, label='Predictions')
plt.plot(x_axis[:100], y_test[:100].reshape(100, ), linewidth=1, label='Ground Truth', linestyle='dashed')
plt.grid(True, which='both', axis='both')
plt.title('MA Model - LSTM Process Prediction of 100 samples')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
if save and not only_yw:
    plt.savefig("./imgs/MA Model - LSTM Process Prediction.png", dpi=800)
plt.show()

# Graphs
x = range(epochs)
train_loss = history.history['loss']
plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x, train_loss, linewidth=1, label='LSTM training')
plt.grid(True, which='both', axis='both')
plt.title('MA Model - Training MSE of LSTM')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
if save and not only_yw:
    plt.savefig("./imgs/MA Model - Training MSE.png", dpi=800)
plt.show()

# ----------- Model2 without ANN ---------------

# # Creating RNN model and fit it:
# model_ANN = keras.Sequential()
# model_ANN.add(keras.Input(shape=(past,)))
# # Add a Dense layer with 64 internal units.
# model_ANN.add(layers.Dense(64, activation='relu'))  # time steps = 3, dim = 1
# # Add a Dense layer with 64 internal units.
# model_ANN.add(layers.Dense(64, activation='relu'))
# # Add a Dense layer with 64 internal units.
# model_ANN.add(layers.Dense(64, activation='relu'))
# model_ANN.add(layers.Dense(32, activation='relu'))
# # Add a Dense layer with 1 units.
# model_ANN.add(layers.Dense(1))
# model_ANN.summary()
# # keras.utils.plot_model(model_ANN, "imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_AR_Model_ANN.png',
# # show_shapes=True) # Model scheme
#
# model_ANN.compile(
#     loss='mse',
#     optimizer="adam",
#     metrics=['mae'],
# )
# # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_MA_Model_ANN"
# # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# history2 = model_ANN.fit(
#     x_train.reshape(-1, past, ), y_train, batch_size=32, epochs=epochs, callbacks=[], verbose=2
# )
#
# # Prediction:
# predict_test_ANN = model_ANN.predict(x_test.reshape(-1, past, ))
# # plot(x_axis[:100], [predict_test_ANN[:100, 0], y_test[:100, 0]], "ANN - Process Prediction of 100 samples",
# #      ["Predictions", "Ground Truth"])
# plt.rcParams['axes.facecolor'] = 'white'
# plt.plot(x_axis[:100], predict_test_ANN[:100, 0].reshape(100, ), linewidth=1, label='Predictions')
# plt.plot(x_axis[:100], y_test[:100].reshape(100, ), linewidth=1, label='Ground Truth', linestyle='dashed')
# plt.grid(True, which='both', axis='both')
# plt.title('MA Model - ANN Process Prediction of 100 samples')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# if save:
#     plt.savefig("./imgs/MA Model - ANN Process Prediction.png", dpi=800)
# plt.show()
#
# from sklearn.metrics import mean_squared_error
#
# mse = [mean_squared_error(predict_test[i] * max, y_test[i] * max) for i in range(100)]
# mse2 = [mean_squared_error(predict_test_ANN[i] * max, y_test[i] * max) for i in range(100)]
#
# plt.rcParams['axes.facecolor'] = 'white'
# plt.plot(x_axis[:100], mse, linewidth=1, label='LSTM test MSE')
# plt.plot(x_axis[:100], mse2, linewidth=1, label='ANN test MSE')
# plt.grid(True, which='both', axis='both')
# plt.title('MA Model - Test MSE of ANN vs LSTM')
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# plt.legend()
# if save:
#     plt.savefig("./imgs/MA Model - Test MSE.png", dpi=800)
# plt.show()
#
# # Graphs
# x = range(epochs)
# train_loss = history.history['loss']
# train_loss2 = history2.history['loss']
# plt.rcParams['axes.facecolor'] = 'white'
# plt.plot(x, train_loss, linewidth=1, label='LSTM training')
# plt.plot(x, train_loss2, linewidth=1, label='ANN training')
# plt.grid(True, which='both', axis='both')
# plt.title('MA Model - Training MSE of ANN vs LSTM')
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# plt.legend()
# if save:
#     plt.savefig("./imgs/MA Model - Training MSE.png", dpi=800)
# plt.show()

# Yule-Walker

rho5, sigma5 = yule_walker(y_train, order=5, method="mle")
rho10, sigma10 = yule_walker(y_train, order=10, method="mle")
rho50, sigma50 = yule_walker(y_train, order=50, method="mle")
rho250, sigma250 = yule_walker(y_train, order=250, method="mle")

yw5_pred = np.ndarray.flatten(y_test)[:5]
for i in range(5, 10000):
    yw5_pred = np.append(yw5_pred, [np.dot(rho5, yw5_pred[-5:])], axis=0)
plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x_axis[5:105], yw5_pred[5:105], linewidth=1, label='Predictions')
plt.plot(x_axis[5:105], y_test[5:105].reshape(100, ), linewidth=1, label='Ground Truth', linestyle='dashed')
plt.grid(True, which='both', axis='both')
plt.title('MA Model - Yule-Walker AR5 Prediction of 100 samples')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
if save:
    plt.savefig("./imgs/MA Model - YW5.png", dpi=800)
plt.show()

yw10_pred = np.ndarray.flatten(y_test)[:10]
for i in range(10, 10000):
    yw10_pred = np.append(yw10_pred, [np.dot(rho10, yw10_pred[-10:])], axis=0)
plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x_axis[10:110], yw10_pred[10:110], linewidth=1, label='Predictions')
plt.plot(x_axis[10:110], y_test[10:110].reshape(100, ), linewidth=1, label='Ground Truth', linestyle='dashed')
plt.grid(True, which='both', axis='both')
plt.title('MA Model - Yule-Walker AR10 Prediction of 100 samples')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
if save:
    plt.savefig("./imgs/MA Model - YW10.png", dpi=800)
plt.show()

yw50_pred = np.ndarray.flatten(y_test)[:50]
for i in range(50, 10000):
    yw50_pred = np.append(yw50_pred, [np.dot(rho50, yw50_pred[-50:])], axis=0)
plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x_axis[50:150], yw50_pred[50:150], linewidth=1, label='Predictions')
plt.plot(x_axis[50:150], y_test[50:150].reshape(100, ), linewidth=1, label='Ground Truth', linestyle='dashed')
plt.grid(True, which='both', axis='both')
plt.title('MA Model - Yule-Walker AR50 Prediction of 100 samples')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
if save:
    plt.savefig("./imgs/MA Model - YW50.png", dpi=800)
plt.show()

yw250_pred = np.ndarray.flatten(y_test)[:250]
for i in range(250, 10000):
    yw250_pred = np.append(yw250_pred, [np.dot(rho250, yw250_pred[-250:])], axis=0)
plt.rcParams['axes.facecolor'] = 'white'
plt.plot(x_axis[250:350], yw250_pred[250:350], linewidth=1, label='Predictions')
plt.plot(x_axis[250:350], y_test[250:350].reshape(100, ), linewidth=1, label='Ground Truth', linestyle='dashed')
plt.grid(True, which='both', axis='both')
plt.title('MA Model - Yule-Walker AR250 Prediction of 100 samples')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
if save:
    plt.savefig("./imgs/MA Model - YW250.png", dpi=800)
plt.show()

# mse5 = (1 / 9995) * (np.linalg.norm(yw5_pred[5:] - y_test[5:], 2)) ** 2
# print("yw5 MSE: ", mse5)
# mse10 = (1 / 9990) * (np.linalg.norm(yw10_pred[10:] - y_test[10:], 2)) ** 2
# print("yw10 MSE: ", mse10)
# mse50 = (1 / 9950) * (np.linalg.norm(yw50_pred[50:] - y_test[50:], 2)) ** 2
# print("yw50 MSE: ", mse50)
# mse250 = (1 / 9750) * (np.linalg.norm(yw250_pred[250:] - y_test[250:], 2)) ** 2
# print("yw250 MSE: ", mse250)
# mse_LSTM = (1 / 9995) * (np.linalg.norm(predict_test[5:] - y_test[5:], 2)) ** 2
# print("LSTM MSE: ", mse_LSTM)
# a=0
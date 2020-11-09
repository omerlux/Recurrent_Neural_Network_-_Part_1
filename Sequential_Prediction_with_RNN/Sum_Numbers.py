import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def SampleCreator(bits=10):
    # _x1 = np.random.randint(2 ** (bits - 1))
    # _x2 = np.random.randint(2 ** (bits - 1))
    # _y = _x1 + _x2
    #
    # _x1 = bin(_x1)[2:].zfill(bits)
    # _x2 = bin(_x2)[2:].zfill(bits)
    # _y = bin(_y)[2:].zfill(bits)
    #
    # _x1 = np.flip(np.array(list(_x1), dtype=int))
    # _x2 = np.flip(np.array(list(_x2), dtype=int))
    # _y = np.flip(np.array(list(_y), dtype=int))
    _x1 = [np.random.randint(0, 2) for _ in range(bits - 1)]
    _x2 = [np.random.randint(0, 2) for _ in range(bits - 1)]
    _x1.append(0)
    _x2.append(0)
    _y = []
    _c = []
    _y.append(_x1[0] + _x2[0])
    _c.append(1 if _y[0] > 1 else 0)
    for i in range(1, bits-1):
        res = _x1[i] + _x2[i] + _c[i - 1]
        _y.append(min(res, 1))
        _c.append(1 if res > 1 else 0)
    _y.append(_c[-1])
    return [_x1, _x2], _y


if __name__ == "__main__":
    N = 10 ** 4  # Samples train
    M = 10 ** 2  # Samples test
    steps_train = 100
    steps_test = 200
    epochs = 25
    save = True

    data_train = [SampleCreator(bits=steps_train) for i in range(N)]
    data_test = [SampleCreator(bits=steps_test) for i in range(M)]
    x_train, y_train = np.array([np.transpose(z) for z, y in data_train]), \
                    np.array([y for z, y in data_train])
    x_test, y_test = np.array([np.transpose(z) for z, y in data_test]), \
                   np.array([y for z, y in data_test])

    y_train, y_test = y_train.reshape(-1, steps_train,), y_test.reshape(-1, steps_test,)

    # --------------------MODEL - COMPILE & FIT--------------------
    # Creating RNN model and fit it:
    model_RNN = keras.Sequential()
    # Add a LSTM layer with 64 internal units.
    model_RNN.add(layers.LSTM(128, input_shape=(None, 2), return_sequences=True))  # time steps = 'dynamic', dim = 2
    # Add a Dense layer with 1 units - output is only 1 or 0
    model_RNN.add(layers.Dense(1))
    model_RNN.summary()
    # keras.utils.plot_model(model_RNN, "imgs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_Markov_Model_RNN.png', show_shapes=True) # Model scheme

    model_RNN.compile(
        loss="binary_crossentropy",
        optimizer="Adam",
        metrics=['accuracy'],
    )
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_Sum"
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model_RNN.fit(
        x_train, y_train, batch_size=25, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[]
    )
    # Graphs
    x = range(epochs)
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']
    plt.rcParams['axes.facecolor'] = 'white'
    plt.plot(x, train_loss, linewidth=1, label='LSTM training')
    plt.plot(x, valid_loss, linewidth=1, label='LSTM testing')
    plt.grid(True, which='both', axis='both')
    plt.title('Binary Sum - Training CE of LSTM')
    plt.xlabel('Epochs')
    plt.ylabel('CE')
    plt.legend()
    if save:
        plt.savefig("./imgs/Binary Sum - Training CE of LSTM", dpi=800)
    plt.show()

    # Prediction:
    # We will check out all the different 5 consecutive numbers from the process

    output_model = model_RNN.predict(x_test)
    results = model_RNN.evaluate(x_test, y_test, batch_size=32)
    print("Test loss:\t\t%f \n"
          "Test accuracy:\t%.2f%%" % (results[0], results[1] * 100))

    plt.rcParams['axes.facecolor'] = 'white'
    plt.plot(range(steps_test), output_model[0], linewidth=1, label='Predictions')
    plt.plot(range(steps_test), y_test[0], linewidth=1, label='Ground Truth', linestyle='dashed')
    plt.grid(True, which='both', axis='both')
    plt.title('Binary Sum - LSTM Process Prediction of 100 samples')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    if save:
        plt.savefig("./imgs/Binary Sum - LSTM Process Prediction.png", dpi=800)
    plt.show()

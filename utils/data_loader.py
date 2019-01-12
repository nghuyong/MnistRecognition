#!/usr/bin/env python
# encoding: utf-8
import keras
import numpy as np


def load_data():
    data_set = np.load('./data/mnist.npz')
    x_train, y_train = data_set['x_train'], data_set['y_train']
    x_test, y_test = data_set['x_test'], data_set['y_test']
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # 转换成one-hot
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    load_data()

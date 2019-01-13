#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class Examples:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.count = len(x)
        self.indices = np.arange(self.count)

    def shuffle(self):
        np.random.shuffle(self.indices)

    def batch(self, index, batchsize):
        i = self.indices[(index * batchsize):((index + 1) * batchsize)]
        return self.x[i], self.y[i]

    def examples(self):
        return self.x, self.y


class Dataset:

    @staticmethod
    def load_mnist():
        old_v = tf.logging.get_verbosity()
        tf.logging.set_verbosity(tf.logging.ERROR)
        mnist = input_data.read_data_sets("./data/", one_hot=True)
        tf.logging.set_verbosity(old_v)
        return Dataset(mnist.train.images, mnist.train.labels, mnist.validation.images, mnist.validation.labels,
                       mnist.test.images, mnist.test.labels)

    def __init__(self, train_x, train_y, validation_x, validation_y, test_x, test_y):
        self.train = Examples(train_x, train_y)
        self.validation = Examples(validation_x, validation_y)
        self.test = Examples(test_x, test_y)

    def slice(self, train_size, validation_size, test_size):
        return Dataset(self.train.x[:train_size], self.train.y[:train_size],
                       self.validation.x[:validation_size], self.validation.y[:validation_size],
                       self.test.x[:test_size], self.test.y[:test_size])


if __name__ == "__main__":
    mnist = Dataset.load_mnist()
    print(mnist.train.x[0])
    print('train_x shape', mnist.train.x.shape)
    print('train_y shape', mnist.train.y.shape)
    print('validation_x shape', mnist.validation.x.shape)
    print('validation_y shape', mnist.validation.y.shape)
    print('test_x shape', mnist.test.x.shape)
    print('test_y shape', mnist.test.y.shape)

    mnist_small = mnist.slice(1000, 500, mnist.test.count)
    print('mini train_x shape', mnist_small.train.x.shape)
    print('mini train_y shape', mnist_small.train.y.shape)
    print('mini validation_x shape', mnist_small.validation.x.shape)
    print('mini validation_y shape', mnist_small.validation.y.shape)
    print('mini test_x shape', mnist_small.test.x.shape)
    print('mini test_y shape', mnist_small.test.y.shape)

#!/usr/bin/env python
# encoding: utf-8
import keras
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from models.config import BaseConfig
from utils.data_loader import load_data


class CNNFcModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.epochs = 3


class CNNFcModel:
    def __init__(self):
        self.config = CNNFcModelConfig()
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        input_shape = (self.config.img_rows, self.config.img_cols, 1)
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.config.num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model

    def train_model(self):
        x_train, y_train, x_test, y_test = load_data()
        x_train = x_train.reshape(x_train.shape[0], self.config.img_rows, self.config.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], self.config.img_rows, self.config.img_cols, 1)

        # 添加tensorboard
        tb_callback = TensorBoard(log_dir='./logs',  # log 目录
                                  histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                  write_graph=True,  # 是否存储网络结构图
                                  write_grads=True,  # 是否可视化梯度直方图
                                  write_images=True,  # 是否可视化参数
                                  embeddings_freq=0,
                                  embeddings_layer_names=None,
                                  update_freq='batch',
                                  embeddings_metadata=None)
        self.model.fit(x_train, y_train,
                       batch_size=self.config.batch_size,
                       epochs=self.config.epochs,
                       verbose=1,
                       callbacks=[tb_callback],
                       validation_split=0.3)
        loss, score = self.model.evaluate(x_test, y_test, verbose=0)
        print(f'score one test:{score}')


if __name__ == "__main__":
    CNNFcModel().train_model()

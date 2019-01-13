from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Dropout, Flatten


# generic model design
def model_cnn_fc(actions):
    # unpack the actions from the list
    kernel_1, filters_1, kernel_2, filters_2 = actions

    ip = Input(shape=(28, 28, 1))
    x = Conv2D(filters_1, (kernel_1, kernel_1), padding='same', activation='relu')(ip)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters_2, (kernel_2, kernel_2), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(ip, x)
    return model


def model_fc(actions):
    # unpack the actions from the list
    hidden_size = actions[0]
    ip = Input(shape=(784,))
    x = Dense(hidden_size, activation='relu')(ip)
    x = Dense(10, activation='softmax')(x)
    model = Model(ip, x)
    return model

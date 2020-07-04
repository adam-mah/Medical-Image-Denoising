from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2DTranspose, Add, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.constraints import max_norm
from tensorflow_core.python.keras.layers import BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# def get_autoencoder_model(img_width=64, img_height=64):
#     encoder = Sequential()
#     encoder.add(Input(shape=(img_width, img_height, 1)))
#     encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     encoder.add(MaxPooling2D((2, 2), padding='same'))
#     encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     encoder.add(MaxPooling2D((2, 2), padding='same'))
#     # encoder.summary()
#     decoder = Sequential()
#     decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     decoder.add(UpSampling2D((2, 2)))
#     decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     decoder.add(UpSampling2D((2, 2)))
#     decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
#     # decoder.summary()
#     autoencoder = Sequential([encoder, decoder])
#     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#
#     return autoencoder


def get_autoencoder_model(img_width=64, img_height=64):
    autoencoder = Sequential()
    # Encoder
    autoencoder.add(Input(shape=(img_width, img_height, 1)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    # Decoder
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    autoencoder.summary()
    return autoencoder


def get_simple_autoencoder_model128(img_width=128, img_height=128):
    autoencoder = Sequential()
    autoencoder.add(Input(shape=(img_width, img_height, 1)))
    autoencoder.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(1, (5, 5), activation='sigmoid', padding='same'))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    autoencoder.summary()
    return autoencoder


def get_gated_connections(gatePercentageFactor, inputLayer):
    gateFactor = Input(tensor=tf.keras.backend.variable([gatePercentageFactor]))
    fractionG = Lambda(lambda x: x[0] * x[1])([inputLayer, gateFactor])
    complement = Lambda(lambda x: x[0] - x[1])([inputLayer, fractionG])
    return gateFactor, fractionG, complement


# x is conv layer
# y is de-conv layer
# gf is gating factor
# fg is fractional input from gate
# c is complement ie remaining fraction from the gate
# jt joining tensor of convolution layer and previous de-conv layer

def get_cnn_dsc_architecture(model_path=None, img_width=64, img_height=64):
    tf.disable_v2_behavior()
    input_img = Input(shape=(img_width, img_height, 1))  # adapt this if using `channels_first` image data format
    x1 = Conv2D(64, (4, 4), activation='relu', padding='same')(input_img)
    gf1, fg1, c1 = get_gated_connections(0.1, x1)

    x = MaxPooling2D((2, 2), padding='same')(fg1)
    x2 = Conv2D(64, (4, 4), activation='relu', padding='same')(x)
    gf2, fg2, c2 = get_gated_connections(0.2, x2)

    x = MaxPooling2D((2, 2), padding='same')(fg2)
    x3 = Conv2D(128, (4, 4), activation='relu', padding='same')(x)
    gf3, fg3, c3 = get_gated_connections(0.3, x3)

    x = MaxPooling2D((2, 2), padding='same')(x3)
    x4 = Conv2D(256, (4, 4), activation='relu', padding='same')(x)
    gf4, fg4, c4 = get_gated_connections(0.4, x4)

    x = MaxPooling2D((2, 2), padding='same')(x4)
    x5 = Conv2D(512, (4, 4), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x5)
    y1 = Conv2DTranspose(256, (4, 4), activation='relu', padding='same')(x)
    jt4 = Add()([y1, c4])
    x = UpSampling2D((2, 2))(jt4)

    y2 = Conv2DTranspose(128, (4, 4), activation='relu', padding='same')(x)
    jt3 = Add()([y2, c3])
    x = UpSampling2D((2, 2))(jt3)

    y3 = Conv2DTranspose(64, (4, 4), activation='relu', padding='same')(x)
    jt2 = Add()([y3, c2])
    x = UpSampling2D((2, 2))(jt2)

    jt1 = Add()([x, c1])
    y4 = Conv2DTranspose(64, (4, 4), activation='relu', padding='same')(jt1)
    y5 = Conv2DTranspose(1, (4, 4), activation='relu', padding='same')(y4)

    layers = y5

    sym_autoencoder = Model([input_img, gf1, gf2, gf3, gf4], layers)
    sym_autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return sym_autoencoder

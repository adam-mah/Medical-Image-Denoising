from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow_core.python.keras import Input, Model
import tensorflow as tf


def get_simple_autoencoder_model(model_path=None):
    encoder = Sequential()
    encoder.add(Input(shape=(64, 64, 1)))
    encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    encoder.add(MaxPooling2D((2, 2), padding='same'))
    encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    encoder.add(MaxPooling2D((2, 2), padding='same'))
    decoder = Sequential()
    decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return autoencoder

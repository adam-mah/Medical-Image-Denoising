import cv2
import tensorflow as tf
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
import numpy as np

import bm3d
import read_pgm

# Model configuration
img_width, img_height = 64, 64
batch_size = 5
no_epochs = 100
validation_split = 0.2
verbosity = 1
max_norm_value = 2.0
noise_factor = 0.2
number_of_visualizations = 6

rawimages = read_pgm.read_mini_mias()
images = np.zeros((322, img_width, img_height))
for i in range(rawimages.shape[0]):
    images[i] = cv2.resize(rawimages[i].reshape(1024, 1024), dsize=(img_width, img_height),
                           interpolation=cv2.INTER_CUBIC)

train_size = int(images.shape[0] * 0.9)
input_train = images[0:train_size]
target_train = images[0:train_size]
input_test = images[train_size:]
target_test = images[train_size:]

input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
input_shape = (img_width, img_height, 1)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Add noise
pure = input_train
pure_test = input_test
noise = np.random.normal(0, 1, pure.shape)
noise_test = np.random.normal(0, 1, pure_test.shape)
noisy_input = pure + noise_factor * noise
noisy_input_test = pure_test + noise_factor * noise_test

# Create the model
encoder = Sequential()
encoder.add(tf.keras.layers.Input(shape=(img_width, img_height, 1)))
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
model = Sequential([encoder, decoder])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(noisy_input, pure,
          epochs=no_epochs,
          batch_size=batch_size,
          validation_split=validation_split)
model.summary()
model.save("models/")

# Generate denoised images
samples = noisy_input_test[:number_of_visualizations]
targets = target_test[:number_of_visualizations]
denoised_images = model.predict(samples)

# Plot denoised images
for i in range(0, number_of_visualizations):
    # Get the sample and the reconstruction
    noisy_image = noisy_input_test[i][:, :, 0]
    pure_image = pure_test[i][:, :, 0]
    denoised_image = denoised_images[i][:, :, 0]
    input_class = targets[i]
    # Matplotlib preparations
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(8, 3.5)
    # Plot sample and reconstruciton
    axes[0].imshow(noisy_image, pyplot.cm.gray)
    axes[0].set_title('Noisy image')
    axes[1].imshow(pure_image, pyplot.cm.gray)
    axes[1].set_title('Pure image')
    axes[2].imshow(denoised_image, pyplot.cm.gray)
    axes[2].set_title('Denoised image')
    plt.show()

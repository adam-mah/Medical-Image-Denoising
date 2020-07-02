import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_core.python.keras.layers import LeakyReLU, BatchNormalization

# Model configuration
img_width, img_height = 28, 28
batch_size = 100
no_epochs = 5
validation_split = 0.2
verbosity = 1
max_norm_value = 2.0
noise_factor = 0.3
number_of_visualizations = 6

# Load MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(input_train, target_train), (input_test, target_test) = fashion_mnist.load_data()  # mnist.load_data()

# Reshape data based on channels first / channels last strategy.
# This is dependent on whether you use TF, Theano or CNTK as backend.
# Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
if K.image_data_format() == 'channels_first':
    input_train = input_train.reshape(input_train.shape[0], 1, img_width, img_height)
    input_test = input_test.reshape(input_test.shape[0], 1, img_width, img_height)
    input_shape = (1, img_width, img_height)
else:
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
encoder.add(tf.keras.layers.Input(shape=(28, 28, 1)))
encoder.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                   kernel_initializer='he_uniform', input_shape=input_shape))
encoder.add(Conv2D(32, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                   kernel_initializer='he_uniform'))
encoder.add(Conv2D(32, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                   kernel_initializer='he_uniform'))
encoder.add(Conv2D(32, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                   kernel_initializer='he_uniform'))
encoder.add(BatchNormalization(axis=-1))

decoder = Sequential()
decoder.add(Conv2DTranspose(32, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                            kernel_initializer='he_uniform'))
decoder.add(Conv2DTranspose(32, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                            kernel_initializer='he_uniform'))
decoder.add(Conv2DTranspose(32, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                            kernel_initializer='he_uniform'))
decoder.add(Conv2DTranspose(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                            kernel_initializer='he_uniform'))
decoder.add(BatchNormalization(axis=-1))
decoder.add(
    Conv2D(1, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='sigmoid', padding='same'))

model = Sequential([encoder, decoder])

# Compile and fit data
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(noisy_input, pure,
          epochs=no_epochs,
          batch_size=batch_size,
          validation_split=validation_split)
model.summary()
model.save("models-fashion/")

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
    axes[0].imshow(noisy_image)
    axes[0].set_title('Noisy image')
    axes[1].imshow(pure_image)
    axes[1].set_title('Pure image')
    axes[2].imshow(denoised_image)
    axes[2].set_title('Denoised image')
    fig.suptitle(f'MNIST target = {input_class}')
    plt.show()
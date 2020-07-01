import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from tensorflow_core.python.keras.callbacks import ModelCheckpoint
import bm3d
import autoencoder
import dataset_reader
from skimage.restoration import denoise_nl_means

# Model configuration
img_width, img_height = 64, 64
batch_size = 10
no_epochs = 50
validation_split = 0.2
verbosity = 1
max_norm_value = 2.0
noise_factor = 0.1
number_of_visualizations = 6

rawimages, rawimages2 = dataset_reader.read_all_datasets()  # read_pgm.read_mini_mias()
images = np.zeros((322, img_width, img_height))
images2 = np.zeros((120, img_width, img_width))
for i in range(rawimages.shape[0]):
    images[i] = cv2.resize(rawimages[i], dsize=(img_width, img_height),
                           interpolation=cv2.INTER_CUBIC)
for i in range(rawimages2.shape[0]):
    images2[i] = cv2.resize(rawimages2[i], dsize=(img_width, img_height),
                            interpolation=cv2.INTER_CUBIC)
images_set = np.append(images, images2, axis=0)
print(images_set.shape)
np.random.shuffle(images_set)

train_size = int(images_set.shape[0] * 0.9)
input_train = images_set[0:train_size]
target_train = images_set[0:train_size]
input_test = images_set[train_size:]
target_test = images_set[train_size:]

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

# denoised = bm3d.bm3d_denoise(noisy_input)
# plt.imshow(denoised[0][:, :, 0])
# plt.imshow(pure[0][:, :, 0])
# plt.show()

# for i in noisy_input:
#    cv2.imwrite('./data/denoise/image' + str(i) + '.png', noisy_input[i] * 255.)

# Create the model
model = autoencoder.get_simple_autoencoder_model()
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
fig, axes = plt.subplots(6, 3)
fig.set_size_inches(8, 3.5*6)
axes[0][0].set_title('Pure image')
axes[0][1].set_title('Noisy image')
axes[0][2].set_title('Denoised image')
# Plot denoised images
for i in range(0, number_of_visualizations):
    # Get the sample and the reconstruction
    noisy_image = noisy_input_test[i][:, :, 0]
    pure_image = pure_test[i][:, :, 0]
    denoised_image = denoised_images[i][:, :, 0]
    input_class = targets[i]

    # Plot sample and reconstruciton
    axes[i][0].imshow(pure_image, pyplot.cm.gray)
    axes[i][1].imshow(noisy_image, pyplot.cm.gray)
    axes[i][2].imshow(denoised_image, pyplot.cm.gray)
plt.show()

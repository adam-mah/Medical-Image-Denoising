import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
import tensorflow as tf
from matplotlib import pyplot
from tensorflow_core.python.keras.callbacks import EarlyStopping

import autoencoder
import measure
import dataset_reader
import denoiser
import pandas as pd

# Model configuration

img_width, img_height = 64, 64
batch_size = 10
no_epochs = 50
validation_split = 0
verbosity = 1
noise_prop = 0.5
noise_std = 1
noise_mean = 0
number_of_visualizations = 4

# rawimages, rawimages2, rawimages3 = dataset_reader.read_all_datasets()  # read_pgm.read_mini_mias()

rawimages = dataset_reader.read_mini_mias()  # Read mias dataset
images = np.zeros((322, img_width, img_height))
for i in range(rawimages.shape[0]):
    images[i] = cv2.resize(rawimages[i], dsize=(img_width, img_height),
                           interpolation=cv2.INTER_CUBIC)

# rawimages2 = dataset_reader.read_dental()  # Read dental dataset
# images2 = np.zeros((120, img_width, img_width))
# for i in range(rawimages2.shape[0]):
#     images2[i] = cv2.resize(rawimages2[i], dsize=(img_width, img_height),
#                             interpolation=cv2.INTER_CUBIC)
#
# rawimages3 = dataset_reader.read_covid()  # Read covid dataset
# images3 = np.zeros((329, img_width, img_width))
# for i in range(rawimages3.shape[0]):
#     images3[i] = cv2.resize(rawimages3[i], dsize=(img_width, img_height),
#                             interpolation=cv2.INTER_CUBIC)

rawimages4 = dataset_reader.read_dx()  # Read DX dataset
images4 = np.zeros((400, img_width, img_width))
for i in range(rawimages4.shape[0]):
    images4[i] = cv2.resize(rawimages4[i], dsize=(img_width, img_height),
                            interpolation=cv2.INTER_CUBIC)

#images_set = np.append(images, images4, axis=0)
# images_set = np.append(images_set, images3, axis=0)
# images_set = np.append(images_set, images4, axis=0)
#print(images_set.shape)
#np.random.shuffle(images_set)
#np.random.shuffle(images)
#np.random.shuffle(images4)

images_set = images[:int(images.shape[0]*0.9)]
images_set = np.append(images_set, images4[:int(images4.shape[0]*0.9)], axis=0)
images_set = np.append(images_set, images4[int(images4.shape[0]*0.9):], axis=0)
images_set = np.append(images_set, images[int(images.shape[0]*0.9):], axis=0)

train_size = int(images.shape[0]*0.9 + images4.shape[0]*0.9)  # int(images_set.shape[0] * 0.9)
input_train = images_set[0:train_size]
input_test = images_set[train_size:]
#np.random.shuffle(input_test)

input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
# input_shape = (img_width, img_height, 1)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Add gaussian noise
pure = input_train
pure_test = input_test
noise = np.random.normal(noise_mean, noise_std, pure.shape)  # np.random.poisson(1, pure.shape)
noise_test = np.random.normal(noise_mean, noise_std, pure_test.shape)  # np.random.poisson(1, pure_test.shape)
noisy_input = pure + noise_prop * noise
noisy_input_test = pure_test + noise_prop * noise_test

# Create the model
model = autoencoder.get_simple_autoencoder_model(img_width=img_width, img_height=img_height)
# early_stop = EarlyStopping(monitor='val_accuracy', patience=10)
model.fit(noisy_input, pure,
          epochs=no_epochs,
          batch_size=batch_size, validation_split=validation_split)

# model.save("trainedModel.h5")


# model.summary()
# model = tf.keras.models.load_model("trainedModel.h5")
# metrics = pd.DataFrame(model.history.history)
# metrics[['loss']].plot()
# metrics[['accuracy', 'val_accuracy']].plot()
# metrics[['loss', 'val_loss']].plot()

# Generate denoised images
samples = noisy_input_test[:]
denoised_images = model.predict(samples)

fig, axes = plt.subplots(number_of_visualizations, 5)
fig.set_size_inches(16, 5 * number_of_visualizations)

axes[0][0].set_title('Original image')
axes[0][1].set_title('Noisy image')
axes[0][2].set_title('Autoencoder denoised image')
axes[0][3].set_title('BM3D denoised image')
axes[0][4].set_title('NL Means denoised image')
# Plot denoised images
noisy_images = []
pure_images = []
bm3d_images = []
nl_images = []
for i in range(0, number_of_visualizations):
    # Get the sample and the reconstruction
    noisy_image = noisy_input_test[i][:, :, 0]
    pure_image = pure_test[i][:, :, 0]
    denoised_image = denoised_images[i][:, :, 0]
    bm3d_denoised = denoiser.bm3d_denoise(noisy_input_test[i].reshape(img_height, img_width))[0]
    nl_denoised = denoiser.nlm_denoise(noisy_input_test[i].reshape(img_height, img_width))[0]
    noisy_images.append(noisy_image)
    pure_images.append(pure_image)
    bm3d_images.append(bm3d_denoised)
    nl_images.append(nl_denoised)
    # Plot sample and reconstruciton
    axes[i][0].imshow(pure_image, pyplot.cm.gray)
    axes[i][0].set_ylabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image, pure_image)))
    axes[i][1].imshow(noisy_image, pyplot.cm.gray)
    axes[i][1].set_ylabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image, noisy_image)))
    axes[i][2].imshow(denoised_image, pyplot.cm.gray)
    axes[i][2].set_ylabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image, denoised_image)))
    axes[i][3].imshow(bm3d_denoised, pyplot.cm.gray)
    axes[i][3].set_ylabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image * 255.0, bm3d_denoised)))
    axes[i][4].imshow(nl_denoised, pyplot.cm.gray)
    axes[i][4].set_ylabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image * 255.0, nl_denoised)))

n1 = measure.get_set_ssim(np.array(pure_images), np.array(noisy_images), img_height, img_width)
n2 = measure.get_set_ssim(np.array(pure_images), denoised_images, img_height, img_width)
n3 = measure.get_set_ssim(np.array(pure_images) * 255.0, np.array(bm3d_images), img_height, img_width)
n4 = measure.get_set_ssim(np.array(pure_images) * 255.0, np.array(nl_images), img_height, img_width)
print("Noisy SSIM: {0}".format(n1))
print("Denoised SSIM: {0}".format(n2))
print("BM3D SSIM: {0}".format(n3))
print("NL Means SSIM: {0}".format(n4))
fig.suptitle(
    "Medical Image Denoiser\nNoise Proportion: {0} - Mean: {1} - Standard Deviation: {2}\nSSIM Results-> Noisy: {3} - "
    "Denoised: {4} - BM3D: {5} - NL Means: {6}".format(noise_prop, noise_mean, noise_std, n1, n2, n3, n4), fontsize=14,
    fontweight='bold')
# plt.savefig("output.png")
plt.show()


def save_samples(noisy_input_test, denoised_images, pure_test):
    fig, axes = plt.subplots(1, 5)
    fig.set_size_inches(20, 5)
    axes[0].set_title('Original image')
    axes[1].set_title('Noisy image')
    axes[2].set_title('Autoencoder denoised image')
    axes[3].set_title('BM3D denoised image')
    axes[4].set_title('NL Means denoised image')
    bm3d_images = []
    nl_images = []
    for i in range(0, len(noisy_input_test)):
        # Get the sample and the reconstruction
        noisy_image = noisy_input_test[i][:, :, 0]
        pure_image = pure_test[i][:, :, 0]
        denoised_image = denoised_images[i][:, :, 0]
        bm3d_denoised = denoiser.bm3d_denoise(noisy_input_test[i].reshape(64, 64))[0]
        nl_denoised = denoiser.nlm_denoise(noisy_input_test[i].reshape(64, 64))[0]
        bm3d_images.append(bm3d_denoised)
        nl_images.append(nl_denoised)
        # Plot sample and reconstruciton
        axes[0].imshow(pure_image, pyplot.cm.gray)
        axes[0].set_xlabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image, pure_image)))
        axes[1].imshow(noisy_image, pyplot.cm.gray)
        axes[1].set_xlabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image, noisy_image)))
        axes[2].imshow(denoised_image, pyplot.cm.gray)
        axes[2].set_xlabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image, denoised_image)))
        axes[3].imshow(bm3d_denoised, pyplot.cm.gray)
        axes[3].set_xlabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image * 255.0, bm3d_denoised)))
        axes[4].imshow(nl_denoised, pyplot.cm.gray)
        axes[4].set_xlabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image * 255.0, nl_denoised)))

        fig.suptitle(
            "Medical Image Denoiser\nNoise Proportion: {0} - Mean: {1} - Standard deviation: {2}".format(noise_prop, noise_mean, noise_std),
            fontsize=14, fontweight='bold')

        plt.savefig("results/DX+MIAS({1},{2},{3}) {0}.png".format(i, noise_prop, noise_mean, noise_std))

    n1 = measure.get_set_ssim(np.array(pure_images), np.array(noisy_images), img_height, img_width)
    n2 = measure.get_set_ssim(np.array(pure_images), denoised_images, img_height, img_width)
    n3 = measure.get_set_ssim(np.array(pure_images) * 255.0, np.array(bm3d_images), img_height, img_width)
    n4 = measure.get_set_ssim(np.array(pure_images) * 255.0, np.array(nl_images), img_height, img_width)

    f = open("SSIM Results.txt", "w")
    f.write("Noise Proportion: {0} - Mean: {1} - Standard Deviation: {2}\n".format(noise_prop, noise_mean, noise_std))
    f.write("Noisy SSIM:" + str(n1) + "\n")
    f.write("Denoised SSIM:" + str(n2) + "\n")
    f.write("BM3D SSIM:" + str(n3) + "\n")
    f.write("NL Means SSIM:" + str(n4) + "\n")
    f.close()

    print("Noisy SSIM: {0}".format(n1))
    print("Denoised SSIM: {0}".format(n2))
    print("BM3D SSIM: {0}".format(n3))
    print("NL Means SSIM: {0}".format(n4))


save_samples(noisy_input_test, denoised_images, pure_test)

# denoised = bm3d.bm3d_denoise(noisy_input[0].reshape(64,64))
# fig, axes = plt.subplots(1, 3)
# fig.set_size_inches(8, 3.5)
# axes[0].set_title('Pure image')
# axes[1].set_title('Noisy image')
# axes[2].set_title('Denoised image')
# axes[0].imshow(pure[0].reshape(64, 64), pyplot.cm.gray)
# axes[1].imshow(noisy_input[0].reshape(64, 64), pyplot.cm.gray)
# axes[2].imshow(denoised[0], pyplot.cm.gray)
# plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import measure
import dataset_reader
import denoiser

# Model configuration

img_width, img_height = 64, 64
batch_size = 5
no_epochs = 50
validation_split = 0.2
verbosity = 1
max_norm_value = 2.0
noise_factor = 0.2
number_of_visualizations = 4

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

# Create the modelautoencoder.get_cnn_dsc_architecture(img_width=img_width,img_height=img_height)
#model = autoencoder.get_simple_autoencoder_model(img_width=img_width, img_height=img_height)
#model.fit(noisy_input, pure,
 #         epochs=no_epochs,
 #         batch_size=batch_size)
#model.save("model")
#model.summary()
model = tf.keras.models.load_model("trainedModel.h5")


# Generate denoised images
# samples = noisy_input_test[:number_of_visualizations]
samples = noisy_input_test[:]
denoised_images = model.predict(samples)

fig, axes = plt.subplots(number_of_visualizations, 5)
fig.set_size_inches(16, 5 * number_of_visualizations)
fig.suptitle("Medical Image Denoiser", fontsize=14, fontweight='bold')
axes[0][0].set_title('Original image')
axes[0][1].set_title('Noisy image')
axes[0][2].set_title('Autoencoder denoised image')
axes[0][3].set_title('BM3D denoised image')
axes[0][4].set_title('NL Means denoised image')
# Plot denoised images
for i in range(0, number_of_visualizations):
    # Get the sample and the reconstruction
    noisy_image = noisy_input_test[i][:, :, 0]
    pure_image = pure_test[i][:, :, 0]
    denoised_image = denoised_images[i][:, :, 0]
    bm3d_denoised = denoiser.bm3d_denoise(noisy_input_test[i].reshape(64, 64))[0]
    nl_denoised = denoiser.nlm_denoise(noisy_input_test[i].reshape(64, 64))[0]

    # Plot sample and reconstruciton
    axes[i][0].imshow(pure_image, pyplot.cm.gray)
    axes[i][0].set_xlabel("PSNR: {0}".format(measure.PSNR(pure_image, pure_image)))
#    axes[i][0].set_ylabel("SSIM: {:.5f}".format(PSNR.get_ssim_result(pure_image, pure_image)))
    axes[i][1].imshow(noisy_image, pyplot.cm.gray)
    axes[i][2].imshow(denoised_image, pyplot.cm.gray)
    axes[i][2].set_xlabel("PSNR: {0}".format(measure.PSNR(pure_image, denoised_image)))
    #axes[i][2].set_ylabel("SSIM: {:.5f}".format(PSNR.get_ssim_result(pure_image, denoised_image)))
    axes[i][3].imshow(bm3d_denoised, pyplot.cm.gray)
    axes[i][3].set_xlabel("PSNR: {0}".format(measure.PSNR(pure_image, bm3d_denoised)))
    #axes[i][3].set_ylabel("SSIM: {:.5f}".format(PSNR.get_ssim_result(pure_image, bm3d_denoised)))
    axes[i][4].imshow(nl_denoised, pyplot.cm.gray)
    axes[i][4].set_xlabel("PSNR: {0}".format(measure.PSNR(pure_image, nl_denoised)))
    #axes[i][4].set_ylabel("SSIM: {:.5f}".format(PSNR.get_ssim_result(pure_image, nl_denoised)))+

plt.savefig("output.png")
plt.show()


def save_samples(noisy_input_test, denoised_images, pure_test):
    fig, axes = plt.subplots(1, 5)
    fig.set_size_inches(20, 5)
    fig.suptitle("Medical Image Denoiser", fontsize=14, fontweight='bold')
    axes[0].set_title('Original image')
    axes[1].set_title('Noisy image')
    axes[2].set_title('Autoencoder denoised image')
    axes[3].set_title('BM3D denoised image')
    axes[4].set_title('NL Means denoised image')
    bm3d_images=[]
    nl_images=[]
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
        axes[0].set_xlabel("PSNR: {0}".format(measure.PSNR(noisy_image, pure_image)))
        axes[0].set_ylabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image, pure_image)))
        axes[1].imshow(noisy_image, pyplot.cm.gray)
        axes[2].imshow(denoised_image, pyplot.cm.gray)
        axes[2].set_xlabel("PSNR: {0}".format(measure.PSNR(noisy_image, denoised_image)))
        axes[2].set_ylabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image, denoised_image)))
        axes[3].imshow(bm3d_denoised, pyplot.cm.gray)
        axes[3].set_xlabel("PSNR: {0}".format(measure.PSNR(noisy_image, bm3d_denoised)))
        axes[3].set_ylabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image, bm3d_denoised)))
        axes[4].imshow(nl_denoised, pyplot.cm.gray)
        axes[4].set_xlabel("PSNR: {0}".format(measure.PSNR(noisy_image, nl_denoised)))
        axes[2].set_ylabel("SSIM: {:.5f}".format(measure.get_image_ssim(pure_image, nl_denoised)))
        plt.savefig("results/{0}.png".format(i))
    print("Original: {0}".format(measure.get_set_ssim(pure_test, pure_test)))
    print("Noisy: {0}".format(measure.get_set_ssim(pure_test, noisy_input_test)))
    print("Denoised: {0}".format(measure.get_set_ssim(pure_test, denoised_images)))
    print("BM3D SSIM: {0}".format(measure.get_set_ssim(pure_test, np.array(bm3d_images))))
    print("NL Mean SSIM: {0}".format(measure.get_set_ssim(pure_test, np.array(nl_images))))
#save_samples(noisy_input_test, denoised_images, pure_test)



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
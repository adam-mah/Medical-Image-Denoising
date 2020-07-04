import argparse
import os

import cv2
import numpy as np
from tensorflow_core.python.keras.callbacks import EarlyStopping

import autoencoder
import dataset_reader
import samples_plt
import pandas as pd
import matplotlib.pyplot as plt

img_width, img_height = 64, 64
batch_size = 10
nu_epochs = 50
validation_split = 0.3
train_split = 0.9
verbosity = 1
noise_prop = 0.2
noise_std = 5
noise_mean = 0
number_of_samples = 4


def load_datasets(img_width=64, img_height=64):
    raw_mias = dataset_reader.read_mini_mias()  # Read mias dataset
    mias_images = np.zeros((raw_mias.shape[0], img_width, img_height))
    for i in range(raw_mias.shape[0]):
        mias_images[i] = cv2.resize(raw_mias[i], dsize=(img_width, img_height),
                                    interpolation=cv2.INTER_CUBIC)

    raw_dx = dataset_reader.read_dx()  # Read DX dataset
    dx_images = np.zeros((raw_dx.shape[0], img_width, img_width))
    for i in range(raw_dx.shape[0]):
        dx_images[i] = cv2.resize(raw_dx[i], dsize=(img_width, img_height),
                                  interpolation=cv2.INTER_CUBIC)

    # raw_dental = dataset_reader.read_dental()  # Read dental dataset
    # dental_images = np.zeros((raw_dental.shape[0], img_width, img_width))
    # for i in range(raw_dental.shape[0]):
    #     dental_images[i] = cv2.resize(raw_dental[i], dsize=(img_width, img_height),
    #                                   interpolation=cv2.INTER_CUBIC)
    #
    # rawimages3 = dataset_reader.read_covid()  # Read covid dataset
    # images3 = np.zeros((329, img_width, img_width))
    # for i in range(rawimages3.shape[0]):
    #     images3[i] = cv2.resize(rawimages3[i], dsize=(img_width, img_height),
    #                             interpolation=cv2.INTER_CUBIC)
    return mias_images, dx_images#, dental_images


def add_noise(pure, pure_test):
    noise = np.random.normal(noise_mean, noise_std, pure.shape)  # np.random.poisson(1, pure.shape)
    noise_test = np.random.normal(noise_mean, noise_std, pure_test.shape)  # np.random.poisson(1, pure_test.shape)
    noisy_input = pure + noise_prop * noise
    noisy_input_test = pure_test + noise_prop * noise_test
    return noisy_input, noisy_input_test


def model_plots(model):
    #plt.plot(model.history.history['accuracy'])
#    plt.plot(model.history.history['val_accuracy'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    # summarize history for loss
    plt.plot(model.history.history['loss'])
    #plt.plot(model.history.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train loss', 'val loss'], loc='upper left')
    #plt.show()
    #metrics = pd.DataFrame(model.history.history)
    #metrics[['loss']].plot()
    #metrics[['accuracy', 'val_accuracy']].plot()
    #metrics[['loss', 'val_loss']].plot()
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, nu_epochs), model.history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, nu_epochs), model.history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, nu_epochs), model.history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, nu_epochs), model.history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()


def train_test_split(set1, set2, train_split=0.9, shuffle_test_set=False):
    images_set = set1[:int(set1.shape[0] * train_split)]
    images_set = np.append(images_set, set2[:int(set2.shape[0] * train_split)], axis=0)
    images_set = np.append(images_set, set2[int(set2.shape[0] * train_split):], axis=0)
    images_set = np.append(images_set, set1[int(set1.shape[0] * train_split):], axis=0)
    # np.random.shuffle(images_set)

    train_size = int(set1.shape[0] * train_split + set2.shape[0] * train_split)
    input_train = images_set[0:train_size]
    input_test = images_set[train_size:]
    if shuffle_test_set:
        np.random.shuffle(input_test)
    input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
    input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
    return (input_train, input_test)


def train_test_split2(images_set, train_split=0.9, shuffle_test_set=False):
    # images_set = set1[:int(set1.shape[0] * train_split)]
    # np.random.shuffle(images_set)

    train_size = 300  # int(images_set.shape[0] * train_split)
    input_train = images_set[0:train_size]
    input_test = images_set[train_size:]
    if shuffle_test_set:
        np.random.shuffle(input_train)
        np.random.shuffle(input_test)
    input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
    input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
    return (input_train, input_test)


def train_test_split3(set1, set2, set3, train_split=0.9, shuffle_test_set=False):
    images_set = set1[:int(set1.shape[0] * train_split)]
    images_set = np.append(images_set, set2[:int(set2.shape[0] * train_split)], axis=0)
    images_set = np.append(images_set, set3[:int(set3.shape[0] * train_split)], axis=0)
    images_set = np.append(images_set, set3[int(set3.shape[0] * train_split):], axis=0)
    images_set = np.append(images_set, set2[int(set2.shape[0] * train_split):], axis=0)
    images_set = np.append(images_set, set1[int(set1.shape[0] * train_split):], axis=0)
    # np.random.shuffle(images_set)

    train_size = int(set1.shape[0] * train_split + set2.shape[0] * train_split + set3.shape[0] * train_split)
    input_train = images_set[0:train_size]
    input_test = images_set[train_size:]
    if shuffle_test_set:
        np.random.shuffle(input_test)
    input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
    input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
    return (input_train, input_test)


if __name__ == "__main__":
    mias_images, dx_images = load_datasets(img_width, img_height)  # Load mias and DX datasets
    input_train, input_test = train_test_split(mias_images, dx_images)  # Split both sets to train and test sets
    # input_train, input_test = train_test_split2(dx_images, shuffle_test_set=True)
    #input_train, input_test = train_test_split3(mias_images, dx_images, dental_images, shuffle_test_set=True)
    # Parse numbers as floats
    input_train = input_train.astype('float32')
    input_test = input_test.astype('float32')

    # Normalize data
    input_train = input_train / 255
    input_test = input_test / 255

    # Add gaussian noise
    pure = input_train
    pure_test = input_test
    noisy_input, noisy_input_test = add_noise(pure, pure_test)  # Add Gaussian noise to train and test sets

    # Create the model
    model = autoencoder.get_simple_autoencoder_model(img_width=img_width, img_height=img_height)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=15)
    model.fit(noisy_input, pure,
              epochs=nu_epochs,
              batch_size=batch_size, validation_split=validation_split, verbose=verbosity)

    test_scores = model.evaluate(noisy_input_test, pure_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    model_plots(model)
    model.save("trainedModel.h5")
    model.summary()
    # model = tf.keras.models.load_model("trainedModel.h5")

    # Generate denoised images
    samples = noisy_input_test[:]
    denoised_images = model.predict(samples)

    samples_plt.plot_samples((noise_prop, noise_std, noise_mean), noisy_input_test, denoised_images, pure_test,
                             number_of_samples, img_width=img_width, img_height=img_height)
    #samples_plt.save_samples((noise_prop, noise_std, noise_mean), noisy_input_test, denoised_images, pure_test)

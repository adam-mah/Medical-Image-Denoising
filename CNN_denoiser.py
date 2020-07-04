import matplotlib.pyplot as plt
import numpy as np
from tensorflow_core.python.keras.callbacks import EarlyStopping
import autoencoder
import samples_plt


class CNN_denoiser():
    def __init__(self, batch_size=10, nu_epochs=50, validation_split=0, img_height=64, img_width=64):
        img_width, img_height = img_height, img_width
        self.batch_size = batch_size
        self.nu_epochs = nu_epochs
        self.validation_split = validation_split
        self.model = autoencoder.get_simple_autoencoder_model(img_width=img_width, img_height=img_height)

    def model_plots(self, noise_prop, noise_mean, noise_std):
        # summarize history for loss
        plt.figure()
        plt.plot(np.arange(0, self.nu_epochs), self.model.history.history["loss"], label="train_loss")
        if self.validation_split != 0:
            plt.plot(np.arange(0, self.nu_epochs), self.model.history.history["val_loss"], label="val_loss")
        plt.title(
            "Model Loss on Dataset\nNoise Proportion: {0} - Mean: {1} - Standard Deviation: {2}".format(noise_prop,
                                                                                                        noise_mean,
                                                                                                        noise_std))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        plt.show()

    @staticmethod  # Split 2 datasets
    def train_test_split(set1, set2, train_split=0.9, shuffle_test_set=False, img_height=64, img_width=64):
        images_set = set1[:int(set1.shape[0] * train_split)]
        images_set = np.append(images_set, set2[:int(set2.shape[0] * train_split)], axis=0)
        images_set = np.append(images_set, set2[int(set2.shape[0] * train_split):], axis=0)
        images_set = np.append(images_set, set1[int(set1.shape[0] * train_split):], axis=0)

        train_size = int(set1.shape[0] * train_split + set2.shape[0] * train_split)
        input_train = images_set[0:train_size]
        input_test = images_set[train_size:]
        np.random.shuffle(input_train)  # Shuffle input train set
        if shuffle_test_set:
            np.random.shuffle(input_test)
        input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
        input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
        return input_train, input_test

    @staticmethod  # Split 1 dataset
    def train_test_split1(images_set, train_split=0.9, shuffle_test_set=False, img_height=64, img_width=64):
        train_size = int(images_set.shape[0] * train_split)
        input_train = images_set[0:train_size]
        input_test = images_set[train_size:]

        if shuffle_test_set:
            np.random.shuffle(input_test)
        input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
        input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
        return input_train, input_test

    @staticmethod  # Split 3 datasets
    def train_test_split3(set1, set2, set3, train_split=0.9, shuffle_test_set=False, img_height=64, img_width=64):
        images_set = set1[:int(set1.shape[0] * train_split)]
        images_set = np.append(images_set, set2[:int(set2.shape[0] * train_split)], axis=0)
        images_set = np.append(images_set, set3[:int(set3.shape[0] * train_split)], axis=0)
        images_set = np.append(images_set, set3[int(set3.shape[0] * train_split):], axis=0)
        images_set = np.append(images_set, set2[int(set2.shape[0] * train_split):], axis=0)
        images_set = np.append(images_set, set1[int(set1.shape[0] * train_split):], axis=0)

        train_size = int(set1.shape[0] * train_split + set2.shape[0] * train_split + set3.shape[0] * train_split)
        input_train = images_set[0:train_size]
        input_test = images_set[train_size:]
        np.random.shuffle(input_train)
        if shuffle_test_set:
            np.random.shuffle(input_test)
        input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
        input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
        return input_train, input_test

    @staticmethod  # Split 4 datasets
    def train_test_split4(set1, set2, set3, set4, train_split=0.9, shuffle_test_set=False, img_height=64, img_width=64):
        images_set = set1[:int(set1.shape[0] * train_split)]
        images_set = np.append(images_set, set2[:int(set2.shape[0] * train_split)], axis=0)
        images_set = np.append(images_set, set3[:int(set3.shape[0] * train_split)], axis=0)
        images_set = np.append(images_set, set4[:int(set4.shape[0] * train_split)], axis=0)
        images_set = np.append(images_set, set4[int(set4.shape[0] * train_split):], axis=0)
        images_set = np.append(images_set, set3[int(set3.shape[0] * train_split):], axis=0)
        images_set = np.append(images_set, set2[int(set2.shape[0] * train_split):], axis=0)
        images_set = np.append(images_set, set1[int(set1.shape[0] * train_split):], axis=0)

        train_size = int(set1.shape[0] * train_split + set2.shape[0] * train_split + set3.shape[0] * train_split
                         + set4.shape[0] * train_split)
        input_train = images_set[0:train_size]
        input_test = images_set[train_size:]
        np.random.shuffle(input_train)
        if shuffle_test_set:
            np.random.shuffle(input_test)
        input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
        input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
        return input_train, input_test

    def train(self, noisy_input, pure, save=False, verbosity=0):
        self.model.fit(noisy_input, pure,
                       epochs=self.nu_epochs,
                       batch_size=self.batch_size, validation_split=self.validation_split, verbose=verbosity)

        if save:
            self.model.save("trainedModel.h5")

    def evaluate(self, noisy_input_test, pure_test):
        test_scores = self.model.evaluate(noisy_input_test, pure_test, verbose=2)
        print("[EVALUATION] Test loss:", test_scores[0])
        print("[EVALUATION] Test accuracy:", test_scores[1])
        return test_scores

    def predict(self, samples):
        return self.model.predict(samples)


if __name__ == "__main__":
    print("Please run main.py")

import argparse

import cv2
import numpy as np

import dataset_reader
import samples_plt
from CNN_denoiser import CNN_denoiser


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
    return mias_images, dx_images  # , dental_images


def add_noise(pure, pure_test):
    noise = np.random.normal(noise_mean, noise_std, pure.shape)  # np.random.poisson(1, pure.shape)
    noise_test = np.random.normal(noise_mean, noise_std, pure_test.shape)  # np.random.poisson(1, pure_test.shape)
    noisy_input = pure + noise_prop * noise
    noisy_input_test = pure_test + noise_prop * noise_test
    return noisy_input, noisy_input_test


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end


if __name__ == "__main__":
    img_width, img_height = 64, 64
    batch_size = 10
    nu_epochs = 50
    validation_split = 0
    train_split = 0.9
    verbosity = 1
    noise_prop = 0.1
    noise_std = 1
    noise_mean = 0
    number_of_samples = 4
    shuffle_test_set = False

    parser = argparse.ArgumentParser(description='Image Denoiser')
    parser.add_argument("-load", "--load", help="Path of dataset to load [default = DX and MIAS are loaded]", type=str)
    parser.add_argument("-size", "--size", help="Image size 64x64 or 128x128 [choices = 128, 64] [default = 64]",
                        type=int,
                        choices=[128, 64])
    parser.add_argument("-p", "--proportion", help="Gaussian noise proportion [default = 0.1]", type=float)
    parser.add_argument("-std", "--sdeviation", help="Gaussian noise standard deviation [default = 1]", type=float)
    parser.add_argument("-m", "--mean", help="Gaussian noise mean [default = 0]", type=float)
    parser.add_argument("-s", "--samples", help="Number of samples [default = 4]", type=int)
    parser.add_argument("-shuffle", "--shuffle", help="Shuffle test set", action="store_true")
    parser.add_argument("-tsplit", "--trainsplit", help="Train split [0-1] [default = 0.9]", type=float,
                        choices=[Range(0.0, 1.0)])
    parser.add_argument("-epoch", "--epoch", help="Number of epochs [default = 50]", type=int)
    parser.add_argument("-batch", "--batch", help="Batch size [default = 10]", type=int)
    parser.add_argument("-vsplit", "--validationsplit", help="Validation split [0-1] [default = 0.1]", type=float,
                        choices=[Range(0.0, 1.0)])
    parser.add_argument("-save", "--save", help="Save test set samples", action="store_true")
    parser.add_argument("-plot", "--plot", help="Plot model loss", action="store_true")
    args = parser.parse_args()
    if args.proportion:
        noise_prop = args.proportion
    if args.sdeviation:
        noise_std = args.sdeviation
    if args.mean:
        noise_mean = args.mean
    if args.samples:
        number_of_samples = args.samples
    if args.epoch:
        nu_epochs = args.epoch
    if args.batch:
        batch_size = args.batch
    if args.validationsplit:
        validation_split = args.validationsplit
    if args.trainsplit:
        train_split = args.trainsplit
    if args.shuffle:
        shuffle_test_set = True
    if args.size:
        img_width = args.size
        img_height = args.size

    print("[LOG] Loading datasets...")
    if args.load:
        print("[LOG] Loading data set from [{0}]".format(args.load))
        data_images = dataset_reader.read_dataset(args.load, img_width, img_height)
        input_train, input_test = CNN_denoiser.train_test_split1(data_images, train_split=train_split,
                                                                 shuffle_test_set=shuffle_test_set, img_width=img_width,
                                                                 img_height=img_height)  # Split 1 set to train and test sets
    else:
        print("Loading default datasets, MIAS and DX")
        mias_images, dx_images = load_datasets(img_width, img_height)  # Load mias and DX datasets
        input_train, input_test = CNN_denoiser.train_test_split(mias_images, dx_images, train_split=train_split,
                                                                shuffle_test_set=shuffle_test_set, img_width=img_width,
                                                                img_height=img_height)  # Split both sets to train and test sets
    print(
        "[LOG] Load completed\n" + "[LOG] Image size {0}x{1}".format(img_width,
                                                                     img_height) + "\n[LOG] Splitting datasets with [{0}] train set size\n[LOG] Shuffle test set: {1}".format(
            train_split, shuffle_test_set))
    # input_train, input_test = CNN_denoiser.train_test_split2(dx_images, train_split=train_split, shuffle_test_set=shuffle_test_set)  # Split 1 set to train and test sets
    # input_train, input_test = train_test_split3(mias_images, dx_images, dental_images, shuffle_test_set=True)

    # Parse numbers as floats
    input_train = input_train.astype('float32')
    input_test = input_test.astype('float32')
    # Normalize data
    input_train = input_train / 255
    input_test = input_test / 255

    print("[LOG] Adding Gaussian noise to train and test sets...\nNoise Proportion: {0}\nMean: {1}\nStandard "
          "Deviation: {2}".format(noise_prop, noise_mean, noise_std))
    # Add gaussian noise
    pure = input_train
    pure_test = input_test
    noisy_input, noisy_input_test = add_noise(pure, pure_test)  # Add Gaussian noise to train and test sets

    print("[LOG] Initializing model...\nEPOCHS: {0}\nBatch size: {1}\nValidation split: {2}".format(nu_epochs,
                                                                                                    batch_size,
                                                                                                    validation_split))
    # Create the model
    cnn_denoiser = CNN_denoiser(batch_size=batch_size, nu_epochs=nu_epochs, validation_split=validation_split,
                                img_height=img_height, img_width=img_width)
    print("[LOG] Training and evaluating model...")
    cnn_denoiser.train(noisy_input, pure, verbosity=verbosity)
    #    cnn_denoiser.evaluate(noisy_input_test, pure_test)
    if args.plot:
        cnn_denoiser.model_plots(noise_prop, noise_mean, noise_std)

    # Generate denoised images
    samples = noisy_input_test[:]
    print("[LOG] Training and model evaluation completed\n[LOG] Denoising images test set...")
    denoised_images = cnn_denoiser.predict(samples)

    print("[LOG] Image denoising completed\n[LOG] Plotting denoised samples")
    samples_plt.plot_samples((noise_prop, noise_std, noise_mean), noisy_input_test, denoised_images, pure_test,
                             number_of_samples, img_width=img_width, img_height=img_height)
    if args.save:
        samples_plt.save_samples((noise_prop, noise_std, noise_mean), noisy_input_test, denoised_images, pure_test,
                                 img_width=img_width, img_height=img_height)

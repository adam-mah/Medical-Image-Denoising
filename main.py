import argparse

import autoencoder
import cnn_denoiser
import samples_plt

if __name__ == "__main__":
    noise_prop = 0.5
    noise_std = 1
    noise_mean = 0
    number_of_visualizations = 4

    parser = argparse.ArgumentParser(description='Image Denoiser')
    parser.add_argument("-p", "--proportion", help="Gaussian noise proportion", type=float)
    parser.add_argument("-std", "--sdeviation", help="Gaussian noise standard deviation", type=float)
    parser.add_argument("-m", "--mean", help="Gaussian noise mean", type=float)
    parser.add_argument("-s", "--samples", help="Number of samples", type=int)
    parser.add_argument("-sz", "--splitsize", help="Train split [0-1]", type=float, choices=range(0,1))
    parser.add_argument("-epoch", "--epoch", help="Number of epochs", type=int)
    parser.add_argument("-batch", "--batch", help="Batch size", type=int)
    parser.add_argument("-save", "--save", help="Save test set samples", action="store_true")
    args = parser.parse_args()
    if args.proportion:
        noise_prop = args.proportion
    if args.sdeviation:
        noise_std = args.sdeviation
    if args.mean:
        noise_mean = args.mean
    if args.samples:
        number_of_visualizations = args.samples
    if args.epoch:
        nu_epochs = args.epoch
    if args.batch:
        print(cnn_denoiser.batch_size)
        batch_size = args.batch
        cnn_denoiser.batch_size = args.batch
        print(cnn_denoiser.batch_size)
    if args.save:
        print(args.save)
    print(noise_prop)
    print(noise_std)
    print(noise_mean)

    mias_images, dx_images = load_datasets(img_width, img_height)  # Load mias and DX datasets
    input_train, input_test = cnn_denoiser.train_test_split(mias_images, dx_images)  # Split both sets to train and test sets
    # input_train, input_test = train_test_split2(dx_images, shuffle_test_set=True)
    # input_train, input_test = train_test_split3(mias_images, dx_images, dental_images, shuffle_test_set=True)
    # Parse numbers as floats
    input_train = input_train.astype('float32')
    input_test = input_test.astype('float32')

    # Normalize data
    input_train = input_train / 255
    input_test = input_test / 255

    # Add gaussian noise
    pure = input_train
    pure_test = input_test
    noisy_input, noisy_input_test = cnn_denoiser.add_noise(pure, pure_test)  # Add Gaussian noise to train and test sets

    # Create the model
    model = autoencoder.get_simple_autoencoder_model(img_width=img_width, img_height=img_height)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=15)
    model.fit(noisy_input, pure,
              epochs=nu_epochs,
              batch_size=batch_size, validation_split=validation_split, verbose=verbosity)

    test_scores = model.evaluate(noisy_input_test, pure_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    cnn_denoiser.model_plots(model)
    model.save("trainedModel.h5")
    model.summary()
    # model = tf.keras.models.load_model("trainedModel.h5")

    # Generate denoised images
    samples = noisy_input_test[:]
    denoised_images = model.predict(samples)

    samples_plt.plot_samples((noise_prop, noise_std, noise_mean), noisy_input_test, denoised_images, pure_test,
                             number_of_samples, img_width=img_width, img_height=img_height)
    # samples_plt.save_samples((noise_prop, noise_std, noise_mean), noisy_input_test, denoised_images, pure_test)

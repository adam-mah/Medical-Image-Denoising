import argparse

if __name__ == "__main__":
    noise_prop = 0.5
    noise_std = 1
    noise_mean = 0
    number_of_visualizations = 4

    parser = argparse.ArgumentParser(description='Image denoiser')
    parser.add_argument("-p", "--proportion", help="Gaussian noise proportion", type=float)
    parser.add_argument("-std", "--sdeviation", help="Gaussian noise standard deviation", type=float)
    parser.add_argument("-m", "--mean", help="Gaussian noise mean", type=float)
    parser.add_argument("-s", "--samples", help="Number of samples", type=int)
    args = parser.parse_args()
    if args.proportion:
        noise_prop = args.proportion
    if args.sdeviation:
        noise_std = args.sdeviation
    if args.mean:
        noise_mean = args.mean
    if args.samples:
        number_of_visualizations = args.samples
    print(noise_prop)
    print(noise_std)
    print(noise_mean)

from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
from bm3d import *
import cv2


def nlm_denoise(noisy_image):
    noisy_image = noisy_image * 255.0
    denoised = []
    sigma_est = np.mean(estimate_sigma(noisy_image, multichannel=False))
    denoised_image = denoise_nl_means(noisy_image, h=1*sigma_est)
    denoised.append(denoised_image)
    print("Image denoised using NL Means")

    return np.array(denoised)

def bm3d_denoise(noisy_image):
    noisy_image = noisy_image * 255.0
    denoised = []
    Basic_img = BM3D_1st_step(noisy_image)
    Final_img = BM3D_2nd_step(Basic_img, noisy_image)
    denoised.append(Final_img)
    print("Image denoised using BM3D")

    return numpy.array(denoised)
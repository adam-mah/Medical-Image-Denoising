import cv2
import math
import numpy
from skimage.measure._structural_similarity import compare_ssim as ssim

def PSNR(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def get_image_ssim(original_img,noisy_img):
    ssim_sum = 0
    return ssim(original_img, noisy_img,data_range=original_img.max() - noisy_img.min(), multichannel=False)
    #return 1.0*ssim_sum/originalSet.shape[0]

def get_set_ssim(originalSet,noisySet):
    ssim_sum = 0
    originalSet = originalSet.reshape(originalSet.shape[0],64, 64, 1)
    noisySet = noisySet.reshape(noisySet.shape[0],64, 64, 1)
    for i in range(originalSet.shape[0]):
        ssim_sum += ssim(originalSet[i], noisySet[i],data_range=originalSet[i].max() - noisySet[i].min(), multichannel=True)
    return 1.0*ssim_sum/originalSet.shape[0]

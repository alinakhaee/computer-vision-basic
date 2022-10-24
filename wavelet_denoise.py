import numpy as np
import matplotlib.pyplot as plt
import random
import pywt
import pywt.data
from cv2 import cv2

def salt_and_pepper_noise(image, density=0.01):
    output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < density:
                output[i][j] = 0
            elif rdn > 1 - density:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def gaussian_noise(image, mean=0, std=0.2):
    gauss = np.random.normal(mean, std, (image.shape[0], image.shape[1]))
    gauss = gauss.reshape(image.shape[0], image.shape[1])
    noisy = image + gauss * 255
    return noisy

def dwt_pyramid(image, level=1):
    coeffs = pywt.wavedec2(image, 'db1', level=level)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return arr, coeff_slices, coeffs

def compress_wavelet(image):
    coeffs = pywt.wavedec2(image, 'db1')
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    Csort = np.sort(np.abs(arr.reshape(-1)))
    for keep in [1, 0.5, 0.1, 0.05, 0.01, 0.005]:
        thresh = Csort[int(np.floor((1 - keep) * len(Csort)))]
        ind = np.abs(arr) > thresh
        Cfilt = arr * ind
        coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')
        Arecon = pywt.waverec2(coeffs_filt, wavelet='db1')
        plt.imshow(Arecon.astype('uint8'), cmap='gray'), plt.xticks([]), plt.yticks([])
        plt.title('keep: {}'.format(keep))
        plt.show()

import math
def PSNR(original, compressed):
    img1 = original.astype(np.float64)
    img2 = compressed.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

original = cv2.imread('Images/tree.jpg', 0)
noisy_image = salt_and_pepper_noise(original, 0.1)
# noisy_image = gaussian_noise(original)
plt.imshow(noisy_image, cmap='gray')
plt.show()
coeffs = dwt_pyramid(noisy_image, 2)[2]
for l in range(1, len(coeffs)):
    for i in range(coeffs[l][0].shape[0]):
        for j in range(coeffs[l][0].shape[1]):
            coeffs[l][0][i, j] = 0 if coeffs[l][0][i, j] < 10 else coeffs[l][0][i, j]
            coeffs[l][1][i, j] = 0 if coeffs[l][1][i, j] < 10 else coeffs[l][1][i, j]
            coeffs[l][2][i, j] = 0 if coeffs[l][2][i, j] < 10 else coeffs[l][2][i, j]
arr, coeff_slices = pywt.coeffs_to_array(coeffs)
plt.imshow(arr, cmap='gray')
plt.show()
print(PSNR(original, pywt.waverec2(coeffs, 'db1')[:,:653]))
print(calculate_ssim(original, pywt.waverec2(coeffs, 'db1')[:,:653]))
plt.imshow(pywt.waverec2(coeffs, 'db1'), cmap='gray')
plt.show()
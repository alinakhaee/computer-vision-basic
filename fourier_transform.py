from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def fourier_transform(img, show_spectrums=False):
    f = np.fft.fft2(img)
    f_shift = replaceZeroes(np.fft.fftshift(f))
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    phase_spectrum = np.angle(f_shift)
    if show_spectrums:
        plt.subplot(131), plt.imshow(img, cmap='gray')
        plt.title('Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(phase_spectrum, cmap='gray')
        plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()
    return f, f_shift, magnitude_spectrum, phase_spectrum


def combine_phase_magnitude(fourier1, fourier2):
    fourier_combined1 = np.multiply(np.abs(fourier1), np.exp(1j * np.angle(fourier2)))
    fourier_combined2 = np.multiply(np.abs(fourier2), np.exp(1j * np.angle(fourier1)))
    img_combined1 = np.real(np.fft.ifft2(fourier_combined1))
    img_combined2 = np.real(np.fft.ifft2(fourier_combined2))
    plt.imshow(img_combined2, cmap='gray')
    plt.xticks([]), plt.yticks([])
    # plt.subplot(121), plt.imshow(img_combined1, cmap='gray')
    # plt.title('Magnitude of 1 and Phase of 2'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(img_combined2, cmap='gray')
    # plt.title('Magnitude of 2 and Phase of 1'), plt.xticks([]), plt.yticks([])
    plt.show()


def low_pass(img, radius=10, show_diff=False, type='gaussian'):
    fshift = fourier_transform(img)[1]
    mask = np.ones((fshift.shape[0], fshift.shape[1]), dtype=np.complex128)
    for u in range(fshift.shape[0]):
        for v in range(fshift.shape[1]):
            D = math.sqrt(math.pow(u - fshift.shape[0] / 2, 2) + math.pow(v - fshift.shape[1] / 2, 2))
            if type=='gaussian':
                temp = math.pow(D, 2) / (2 * math.pow(radius, 2))
                mask[u, v] = math.pow(math.e, -temp)
            elif type=='butterworth':
                temp = math.pow(D/radius, 2*1.)
                mask[u, v] = 1 / (1 + temp)
            elif type=='ideal':
                mask[u, v] = 1 if D <= radius else 0
    f_ishift = np.fft.ifftshift(fshift * mask)
    # plt.imshow(abs(mask), cmap='gray')
    # plt.show()
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    if show_diff:
        plt.subplot(131), plt.imshow(img_back, cmap='gray')
        plt.title('Low Pass(r={})'.format(radius)), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(img - img_back, cmap='gray')
        plt.title('Spatial Diff'), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(abs(fshift - fshift * mask), cmap='gray')
        plt.title('Frequency Diff'), plt.xticks([]), plt.yticks([])
        plt.show()
    plt.imshow(img_back, cmap='gray')
    plt.title('Low Pass with Radius: {}'.format(radius)), plt.xticks([]), plt.yticks([])
    plt.show()
    return img_back, fshift * mask


img = cv2.imread('Images/Im183.BMP', 0)
img2 = cv2.imread('Images/Im184.bmp', 0)

combine_phase_magnitude(fourier_transform(img, True)[0], fourier_transform(img2, True)[0])
img = cv2.imread('Images/Im421.jpg', 0)
img2 = cv2.imread('Images/Im423.jpg', 0)
#
# fshift = fourier_transform(img, True)[1]
for i in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
    low_pass(img2, i,True, type='gaussian')

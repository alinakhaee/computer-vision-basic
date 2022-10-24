from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from fourier_transform import low_pass, fourier_transform

def gaussian_high_pass(img, radius=10):
    fshift = fourier_transform(img)[1]
    mask = np.zeros((fshift.shape[0], fshift.shape[1]), dtype=np.complex128)
    for u in range(fshift.shape[0]):
        for v in range(fshift.shape[1]):
            D = math.sqrt(math.pow(u - fshift.shape[0] / 2, 2) + math.pow(v - fshift.shape[1] / 2, 2))
            temp = math.pow(D, 2) / (2 * math.pow(radius, 2))
            mask[u, v] = 1 - math.pow(math.e, -temp)
    f_ishift = np.fft.ifftshift(fshift * mask)
    plt.imshow(abs(mask), cmap='gray')
    plt.show()
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    plt.imshow(img_back, cmap='gray')
    plt.title('High Pass with Radius: {}'.format(radius)), plt.xticks([]), plt.yticks([])
    plt.show()
    return img_back, fshift * mask


img = cv2.imread('Images/ronaldo.jpg', 0)
img2 = cv2.imread('Images/messi.jpg', 0)
high = gaussian_high_pass(img, 20)[1]
low = low_pass(img2, 9)[1]
for i in range(high.shape[0]):
    for j in range(high.shape[1]):
        low[i, j] = high[i, j] + low[i, j]

f_ishift = np.fft.ifftshift(low)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

cv2.imwrite('hi.jpg', img_back)
cv2.waitKey(0)

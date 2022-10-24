from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

def equalize_histogram(image):
    new_img = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
    hist, bins = np.histogram(image.ravel(), 256, [0, 256])  # get histogram
    p = [hist[i] / (image.shape[0] * image.shape[1]) for i in range(256)]
    s = np.zeros(shape=256)
    for i in range(256):
        s[i] = 255 * p[i] if i == 0 else s[i - 1] + 255 * p[i]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_img[i][j] = round(s[image[i][j]])
    plt.hist(image.ravel(), 256, [0, 256])
    plt.hist(new_img.ravel(), 256, [0, 256])
    print(s)
    # plt.show()
    return new_img

def get_cdf(image, channel):
    hist = [0] * 256
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i, j, channel]] += 1
    print(hist)
    p = [hist[i] / (image.shape[0] * image.shape[1]) for i in range(256)]
    s = np.zeros(shape=256)
    for i in range(256):
        s[i] = 255 * p[i] if i == 0 else s[i - 1] + 255 * p[i]
    return s

def equalize_color(image):
    new_img = np.zeros(shape=(image.shape[0], image.shape[1], 3), dtype=np.uint8)
    blue = get_cdf(image, 0)
    green = get_cdf(image, 1)
    red = get_cdf(image, 2)
    final_lut = [(blue[i] + red[i] + green[i]) / 3 for i in range(256)]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_img[i][j][0] = round(final_lut[image[i][j][0]])
            new_img[i][j][1] = round(final_lut[image[i][j][1]])
            new_img[i][j][2] = round(final_lut[image[i][j][2]])
    cv2.imshow("hi", new_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    he1 = cv2.imread('Images/tree.jpg', 1)
    # he2 = cv2.imread('Images/he2.jpg', 0)
    # he3 = cv2.imread('Images/he3.jpg', 0)
    # he4 = cv2.imread('Images/he4.jpg', 0)
    # # cv2.imwrite('equalized_he1.jpg', equalize_histogram(he1))
    # cv2.imwrite('equalized_he2.jpg', equalize_histogram(he2))
    # # cv2.imwrite('equalized_he3.jpg', equalize_histogram(he3))
    # # cv2.imwrite('equalized_he4.jpg', equalize_histogram(he4))
    # cv2.waitKey(0)
    equalize_color(he1)

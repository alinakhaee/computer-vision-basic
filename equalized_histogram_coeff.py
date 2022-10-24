from cv2 import cv2
import numpy as np
from equalize_histogram import equalize_histogram

def equalize_histogram_with_coefficient(image, real_image_coefficient):
    image_equalized = equalize_histogram(image)
    new_image = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i][j] = real_image_coefficient * image[i][j] + (1-real_image_coefficient) * image_equalized[i][j]
    return new_image


if __name__ == '__main__':
    he1 = cv2.imread('Images/he1.jpg', 0)
    he2 = cv2.imread('Images/he2.jpg', 0)
    he3 = cv2.imread('Images/he3.jpg', 0)
    he4 = cv2.imread('Images/he4.jpg', 0)
    for k in [0.1, 0.2, 0.3, 0.4, 0.5]:
        cv2.imwrite('h1'+str(k), equalize_histogram_with_coefficient(he1, k))
        cv2.imwrite('h2'+str(k), equalize_histogram_with_coefficient(he2, k))
        cv2.imwrite('h3'+str(k), equalize_histogram_with_coefficient(he3, k))
        cv2.imwrite('h4'+str(k), equalize_histogram_with_coefficient(he4, k))
    cv2.waitKey(0)

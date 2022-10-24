import numpy as np
from robert_edge_detection import blur, spatial_operation
from cv2 import cv2

def median_filter(image, k=3):
    median_image = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            in_filter = []
            for m in range(i - k // 2, i + k // 2 + 1):
                for n in range(j - k // 2, j + k // 2 + 1):
                    if 0 <= m < image.shape[0] and 0 <= n < image.shape[1]:
                        in_filter.append(image[m][n])
            median_image[i][j] = int(np.median(in_filter))
    return median_image

def sharpen_image(img, soften_img):
    details = img - soften_img
    sharped = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            t = img[i][j] + 5 * details[i][j]
            sharped[i][j] = t if t < 255 else child[i][j]
    return sharped


child = cv2.imread('Images/child.jpg', 0)
child_blured_3 = blur(child, 3)
child_blured_5 = blur(child, 5)
child_blured_7 = blur(child, 7)
gaussian = [
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]
child_blured_gaussian = spatial_operation(child, gaussian, 16)
child_blured_median = median_filter(child, 3)


cv2.imshow('blured_3', sharpen_image(child, child_blured_3))
cv2.imshow('blured_5', sharpen_image(child, child_blured_5))
cv2.imshow('blured_7', sharpen_image(child, child_blured_7))
cv2.imshow('blured_gaussian', sharpen_image(child, child_blured_gaussian))
cv2.imshow('blured_median', sharpen_image(child, child_blured_median))
cv2.waitKey(0)

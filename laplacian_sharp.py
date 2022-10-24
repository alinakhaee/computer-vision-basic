import numpy as np
from robert_edge_detection import spatial_operation
from cv2 import cv2

laplacian = [
    [-1, -2, -1],
    [-2, 12, -2],
    [-1, -2, -1]
]
laplacians =[
    [[0, 0, 0], [-2, 2, 0], [0, 0, 0]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, -2, 0], [0, 2, 0], [0, 0, 0]],
    [[0, 0, -1], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 2, -2], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [-1, 0, 0]],
    [[0, 0, 0], [0, 2, 0], [0, -2, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, -1]]
]

def adaptive_laplacian(img):
    laplacian_img = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_sum = 0
            for m in range(i - 1, i + 2):
                for n in range(j - 1, j + 2):
                    if 0 <= m < img.shape[0] and 0 <= n < img.shape[1]:
                        for l in laplacians:
                            t = l[m % 3][n % 3] * img[m][n]
                            pixel_sum += t if t < -250 else 0
            laplacian_img[i][j] = round(-pixel_sum//16) if -pixel_sum > 0 else 0
    sharped = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = int(img[i][j]) - int(laplacian_img[i][j])
            sharped[i][j] = temp if temp < 255 else img[i][j]
    return sharped

def laplacian_sharp(img):
    laplacian_img = spatial_operation(img, laplacian, 16)
    sharped = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = int(img[i][j]) + 2*int(laplacian_img[i][j])
            sharped[i][j] = temp if temp < 255 else img[i][j]
    return sharped


face1 = cv2.imread('Images/face1.jpg', 0)
face2 = cv2.imread('Images/face2.jpg', 0)
cv2.imshow('face1_l', laplacian_sharp(face1))
cv2.imshow('face2_l', laplacian_sharp(face2))
cv2.imshow('face1_al', adaptive_laplacian(face1))
cv2.imshow('face2_al', adaptive_laplacian(face2))
cv2.waitKey(0)

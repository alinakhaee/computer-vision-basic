from cv2 import cv2
import numpy as np

def transform_image(in_img, affine_matrix):
    out_img = np.zeros(shape=(round(affine_matrix[0][0]*in_img.shape[0] + affine_matrix[1][0]*in_img.shape[1] + affine_matrix[2][0]) + 1,
                              round(affine_matrix[0][1]*in_img.shape[0] + affine_matrix[1][1]*in_img.shape[1] + affine_matrix[2][1]) + 1,
                              4), dtype=np.uint8)
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            x = round(affine_matrix[0][0]*i + affine_matrix[1][0]*j + affine_matrix[2][0])
            y = round(affine_matrix[0][1]*i + affine_matrix[1][1]*j + affine_matrix[2][1])
            out_img[x][y][:3] = in_img[i][j]
            out_img[x][y][3] = 255
    return out_img

def apply_overlay(main_img, overlay_img, start_pixels):
    for i in range(overlay_img.shape[0]):
        for j in range(overlay_img.shape[1]):
            if overlay_img[i][j][3] != 0:
                main_img[i+start_pixels[0]][j+start_pixels[1]][:3] = overlay_img[i][j][:3]


# define transformation constants
TOP_AFFINE = [[0.445, 0.415, 0], [0, 0.97, 0], [0 , 0 , 1]]
LEFT_AFFINE = [[0.97, 0, 0], [0.44, 0.41, 0], [0 , 0 , 1]]
FRONT_AFFINE = [[0.97, 0, 0], [0, 0.97, 0], [0 , 0 , 1]]

# loading images
img1 = cv2.imread("Images/barbara.bmp")
img2 = cv2.imread("Images/lena.bmp")
img3 = cv2.imread("Images/girl.bmp")
cube = np.zeros(shape=(956, 964, 4), dtype=np.uint8)
cube[:, :, :3] = cv2.imread('Images/Cube.png')
cube[:, :, 3] = 255

top = transform_image(img1, TOP_AFFINE)
left = transform_image(img2, LEFT_AFFINE)
front = transform_image(img3, FRONT_AFFINE)
apply_overlay(cube, top, (117, 121))
apply_overlay(cube, left, (120, 119))
apply_overlay(cube, front, (348, 334))
cv2.imwrite("cube.png", cube)
cv2.waitKey(0)
from cv2 import cv2
import numpy as np

def spatial_operation(image, matrix, coefficient=1):
    k = len(matrix[:])
    image_edge = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_sum = 0
            for m in range(i - k//2, i + k//2 + 1):
                for n in range(j - k//2, j + k//2 + 1):
                    if 0 <= m < image.shape[0] and 0 <= n < image.shape[1]:
                        pixel_sum += matrix[m % k][n % k] * image[m][n]
            image_edge[i][j] = round(pixel_sum / coefficient) if pixel_sum > 0 else 0
    return image_edge

def robert_edge_detection(image):
    robert_matrix = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
    robert_matrix_2 = [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    r_edge = spatial_operation(image, robert_matrix)
    l_edge = spatial_operation(image, robert_matrix_2)
    sum_edge = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(r_edge.shape[0]):
        for j in range(r_edge.shape[1]):
            sum_edge[i][j] = (r_edge[i][j] // 2 + l_edge[i][j] // 2)
    return sum_edge

def blur(image, k=3):
    blur_matrix = np.ones((k, k))
    return spatial_operation(image, blur_matrix, k*k)

new_5_5 = [
    [1, 2, 3, 2, 1],
    [2, 4, 6, 4, 2],
    [3, 6, 9, 6, 3],
    [2, 4, 6, 4, 2],
    [1, 2, 3, 2, 1],
]
new_7_7 = [
    [1, 2, 3, 3, 3, 2, 1],
    [2, 4, 6, 6, 6, 4, 2],
    [3, 6, 9, 9, 9, 6, 3],
    [3, 6, 9, 9, 9, 6, 3],
    [3, 6, 9, 9, 9, 6, 3],
    [1, 2, 3, 3, 3, 2, 1],
    [2, 4, 6, 6, 6, 4, 2],
]
if __name__ == '__main__':
    mosque = cv2.imread('Images/mosque.bmp', 0)
    edge_detected = robert_edge_detection(mosque)
    edge_detected_blurred_3 = robert_edge_detection(blur(mosque, 3))
    edge_detected_blurred_5 = robert_edge_detection(blur(mosque, 5))
    edge_detected_blurred_7 = robert_edge_detection(blur(mosque, 7))
    cv2.imwrite('edge_detected_blurred_3.jpg', edge_detected_blurred_3)
    cv2.imwrite('edge_detected_blurred_5.jpg', edge_detected_blurred_5)
    cv2.imwrite('edge_detected_blurred_7.jpg', edge_detected_blurred_7)

    blurred_3 = blur(mosque, 3)
    blurred_3_twice = blur(blurred_3, 3)
    new_blurred_5 = spatial_operation(mosque, new_5_5, 81)
    blurred_3_trice = blur(blurred_3_twice, 3)
    new_blurred_7 = spatial_operation(mosque, new_7_7, 225)
    cv2.imwrite('blurred_3.jpg', blurred_3)
    cv2.imwrite('blurred_3_twice.jpg', blurred_3_twice)
    cv2.imwrite('blurred_3_trice.jpg', blurred_3_trice)
    cv2.imwrite('blurred_5_new.jpg', new_blurred_5)
    cv2.imwrite('blurred_7_new.jpg', new_blurred_7)
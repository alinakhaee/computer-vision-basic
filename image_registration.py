from cv2 import cv2
import numpy as np

# Open the image files.
cap = cv2.VideoCapture("Images/Map1.gif")  # Reference
ret, reference_img = cap.read()
cap.release()
cap = cv2.VideoCapture("Images/Map2.gif")  # Input
ret, input_img = cap.read()
cap.release()

reference_tie_points = np.array([124, 184, 214, 247, 150, 379, 321, 157])
input_tie_points = np.array([94, 121, 170, 186, 83, 310, 290, 126])
A = np.array([[1, input_tie_points[0], input_tie_points[1], input_tie_points[0] * input_tie_points[1], 0, 0, 0, 0],
              [0, 0, 0, 0, 1, input_tie_points[0], input_tie_points[1], input_tie_points[0] * input_tie_points[1]],
              [1, input_tie_points[2], input_tie_points[3], input_tie_points[2] * input_tie_points[3], 0, 0, 0, 0],
              [0, 0, 0, 0, 1, input_tie_points[2], input_tie_points[3], input_tie_points[2] * input_tie_points[3]],
              [1, input_tie_points[4], input_tie_points[5], input_tie_points[4] * input_tie_points[5], 0, 0, 0, 0],
              [0, 0, 0, 0, 1, input_tie_points[4], input_tie_points[5], input_tie_points[4] * input_tie_points[5]],
              [1, input_tie_points[6], input_tie_points[7], input_tie_points[6] * input_tie_points[7], 0, 0, 0, 0],
              [0, 0, 0, 0, 1, input_tie_points[6], input_tie_points[7], input_tie_points[6] * input_tie_points[7]]
              ])
X2 = np.linalg.solve(A, reference_tie_points)
print(X2)
x = X2[0] + X2[1] * input_img.shape[0] + X2[2] * input_img.shape[1] + X2[3] * input_img.shape[0] * input_img.shape[1]
y = X2[4] + X2[5] * input_img.shape[0] + X2[6] * input_img.shape[1] + X2[7] * input_img.shape[0] * input_img.shape[1]
output_img = np.zeros(shape=(reference_img.shape[0], reference_img.shape[1], 3), dtype=np.uint8)

for i in range(input_img.shape[0]):
    for j in range(input_img.shape[1]):
        x = X2[0] + X2[1] * i + X2[2] * j + X2[3] * i * j
        y = X2[4] + X2[5] * i + X2[6] * j + X2[7] * i * j
        x = round(x)
        y = round(y)
        if x >= reference_img.shape[0] or y >= reference_img.shape[1]:
            continue
        reference_img[x][y] = input_img[i][j]
        output_img[x][y] = input_img[i][j]

for i in range(output_img.shape[0]):
    for j in range(output_img.shape[1]):
        if np.all(output_img[i][j] == 0):
            if i != 0 and i != output_img.shape[0] - 1 and np.any(output_img[i - 1][j] != 0) and np.any(
                    output_img[i + 1][j] != 0):
                output_img[i][j] = output_img[i - 1][j]
            if j != 0 and j != output_img.shape[1] - 1 and np.any(output_img[i][j - 1] != 0) and np.any(
                    output_img[i][j + 1] != 0):
                output_img[i][j] = output_img[i][j - 1]

for i in range(output_img.shape[0]):
    for j in range(output_img.shape[1]):
        if np.any(output_img[i][j] != 0):
            reference_img[i][j] = output_img[i][j]

cv2.imwrite("registrated.png", output_img)
cv2.waitKey(0)

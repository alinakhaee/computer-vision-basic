import cv2
import matplotlib.pyplot as plt
import numpy as np

def find_match_points_sift(img1, img2, show_result=False):
    MIN_MATCH_COUNT = 10
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) >= 60:
        print(len(good))
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w, color = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # Use homography
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        return None

    if show_result:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        cv2.imwrite("c.jpg", img3)
        plt.imshow(img3, ), plt.show()
    return src_pts, dst_pts


def find_corner_points_harris(img1, show_result=False):
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 10, 3, 0.04)
    corners_x, corners_y = np.where(dst > 0.01 * dst.max())
    corner_points = list(zip(corners_x, corners_y))
    if show_result:
        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)
        # Threshold for an optimal value, it may vary depending on the image.
        img1[dst > 0.1 * dst.max()] = [0, 0, 255]
        cv2.imwrite('corner_points.jpg', img1)
    return corner_points

def overlay_transparent(background, overlay, x=0, y=0):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


video = cv2.VideoCapture('project/Test/1.MOV')
# video = cv2.VideoCapture('project/Test/0.MOV')
video2 = cv2.VideoCapture('project/Angels and Demons.mp4')
# video2 = cv2.VideoCapture('project/David Copperfield.mp4')
success, image1 = video.read()
success2, image3 = video2.read()
image2 = cv2.imread('project/Angels and Demons.jpg')
# image2 = cv2.imread('project/David Copperfield.jpg')
image_list = []
count = 0
M_prev = []

while success and success2:
    success, image1 = video.read()
    success2, image3 = video2.read()
    if not success or not success2:
        break

    try:
        src_pts, dst_pts = find_match_points_sift(image1, image2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        M_prev = M
    except:
        print("using previous")
        M = M_prev
    height, width, channels = image1.shape
    image3 = cv2.resize(image3, (image2.shape[0], image2.shape[1]), interpolation=cv2.INTER_AREA)
    image3 = cv2.rotate(image3, cv2.cv2.ROTATE_90_CLOCKWISE)
    im1Reg = cv2.warpPerspective(image3, M, (width, height), borderValue=(255,255,255))
    alpha = np.sum(im1Reg, axis=-1) < 255+255+255
    alpha = np.uint8(alpha * 255)
    res = np.dstack((im1Reg, alpha))
    final = overlay_transparent(image1, res)
    cv2.imwrite("overlay.jpg", final)
    image_list.append(final)
    count += 1
    print("frame end", count)

video.release()

codec = cv2.VideoWriter_fourcc("D", "I", "V", "X")
frame_rate = 15
resolution = (image_list[0].shape[1], image_list[0].shape[0])
video2 = cv2.VideoWriter("projects3.avi", cv2.VideoWriter_fourcc(*'DIVX'), 15, resolution)

for img in image_list:
    video2.write(img)

cv2.destroyAllWindows()
video2.release()
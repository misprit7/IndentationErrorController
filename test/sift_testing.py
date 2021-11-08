import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import imutils

image = cv2.imread('test_image_1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
thresh = cv2.inRange(hsv, (100, 0, 150), (130, 12, 200))

# edged = cv2.Canny(thresh, 0, 1)

# cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# peri = cv2.arcLength(cnts[0], True)
# approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)

# cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

# kernel = np.ones((2,2), np.uint8)
# img_erosion = cv2.erode(frame_threshold, kernel, iterations=1)
# img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

# plt.imshow(frame_threshold)
# plt.show()

template_img = cv2.imread('plate_0.png')

sift = cv2.xfeatures2d.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(template_img, None)
# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

kp_grayframe, desc_grayframe = sift.detectAndCompute(gray, None)
matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
good_points = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_points.append(m)
img3 = cv2.drawMatches(template_img, kp_image, gray, kp_grayframe, good_points, gray)

# if len(good_points) > 4:
#     query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
#     train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
#     matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
#     matches_mask = mask.ravel().tolist()
#     # Perspective transform
#     h, w, _ = template_img.shape
#     pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
#     dst = cv2.perspectiveTransform(pts, matrix)
#     homography = cv2.polylines(image, [np.int32(dst)], True, (255, 0, 0), 3)
#     # cv2.imshow("Homography", homography)
#     img3 = homography
# else:
#     # cv2.imshow("Homography", grayframe)
#     img3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# pixmap = self.convert_cv_to_pixmap(img3)
# self.live_image_label.setPixmap(pixmap)

cv2.imshow('SIFT', img3)
cv2.waitKey(0)
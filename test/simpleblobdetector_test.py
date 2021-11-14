import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import imutils

image = cv2.imread('license-threshold.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
thresh = cv2.inRange(hsv, (130, 1, 250), (135, 5, 255))

# edged = cv2.Canny(thresh, 0, 1)

# cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# peri = cv2.arcLength(cnts[0], True)
# approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)

# cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

kernel = np.ones((2,2), np.uint8)
img_erosion = cv2.erode(thresh, kernel, iterations=1)
img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

# plt.imshow(frame_threshold)
# plt.show()

detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(thresh)
im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("", thresh)
cv2.waitKey(0)
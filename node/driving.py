import cv2
import numpy as np

# Get centroid of road from image
# hsv: image to get centroid for
# returns: cx, cy where cx and cy are centroids
def getCentroid(hsv):
    frame_threshold = cv2.inRange(hsv, (0, 0, 80), (10, 10, 90))

    height, width = frame_threshold.shape

    M = cv2.moments(frame_threshold)
    if M["m00"] == 0:
        cX = width / 2
        cY = height / 2
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    return cX, cY

def getCX(cnt, width):
    M = cv2.moments(cnt)

    if M["m00"] == 0:
        cX = 0
    else:
        cX = int(M["m10"] / M["m00"])
    return cX

def getCY(cnt, width):
    M = cv2.moments(cnt)

    if M["m00"] == 0:
        cY = 4 * width / 5
    else:
        cY = int(M["m01"] / M["m00"])
    return cY

def getRightLine(hsv):
    frame_threshold = cv2.inRange(hsv, (0, 0, 235), (108, 255, 255))

    height, width = frame_threshold.shape

    _,contours,hierarchy = cv2.findContours(frame_threshold, 1, 2)

    cX = 4 * width / 5
    cY = 4 * width / 5

    if len(contours) > 0:
        right_line = max(contours, key=lambda cnt : getCX(cnt, width))
        cX = getCX(right_line, width)

    return cX, cY

def getLeftLine(hsv):
    frame_threshold = cv2.inRange(hsv, (0, 0, 235), (108, 255, 255))

    height, width = frame_threshold.shape

    _,contours,hierarchy = cv2.findContours(frame_threshold[:,:width/2], 1, 2)

    cX = 1 * width / 5
    cY = 4 * width / 5

    if len(contours) > 0:
        sort = sorted(contours, key=lambda cnt : getCX(cnt, width), reverse=True)
        left_line = max(sort, key=lambda cnt : getCX(cnt, width))

        cX = getCX(left_line, width)

    return cX, cY

# Calculates pid from error
# err: Error of prediction
def pidCalc(err):
    wP = 2.0
    wI = 1.0
    wD = 1.0

    p = -err * wP
    i=0
    d=0
    return p + i + d

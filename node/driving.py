import cv2
import numpy as np

# Get centroid of road from image
# hsv: image to get centroid for
# returns: cx, cy where cx and cy are centroids
def getCentroid(hsv):
    frame_threshold = cv2.inRange(hsv, (0, 0, 80), (10, 10, 90))

    height, width = frame_threshold.shape
    frame_threshold = frame_threshold[:, int(2*width/9):]
    # cv2.imshow("thresh", frame_threshold)

    M = cv2.moments(frame_threshold)
    if M["m00"] == 0:
        cX = width / 2
        cY = height / 2
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    return cX + 2*width/9, cY

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
        
        # cv2.imshow('Left Line', cv2.drawContours(hsv, [right_line], 0, (255, 0, 0), 3))

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
        # cv2.imshow('Left Line', cv2.drawContours(hsv, [left_line], 0, (255, 0, 0), 3))

        cX = getCX(left_line, width)



    return cX, cY

# Calculates pid from error
# err: Error of prediction
def pidCalc(err, wP, wI, wD):

    p = -err * wP
    i=0
    d=0
    return p + i + d

# Checks if car is in center of image
def checkForCar(hsv):
    height, width, _ = hsv.shape
    threshold = cv2.inRange(hsv[height/4:3*height/4, width/4:], (0, 0, 45), (0, 0, 79))
    kernel = np.ones((2, 2), np.uint8)
    thresh_erode = cv2.erode(threshold, kernel, iterations=1)
    cv2.imshow("Threshold", thresh_erode)
    return np.count_nonzero(thresh_erode) > 100

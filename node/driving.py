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

# Calculates pid from error
# err: Error of prediction
def pidCalc(err):
    wP = 1.0
    wI = 1.0
    wD = 1.0

    p = -err * wP
    i=0
    d=0
    return p + i + d

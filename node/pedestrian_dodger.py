import cv2
import numpy as np

def get_bottom_red(hsv):
    red_threshold = cv2.inRange(hsv, (0, 182, 99), (0, 255, 255))
    x, y = np.nonzero(red_threshold)
    if len(y) == 0:
        return 0
    bottom_red = x[-1]
    # cv2.imshow('Red Mask', red_threshold)
    return bottom_red

bg_subtract = cv2.bgsegm.createBackgroundSubtractorMOG()
def is_movement(img):
    # Effectively a static variable
    global bg_subtract
    mask = bg_subtract.apply(img)

    kernel = np.ones((3, 3), np.uint8)
    mask_erosion = cv2.erode(mask, kernel, iterations=2)

    # cv2.imshow('Background Subtraction', mask_erosion)
    return np.any(mask_erosion)


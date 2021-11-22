import cv2
import numpy as np

def get_bottom_red(hsv):
    red_threshold = cv2.inRange(hsv, (0, 182, 99), (0, 255, 255))
    x, y = np.nonzero(red_threshold)
    if len(y) == 0:
        return 0
    bottom_red = y[-1]
    return bottom_red

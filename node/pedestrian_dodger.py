import cv2
import numpy as np

def get_bottom_red(hsv, height):
    red_threshold = cv2.inRange(hsv[height-25:height,:], (0, 182, 99), (0, 255, 255))
    x, y = np.nonzero(red_threshold)
    if len(y) == 0:
        return 0
    bottom_red = x[-1]
    # cv2.imshow('Red Mask', red_threshold)
    return bottom_red

bg_subtract = cv2.bgsegm.createBackgroundSubtractorMOG()
last_center_x = 0
def is_movement(img):
    # Effectively a static variable
    global bg_subtract
    global last_center_x
    img = img[:, img.shape[1]/5:img.shape[1]*4/5]
    mask = bg_subtract.apply(img)

    # cv2.imshow('Background Subtraction', mask)

    count = np.count_nonzero(mask)
    print(count)
    # If too much or too little on screen return
    if count > 1700 or count < 400:
        return True
    
    # Get centroids
    m = cv2.moments(mask)
    if m['m00'] == 0:
        return True
    # Check if change from left to right
    center_x = m['m10'] / m['m00']
    print(center_x, last_center_x, np.abs(center_x - last_center_x) < 60, max(center_x, last_center_x) > img.shape[1] / 2, min(center_x, last_center_x) < img.shape[1] / 2)
    if np.abs(center_x - last_center_x) < 50 and max(center_x, last_center_x) > img.shape[1] / 2 and min(center_x, last_center_x) < img.shape[1] / 2:
        return False
    else:
        last_center_x = center_x
        return True

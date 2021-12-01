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
is_on_left = None
last_center_x = 0
def is_movement(img):
    global is_on_left
    # Effectively a static variable
    global bg_subtract
    global last_center_x
    img = img[:, img.shape[1]/5:img.shape[1]*4/5]
    mask = bg_subtract.apply(img)

    # cv2.imshow('Background Subtraction', mask)

    count = np.count_nonzero(mask)
    print(count)
    if count > 1500 or count < 400:
        return True
    
    m = cv2.moments(mask)
    if m['m00'] == 0:
        return True

    center_x = m['m10'] / m['m00']
    print(center_x, last_center_x, np.abs(center_x - last_center_x) < 60, max(center_x, last_center_x) > img.shape[1] / 2, min(center_x, last_center_x) < img.shape[1] / 2)
    if np.abs(center_x - last_center_x) < 50 and max(center_x, last_center_x) > img.shape[1] / 2 and min(center_x, last_center_x) < img.shape[1] / 2:
        return False
    else:
        last_center_x = center_x
        return True

    # print(is_on_left, m['m10'] / m['m00'])
    # if is_on_left is None:
    #     if m['m10'] / m['m00'] < img.shape[1] / 2:
    #         is_on_left = True
    #     else:
    #         is_on_left = False


    # if m['m10'] / m['m00'] < img.shape[1] / 2:
    #     if is_on_left == False:
    #         return False
    # else:
    #     if is_on_left == True:
    #         return False
    
    # return True

    # kernel = np.ones((2, 2), np.uint8)
    # mask_erosion = cv2.erode(mask, kernel, iterations=2)

    # cv2.imshow('Background Subtraction', mask_erosion)
    # return np.count_nonzero(mask_erosion) > 5


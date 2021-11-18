import cv2
import numpy as np

def comp_h(tup):
    return tup[1]

def comp_x(tup):
    return tup[0]

def assemble_box(top_box, bottom_box):
    top_line = sorted(top_box, key = comp_h, reverse = True)[:2]
    bottom_line = sorted(bottom_box, key = comp_h, reverse = False)[:2]

    top_line = sorted(top_line, key = comp_x, reverse = False)
    bottom_line = sorted(bottom_line, key = comp_x, reverse = True)

    pts = np.concatenate([top_line, bottom_line])

    return pts



# Parses image
# image: image of the road to parse
# width: width of returned image
# height: height of returned image
# returns: warped perspective image of plate, None if none found
def plate_parse(image, width, height):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, (0, 0, 100), (255, 0, 125))

    kernel = np.ones((3, 3), np.uint8)
    img_erosion = cv2.erode(thresh, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

    _,contours,hierarchy = cv2.findContours(img_dilation, 1, 2)
    if len(contours) < 2:
        return None

    top_cnt = max(contours, key = cv2.contourArea)
    contours.remove(top_cnt)
    bottom_cnt = max(contours, key = cv2.contourArea)

    if cv2.contourArea(top_cnt) < 2000:
        return None

    top_box = cv2.boxPoints(cv2.minAreaRect(top_cnt))
    bottom_box = cv2.boxPoints(cv2.minAreaRect(bottom_cnt))


    pts = assemble_box(top_box, bottom_box)


    # pts = np.concatenate([top_box[2:4], bottom_box[0:2]])


    # is box good????? Check

    if abs(top_box[0][0] - bottom_box[1][0]) > 30:
        return None

    if abs(top_box[1][0] - bottom_box[0][0]) > 30:
        return None

    # pts = np.int0(pts)
    # return cv2.drawContours(image, [pts], 0, (0,255,0), 3)

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (width, height))

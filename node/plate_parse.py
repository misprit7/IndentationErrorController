import cv2
import numpy as np
# import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import h5py
import numpy

from keras.utils import CustomObjectScope

from keras.initializers import glorot_uniform

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    json_file = open('/home/fizzer/ros_ws/src/indentation_error_controller/cnn_models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/home/fizzer/ros_ws/src/indentation_error_controller/cnn_models/model.h5")

# json_file = open('/home/fizzer/ros_ws/src/indentation_error_controller/cnn_models/model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = keras.models.model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("/home/fizzer/ros_ws/src/indentation_error_controller/cnn_models/model.h5")

# loaded_model = keras.models.load_model('/home/fizzer/ros_ws/src/indentation_error_controller/cnn_models/model.h5')

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

def parse(license):
    predicted = ""
  
    for i in [40, 145, 350, 455]:
        piece = np.array(license.crop((i, 100, i + 105, 250)))
        piece_aug = np.expand_dims(piece, axis=0)
        prediction = conv_model.predict(piece_aug)[0]
        predicted += decode(prediction)

    return predicted


# Parses image
# image: image of the road to parse
# width: width of returned image
# height: height of returned image
# returns: warped perspective image of plate, None if none found
def plate_parse(image, width, height):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #thresh = cv2.inRange(hsv, (0, 0, 100), (0, 255, 205))
    
    thresh1 = cv2.inRange(hsv, (0, 0, 100), (0, 255, 125))
    thresh2 = cv2.inRange(hsv, (0, 0, 180), (0, 255, 205))

    thresh = max([thresh1, thresh2], key = cv2.countNonZero)

    kernel = np.ones((3, 3), np.uint8)
    img_erosion = cv2.erode(thresh, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

    _,contours,hierarchy = cv2.findContours(img_dilation, 1, 2)
    if len(contours) < 2:
        return None

    top_cnt_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    top_cnt = contours[top_cnt_index]
    if cv2.contourArea(top_cnt) < 2000 or cv2.contourArea(top_cnt) > 8000:
        return None
    contours.pop(top_cnt_index)

    bottom_cnt = max(contours, key = cv2.contourArea)

    ratio = cv2.contourArea(top_cnt)/cv2.contourArea(bottom_cnt)

    if ratio < 5 or ratio > 6:
        return None

    top_box = cv2.boxPoints(cv2.minAreaRect(top_cnt))
    bottom_box = cv2.boxPoints(cv2.minAreaRect(bottom_cnt))


    pts = assemble_box(top_box, bottom_box)


    # pts = np.concatenate([top_box[2:4], bottom_box[0:2]])


    # is box good????? Check

    # if abs(top_box[0][0] - bottom_box[1][0]) > 30:
    #     return None

    # if abs(top_box[1][0] - bottom_box[0][0]) > 30:
    #     return None

    # pts = np.int0(pts)
    # return cv2.drawContours(image, [pts], 0, (0,255,0), 3)

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    # return cv2.warpPerspective(image, M, (width, height))
    license = cv2.warpPerspective(image, M, (width, height))
    return parse(license)

import cv2
import numpy as np
import numpy


# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.models import model_from_json
import h5py

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model


sess1 = tf.Session()    
graph1 = tf.get_default_graph()
set_session(sess1)

# load json
json_file = open('/home/fizzer/ros_ws/src/indentation_error_controller/cnn_models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/home/fizzer/ros_ws/src/indentation_error_controller/cnn_models/model.h5")
# loaded_model = models.load_model("/home/fizzer/ros_ws/src/indentation_error_controller/cnn_models/model.h5")

encoder = {}
[encoder.update({chr(i):i-65}) for i in range(65, 91)]
[encoder.update({chr(i):i-22}) for i in range(48, 58)]


def comp_h(tup):
    return tup[1]

def comp_x(tup):
    return tup[0]

def assemble_box(top_box, bottom_box):
    top_line = sorted(top_box, key = comp_h, reverse = False)[:2]
    bottom_line = sorted(bottom_box, key = comp_h, reverse = True)[:2]

    top_line = sorted(top_line, key = comp_x, reverse = False)
    bottom_line = sorted(bottom_line, key = comp_x, reverse = True)

    pts = np.concatenate([top_line, bottom_line])

    return pts

def license_parse(license):
    global loaded_model
    global sess1
    global graph1
    predicted = []
    license = license[1250:1551]
    cv2.imshow("license", license)

    with graph1.as_default():
        set_session(sess1)
        for i in [40, 145, 350, 455]:
            piece = license[100:250, i:i+105]
            piece = cv2.cvtColor(piece, cv2.COLOR_RGB2BGR)
            piece_aug = np.expand_dims(piece, axis=0)
            prediction = loaded_model.predict(piece_aug)[0]
            predicted += decode(prediction)

    return predicted

def parking_parse(parking_stall):
    parking_stall = parking_stall[700:1151, 300:]
    cv2.imshow('Parking Stall', parking_stall)
    # cv2.imwrite('/home/fizzer/ros_ws/src/indentation_error_controller/test/parking_num.png', parking_stall)

def decode(encoded):
    encoder_keys = list(encoder.keys())
    return encoder_keys[np.argmax(encoded)]

# Parses image
# image: image of the road to parse
# width: width of returned image
# height: height of returned image
# returns: warped perspective image of plate, None if none found
def plate_parse(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    width = 600
    height = 1800
    
    thresh1 = cv2.inRange(hsv, (0, 0, 100), (0, 255, 125))
    thresh2 = cv2.inRange(hsv, (0, 0, 180), (0, 255, 205))


    thresh = max([thresh1, thresh2], key = cv2.countNonZero)

    kernel = np.ones((3, 3), np.uint8)
    img_erosion = cv2.erode(thresh, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    # cv2.imshow("dilate", img_dilation)

    _,contours,hierarchy = cv2.findContours(img_dilation, 1, 2)
    if len(contours) < 2:
        return (None, None)

    top_cnt_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    top_cnt = contours[top_cnt_index]
    if cv2.contourArea(top_cnt) < 2000 or cv2.contourArea(top_cnt) > 8000:
        return (None, None)
    contours.pop(top_cnt_index)

    bottom_cnt = max(contours, key = cv2.contourArea)

    ratio = cv2.contourArea(top_cnt)/cv2.contourArea(bottom_cnt)

    if ratio < 5 or ratio > 6:
        return (None, None)

    top_box = cv2.boxPoints(cv2.minAreaRect(top_cnt))
    bottom_box = cv2.boxPoints(cv2.minAreaRect(bottom_cnt))


    pts = assemble_box(top_box, bottom_box)




    height = 1800
    width = 600

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    # return cv2.warpPerspective(image, M, (width, height))
    license = cv2.warpPerspective(image, M, (width, height))


    return parking_parse(license), license_parse(license)

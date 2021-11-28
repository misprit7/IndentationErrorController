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
    top_line = sorted(top_box, key = comp_h, reverse = True)[:2]
    bottom_line = sorted(bottom_box, key = comp_h, reverse = False)[:2]

    top_line = sorted(top_line, key = comp_x, reverse = False)
    bottom_line = sorted(bottom_line, key = comp_x, reverse = True)

    pts = np.concatenate([top_line, bottom_line])

    return pts

def license_parse(license):
    global loaded_model
    global sess1
    global graph1
    predicted = []
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
    print(parking_stall.shape)
    num = parking_stall[210:370, 175:350]
    cv2.imshow('Parking Stall', num)

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


    license_height = 300
    license_width = 600

    license_dst = np.array([
        [0, 0],
        [license_width - 1, 0],
        [license_width - 1, license_height - 1],
        [0, license_height - 1]], dtype = "float32")

    M1 = cv2.getPerspectiveTransform(pts, license_dst)
    # return cv2.warpPerspective(image, M, (width, height))
    license = cv2.warpPerspective(image, M1, (license_width, license_height))


    parking_height = 400
    parking_width = 350

    dst = np.array([
        [0, 0],
        [parking_width - 1, 0],
        [parking_width - 1, parking_height - 1],
        [0, parking_height - 1]], dtype = "float32")

    M2 = cv2.getPerspectiveTransform(assemble_box(top_box, top_box)[::-1], dst)
    parking_stall = cv2.warpPerspective(image, M2, (parking_width, parking_height))
    return parking_parse(parking_stall), license_parse(license)

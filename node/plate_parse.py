import cv2
import numpy as np
import numpy


import h5py

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model


sess1 = tf.Session()    
graph1 = tf.get_default_graph()
set_session(sess1)

# load json
from tensorflow import keras
model_n = keras.models.load_model("/home/fizzer/ros_ws/src/indentation_error_controller/cnn_models/model-ideal-nums.h5")
model_l = keras.models.load_model("/home/fizzer/ros_ws/src/indentation_error_controller/cnn_models/model-ideal-chars.h5")

encoder_l = {}
encoder_n = {}
[encoder_l.update({i-65:chr(i)}) for i in range(65, 91)]
[encoder_n.update({i-48:chr(i)}) for i in range(48, 58)]
print(encoder_l)
print(encoder_n)


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

def make_brighter(hsv):
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*2
    hsv[:,:,1][hsv[:,:,1]>255] = 255
    hsv[:,:,2] = hsv[:,:,2]*2
    hsv[:,:,2][hsv[:,:,2]>255] = 255
    return np.array(hsv, dtype=np.uint8)

def get_digits(license):

    # license_hsv = cv2.cvtColor(license, cv2.COLOR_BGR2HSV)
    license = make_brighter(license)

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(license, -1, sharpen_kernel)
    # temp = cv2.cvtColor(license, cv2.COLOR_HSV2BGR)
    # cv2.imshow('sharpened', license)
    # cv2.imwrite('/home/fizzer/ros_ws/src/indentation_error_controller/test/license_bright.png', temp)

    # thresh = cv2.inRange(license_hsv, (106, 105, 84), (127, 255, 255))
    # sharpen = np.array(sharpen, dtype=np.uint8)
    sharpen = cv2.cvtColor(sharpen, cv2.COLOR_BGR2GRAY)
    # _,thresh = cv2.threshold(sharpen, 135, 150, cv2.THRESH_BINARY_INV)
    # _,thresh = cv2.threshold(sharpen, 100, 155, cv2.THRESH_BINARY_INV)
    # P1,2,3,6
    # _,thresh1 = cv2.threshold(sharpen, 90, 100, cv2.THRESH_BINARY_INV)
    # _,thresh2 = cv2.threshold(sharpen, 160, 250, cv2.THRESH_BINARY_INV)

    # thresh = min([thresh1, thresh2], key = cv2.countNonZero)
    _,thresh = cv2.threshold(sharpen,50,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    img_erosion = cv2.erode(thresh, kernel, iterations=1)

    # _,thresh = cv2.threshold(sharpen,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # print(thresh.shape)
    # thresh = np.array(thresh, dtype=np)
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    thresh = cv2.GaussianBlur(thresh,(5,5),cv2.BORDER_DEFAULT)
    cv2.imshow("thresh", thresh)

    pieces = []
    pieces_x = []
    for i, cnt in enumerate(contours[0:4]):
        # cv2.drawContours(license, np.int0([cnt]), 0, (0, 255, 0), 3)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(license,(x,y),(x+w,y+h),(0,255,0),2)
        piece = cv2.resize(thresh[y:y+h, x:x+w], (105, 150), interpolation = cv2.INTER_AREA)
        pieces.append(piece)
        pieces_x.append(x)

    cv2.imshow('sharpened', license)


    pieces = [x for _,x in sorted(zip(pieces_x, pieces), key=lambda x : x[0])]

    predict = ""
    for i, piece in enumerate(pieces[:2]):
        # cv2.imshow('pieces' + str(i), piece)
        predict += read_char(piece)

    for i, piece in enumerate(pieces[2:]):
        # cv2.imshow('pieces' + str(i), piece)
        predict += read_num(piece)

    return predict

def read_char(char):
    with graph1.as_default():
        set_session(sess1)
        char = char[..., np.newaxis]
        piece_aug = np.expand_dims(char, axis=0)
        prediction = model_l.predict(piece_aug)[0]
        return decode_l(prediction)

def read_num(num):
    with graph1.as_default():
        set_session(sess1)
        num = num[..., np.newaxis]
        piece_aug = np.expand_dims(num, axis=0)
        prediction = model_n.predict(piece_aug)[0]
        return decode_n(prediction)


def license_parse(license):
    global model
    global sess1
    global graph1
    predicted = []
    license = license[1250:1551]

    predicted = get_digits(license)


    # with graph1.as_default():
    #     set_session(sess1)
    #     for piece in pieces:
    #         piece = cv2.cvtColor(piece, cv2.COLOR_RGB2BGR)
    #         piece_aug = np.expand_dims(piece, axis=0)
    #         prediction = model.predict(piece_aug)[0]
    #         predicted += decode(prediction)

    return predicted

def parking_parse(parking_stall):
    parking_stall = parking_stall[700:1151, 300:]
    parking_stall = np.expand_dims(parking_stall, axis=0)
    _,thresh = cv2.threshold(parking_stall,50,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    x,y,w,h = cv2.boundingRect(contours[0])
    piece = cv2.resize(thresh[y:y+h, x:x+w], (105, 150), interpolation = cv2.INTER_AREA)

    return read_num(piece)



    # cv2.imshow('Parking Stall', parking_stall)
    # cv2.imwrite('/home/fizzer/ros_ws/src/indentation_error_controller/test/parking_num.png', parking_stall)

def decode_n(encoded):
    global encoder_n
    return encoder_n[np.argmax(encoded)]

def decode_l(encoded):
    global encoder_l
    return encoder_l[np.argmax(encoded)]

# Parses image
# image: image of the road to parse
# width: width of returned image
# height: height of returned image
# returns: warped perspective image of plate, None if none found
def plate_parse(image, hsv):
    width = 600
    height = 1800
    
    thresh1 = cv2.inRange(hsv, (0, 0, 100), (0, 255, 125))
    thresh2 = cv2.inRange(hsv, (0, 0, 180), (0, 255, 205))


    # thresh = max([thresh1, thresh2], key = cv2.countNonZero)
    thresh = thresh1 | thresh2
    area = np.count_nonzero(thresh)

    if np.count_nonzero(thresh) < 6000 or area > 40000:
        print("Not enough area / Too much area")
        return (None, None)

    kernel = np.ones((2, 2), np.uint8)
    img_erosion = cv2.erode(thresh, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

    # cv2.imshow('contours', img_dilation)

    _,contours,hierarchy = cv2.findContours(img_dilation, 1, 2)
    if len(contours) < 2:
        print("Not enough contours")
        return (None, None)

    top_cnt_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    top_cnt = contours[top_cnt_index]
    # if cv2.contourArea(top_cnt) < 1000 or cv2.contourArea(top_cnt) > 100000:
    #     print('Contours outside of range')
    #     return (None, None)
    contours.pop(top_cnt_index)

    bottom_cnt = max(contours, key = cv2.contourArea)


    # cv2.drawContours(image, np.int0([top_cnt]), 0, (0, 255, 0), 3)
    # cv2.drawContours(image, np.int0([bottom_cnt]), 0, (0, 0, 255), 3)

    ratio = cv2.contourArea(top_cnt)/cv2.contourArea(bottom_cnt)

    # if ratio < 4 or ratio > 7:
    #     print("Wrong contour ratio")
    #     return (None, None)

    top_box = cv2.boxPoints(cv2.minAreaRect(top_cnt))
    bottom_box = cv2.boxPoints(cv2.minAreaRect(bottom_cnt))


    pts = assemble_box(top_box, bottom_box)

    if pts[0][0] < 5:
        print("Cut off")
        return (None, None)


    # cv2.drawContours(image, np.int0([pts]), 0, (0, 255, 0), 3)


    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    # return cv2.warpPerspective(image, M, (width, height))
    license = cv2.warpPerspective(image, M, (width, height))


    return parking_parse(license), license_parse(license)

#! /usr/bin/env python

import rospy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

import os
import thread


# from plate_parse import plate_parse

bridge = CvBridge()

# Init node
rospy.init_node('indentation_error_controller')
rate = rospy.Rate(2)

move_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=1)

def show_image(img):
    cv2.imshow("Image Window", img)
    cv2.waitKey(3)

def contour_centroid_X(cnt, width):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
      cX = 0
    else:
      cX = int(M["m10"] / M["m00"])
    return cX

def contour_centroid_Y(cnt, width):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
      cY = 0
    else:
      cY = int(M["m01"] / M["m00"])
    return cY


def image_callback(img_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    # hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    # frame_threshold = cv2.inRange(hsv, (0, 0, 80), (10, 10, 90))

    # height, width = frame_threshold.shape

    # M = cv2.moments(frame_threshold)
    # if M["m00"] == 0:
    #   cX = width / 2
    #   cY = height / 2
    # else:
    #   cX = int(M["m10"] / M["m00"])
    #   cY = int(M["m01"] / M["m00"])

    # cv2.circle(cv_image, (cX, cY), 5, [255, 255, 255], -1)

    
    # wP = 1
    # wI = 1
    # wD = 1

    # p = -(2.0 * cX / width - 1) * wP

    # pid = p

    # move = Twist()
    # move.angular.z = p
    # move.linear.x = 0.1

    # move_pub.publish(move)


    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(hsv, (0, 0, 235), (108, 255, 255))

    height, width = frame_threshold.shape

    _,contours,hierarchy = cv2.findContours(frame_threshold[:,:width/2], 1, 2)
    if len(contours) > 0:
      sort = sorted(contours, key=lambda cnt: contour_centroid_X(cnt,width), reverse=True)
      sort = sorted(sort, key=lambda cnt: contour_centroid_Y(cnt,width), reverse=True)
      # left_line = min(contours, key=lambda cnt: contour_centroid_X(cnt,width))
      # for s in sort:
      #   if contour_centroid_X(s,width) >= width/2:
      #     sort.remove(s)
      left_line = sort[0]

      M = cv2.moments(left_line)
      if M["m00"] == 0:
        cX = 1 * width / 5
        cY = 1 * height / 5
      else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
      # print(width)
      cv2.circle(cv_image, (cX, cY), 5, [0, 0, 255], -1)
      cv2.drawContours(cv_image, [left_line], -1, (0,255,0), 3)
    
    wP = 2
    wI = 1
    wD = 1

    p = -(2.0 * (cX - 1 * width / 5) / width) * wP

    pid = p

    move = Twist()
    move.angular.z = p
    move.linear.x = 0.1

    move_pub.publish(move)

    show_image(cv_image)









    # plate_img = plate_parse(cv_image, 600, 300)

    # if not (plate_img is None):
    #   show_image(plate_img)


    #   cv2.imwrite('/home/fizzer/Pictures/plate_image.png',plate_img)

    # thresh1 = cv2.inRange(hsv, (0, 0, 100), (0, 255, 125))
    # thresh2 = cv2.inRange(hsv, (0, 0, 180), (0, 255, 205))
    # show_image(thresh1)
    # show_image(thresh2)

    # plate = plate_parse(cv_image, 600, 300)
    # print(plate)


image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,image_callback)
velocity_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=1)


while not rospy.is_shutdown():
    rate.sleep()

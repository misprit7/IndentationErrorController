#! /usr/bin/env python

import rospy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

import os
import thread


from plate_parse import plate_parse
from pedestrian_dodger import get_bottom_red
from driving import getCentroid, pidCalc

from enum import Enum

class State(Enum):
  STARTUP = 1
  OUTSIDE_LOOP = 2
  INSIDE_LOOP = 3
  PEDESTRIAN_STOP = 4
  PEDESTRIAN_RUN = 5

state = State.STARTUP

bridge = CvBridge()

# Init node
rospy.init_node('indentation_error_controller')
rate = rospy.Rate(2)

move_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=1)

# Shows an image
# img: an image in opencv format in BGR formatting
def show_image(img):
    cv2.imshow("Image Window", img)
    cv2.waitKey(3)

# Publishes move with specified values
# linear: linear speed to move
# angular: angular speed to move
def move(linear, angular):
    move = Twist()
    move.angular.z = angular
    move.linear.x = linear

    move_pub.publish(move)

def image_callback(img_msg):
    global state
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    height, width, _ = hsv.shape

    if state == State.STARTUP:
        state = State.OUTSIDE_LOOP
    if state == State.OUTSIDE_LOOP:
        # Get Centroid
        cX, cY = getCentroid(hsv)
        cv2.circle(cv_image, (cX, cY), 5, [255, 255, 255], -1)

        # Check for red line
        bottom_red = get_bottom_red(hsv)
        if bottom_red > hsv.shape[1] - 100:
            move(0, 0)
            state = State.PEDESTRIAN_STOP
            print('stopped')
            return


        pid = pidCalc(2.0 * cX / width - 1)
        move(0.1, pid)

    elif state == State.PEDESTRIAN_STOP:
        pass
    elif state == State.PEDESTRIAN_RUN:
        pass
    show_image(cv_image)



    # plate_img = plate_parse(cv_image, 600, 300)

    # if not (plate_img is None):
    #   show_image(plate_img)


    #   cv2.imwrite('/home/fizzer/Pictures/plate_image.png',plate_img)

    # thresh1 = cv2.inRange(hsv, (0, 0, 100), (0, 255, 125))
    # thresh2 = cv2.inRange(hsv, (0, 0, 180), (0, 255, 205))
    # show_image(thresh1)
    # show_image(thresh2)

    plate = plate_parse(cv_image, 600, 300)


image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,image_callback)
velocity_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=1)


while not rospy.is_shutdown():
    rate.sleep()

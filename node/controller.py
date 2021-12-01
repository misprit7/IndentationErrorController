#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import os
import thread
from enum import Enum


from plate_parse import plate_parse
from pedestrian_dodger import get_bottom_red, is_movement
from driving import getRightLine, getLeftLine, pidCalc, checkForCar, getCentroid

from std_msgs.msg import String

class State(Enum):
  STARTUP = 1
  OUTSIDE_LOOP = 2
  INSIDE_LOOP = 3
  PEDESTRIAN_STOP = 4
  PEDESTRIAN_RUN = 5
  TURN_INTO_LOOP = 6
  LOOK_AROUND = 7
  INITIAL_TURN = 8

state = State.STARTUP

bridge = CvBridge()

# Init node
rospy.init_node('indentation_error_controller')
rate = rospy.Rate(2)

move_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=1)

plate_pub = rospy.Publisher('/license_plate', String,
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

# Changes State, use this instead of directly setting state
# to keep track of state change
# destState: State to change to
def state_change(destState):
    global state
    state = destState
    print(state)

pedestrian_no_move_counter = 0
timer = 0
angle = 0
plates = ['']*8
lastCar = 0
def image_callback(img_msg):
    global state
    global timer
    global pedestrian_no_move_counter
    global plates
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    height, width, _ = hsv.shape

    if state == State.STARTUP:
        plate_pub.publish(str("IndError,naderson,0,XR58"))

        state_change(State.OUTSIDE_LOOP)

        # state_change(State.INITIAL_TURN)
        move(0, 0)
        timer = rospy.get_time()

    elif state == State.INITIAL_TURN:
        move(0.2, 0.5)
        if rospy.get_time() - timer > 1:
            state_change(State.OUTSIDE_LOOP)

    elif state == State.OUTSIDE_LOOP:
        # Get Centroid of right side white line
        cX, cY = getRightLine(hsv, height, width)
        # cv2.circle(cv_image, (cX, cY), 5, [0, 255, 0], -1)

        # Check for red line
        bottom_red = get_bottom_red(hsv, height)
        # if bottom_red > height - 100:
        if bottom_red > 0:
            move(0, 0)
            state_change(State.PEDESTRIAN_STOP)
            return


        pid = pidCalc(2.0 * (cX - 4 * width / 5) / width, 4.0, 1.0, -5000.0)
        # if abs(pid) >= 0.4:
        #     move(0.0, pid)
        # else:
        #     move(0.3, pid)
        # move(0, 0)
        move(0.3, pid)

        parking_num, plate = plate_parse(cv_image, hsv)
        if plate != None:
            print(plate, " at parking stall ", parking_num)
            plate_pub.publish(str('IndError,naderson,{0},{1}'.format(parking_num, plate)))

        # if True:
        #     timer = rospy.get_time()
        #     print(timer)
        #     state_change(State.TURN_INTO_LOOP)

    elif state == State.TURN_INTO_LOOP:
        global lastCar
        print(rospy.get_time() - lastCar)
        carInFront = checkForCar(hsv)
        if rospy.get_time() - lastCar < 1 or carInFront:
            # Wait until car passes
            print("Waiting for car...")
            move(0, 0)
            timer = rospy.get_time()
            if carInFront:
                lastCar = rospy.get_time()
        else:
            global angle
            cX, cY = getRightLine(hsv[2*height/3:, :width/2])


            cv2.circle(cv_image, (cX, cY), 5, [0, 255, 0], -1)

            pid = pidCalc(2.0 * (cX - 1 * width / 5) / width, 4.0, 1.0, 1.0)
            
            move(0.3, pid)
            angle += pid * (rospy.get_time() - timer)
            # print(angle)
            timer = rospy.get_time()
            if angle > 3.5:
                state_change(State.INSIDE_LOOP)
                # Sets the timer negative so first loop triggers turning
                timer = -20

    elif state == State.INSIDE_LOOP:
        cX, cY = getCentroid(hsv)
        cv2.circle(cv_image, (cX, cY), 5, [0, 255, 0], -1)

        pid = pidCalc(2.0 * (cX - width / 2) / width, 3.0, 1.0, 1.0)
        # move(0, 0)
        print(pid)
        move(0.3, pid)

        # print(plate)

        if np.abs(pid) > 0.6 and rospy.get_time() - timer > 7:
            state_change(State.LOOK_AROUND)
            timer = rospy.get_time()

    elif state == State.LOOK_AROUND:
        if rospy.get_time() - timer > 2:
            state_change(State.INSIDE_LOOP)
            timer = rospy.get_time()
        elif rospy.get_time() - timer > 1.25:
            move(0, 1)
        elif rospy.get_time() - timer > 0.75:
            move(0, 0)
        else: 
            move(0, -1)

        parking_num, plate = plate_parse(cv_image, hsv)


    elif state == State.PEDESTRIAN_STOP:
        # if not is_movement(cv_image):
        if not is_movement(cv_image):
            if pedestrian_no_move_counter > 5:
                state_change(State.PEDESTRIAN_RUN)
                timer = rospy.get_time()
                pedestrian_no_move_counter = 0 
            else:
                pedestrian_no_move_counter += 1
    elif state == State.PEDESTRIAN_RUN:
        if rospy.get_time() - timer < 1:
            move(0.5, 0)
        else:
            move(0.2, 0)
            state_change(State.OUTSIDE_LOOP)


    show_image(cv_image)



rospy.sleep(0.1)
image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,image_callback)
velocity_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=1)

while not rospy.is_shutdown():
    rate.sleep()

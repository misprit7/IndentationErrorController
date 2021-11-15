#! /usr/bin/env python

import rospy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import String

bridge = CvBridge()

# Init node
rospy.init_node('indentation_error_controller')
rate = rospy.Rate(2)


move_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=1)

plate_pub = rospy.Publisher('/license_plate', String,
  queue_size=1)

def show_image(img):
    cv2.imshow("Image Window", img)
    cv2.waitKey(3)

def image_callback(img_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(hsv, (0, 0, 80), (10, 10, 90))

    height, width = frame_threshold.shape

    M = cv2.moments(frame_threshold)
    if M["m00"] == 0:
      cX = width / 2
      cY = height / 2
    else:
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])

    cv2.circle(cv_image, (cX, cY), 5, [255, 255, 255], -1)

    
    wP = 1
    wI = 1
    wD = 1

    p = -(2.0 * cX / width - 1) * wP

    pid = p

    move = Twist()
    move.angular.z = p
    move.linear.x = 0.1

    move_pub.publish(move)

image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,image_callback)
velocity_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=1)

plate_pub.publish("TeamRed,multi12,0,XR58")
print("starting")


# while not rospy.is_shutdown():
#     rate.sleep()

for i in range(15):
    rate.sleep()
    print("sleeping 2")


print("closing")
plate_pub.publish("TeamRed,multi12,-1,XR58")
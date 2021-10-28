#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

from sensor_msgs.msg import Image

rospy.init_node('indentation_error_controller')
rate = rospy.Rate(2)
move = Twist()
move.linear.x = 0.1
move.angular.z = 0.5
pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=1)

def image_callback(img_msg):
    pub.publish(move)

sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,image_callback)


while not rospy.is_shutdown():
    rate.sleep()

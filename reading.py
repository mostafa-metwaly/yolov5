#!/usr/bin/env python
import numpy as np
import math
import rospy
import csv
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose
from std_msgs.msg import String , Float32MultiArray , Float32

def talker():

	pub = rospy.Publisher('/desired_pose', PoseStamped, queue_size=100)
	rospy.init_node('camera_pose', anonymous=True)
	rate = rospy.Rate(10) # 10hz
	text = open('/home/ComputerVision-Project/yolov5/bo7sen.txt', 'r')
	for line in text:
	    t = line.split(" ")
	    X,Y,Z = t[0],t[1],t[2]
	print(X)
	text.close()


	pose_goal = geometry_msgs.msg.PoseStamped() #geometry_msgs.msg.Pose()

	pose_goal.position.x = X
	pose_goal.position.y = Y
	pose_goal.position.z = Z

#################### determining transformation matrix of object w.r.t camera ################################


            
################### transformation matrix of camera w.r.t base_link ################################

        # T_bc = np.array([
        #     [  6.12323400e-17 , -6.12323400e-17  , 1.00000000e+00 , 1.59],
        #     [  1.00000000e+00 , 3.74939946e-33   , 6.12323400e-17 , 0.23],
        #     [  0.00000000e+00 , -1.00000000e+00  , 6.12323400e-17 , 0.44],
        #     [       0         ,         0        ,        0       ,  1  ]
        #     ])

        # print "T_bc Value:\n",T_bc
#################### calculating transformation matrix of object w.r.t base_link ################################

        # T_bo = np.dot(T_bc , T_co)
        # print "T_bo Value:\n",T_bo

#################### publish object pose w.r.t base_link ################################

        # euler_angles = rotationMatrixToEulerAngles(T_bo)
        # fpose = [T_bo[0][3] , T_bo[1][3] , T_bo[2][3] , euler_angles[0] , euler_angles[1] , euler_angles[2]]
        # print "fpose Value:\n",fpose
        # #print (lpose[1])
        # position.data = fpose
    print(pose_goal)
    #rospy.loginfo(row)
    pub.publish(pose_goal)
    rate.sleep()

if __name__ == '__main__':
    try:
        m = talker()
        while not rospy.is_shutdown():
            connections = m.pub.get_num_connections()
            rospy.loginfo('Connections: %d', connections)
            if connections > 0 :
                m.run()
                rospy.loginfo('Published')
                break
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
import rosbag
import sys
import os
import numpy as np
import glob

from robot_model import Model, InverseKinematicsParameters, QPInverseVelocityParameters
import state_representation as sr
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

def process_user_rosbags(user_num='1'):

	# Set up data path
	data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data"

	# Set up user dir
	#subject_dir = os.path.join(data_dir, "User_"+user_num)
	subject_dir = os.path.join(data_dir, "example_traj_to_replay")
	rosbag_dir = os.path.join(subject_dir, "rosbags")
	csv_dir = os.path.join(subject_dir, "csv")

	# import panda urdf
	urdf_path = "/home/ros/ros_ws/src/learning_safety_margin/urdf/panda_arm.urdf"
	robot = Model("franka", urdf_path)

	# Verify directory
	listOfBagFiles = glob.glob(rosbag_dir + '/*.bag', recursive=True)

	# Loop for all bags in 'User_X' folder
	numberOfFiles = str(len(listOfBagFiles))
	print("Reading all " + numberOfFiles + " bagfiles in "+rosbag_dir+" directory: \n")

	count = 0
	countSafe = 0
	countUnsafe = 0
	countDaring = 0
	for bagFile in listOfBagFiles:

		count +=1
		print("Reading file " + str(count) + " of  " + str(numberOfFiles) + ": " + bagFile)

		# current structure safe, unsafe, daring directories with trajNumber_eePosition.txt

		# Set csv file path depending on safety label
		if "/safe/" in bagFile:
			countSafe += 1
			save_dir = os.path.join(csv_dir, "safe", str(countSafe))
		if "/unsafe/" in bagFile:
			countUnsafe += 1
			save_dir = os.path.join(csv_dir, "unsafe", str(countUnsafe))
		if "/daring/" in bagFile:
			countDaring += 1
			save_dir = os.path.join(csv_dir, "daring", str(countDaring))
		else:
			save_dir = os.path.join(csv_dir, str(count))

		eePositions = []
		eeVelocities = []
		jointPositions = []
		jointVelocities = []
		temp_jointState = sr.JointState("franka", robot.get_joint_frames())

		# access bag
		bag = rosbag.Bag(bagFile)

		# s = JointState()
		# s.position = 'foo'
		#
		# bag.write('joint_states', s)
		# print(s.position)
		# DEBUG PRINT
		# frames = robot.get_frames()
		# base_frame = robot.get_base_frame()
		# joint_frames = robot.get_joint_frames()
		# print("Frames : ", frames ,"\n")
		# print(" BASE Frames : ", base_frame, "\n")
		# print(" JOINT Frames : ", joint_frames, "\n")


		for topic, msg, t in bag.read_messages():

			# positions
			# must convert to sr to compute forward kinematics
			# print(topic)

			jointPos = sr.JointPositions("franka", np.array(msg.position))
			eePos = robot.forward_kinematics(jointPos, 'panda_link8')  # outputs cartesian pose (7d)
			# convert back from sr to save to list
			eePositions.append(eePos.data()) # only saving transitional pos, not orientation

			# velocities
			# must convert to sr Joint state to compute forward velocities
			temp_jointState.set_velocities( np.array(msg.velocity))
			temp_jointState.set_positions( np.array(msg.position))
			eeVel = robot.forward_velocity(temp_jointState, 'panda_link8')  # outputs cartesian twist (6d)
			# convert back from sr to save to list
			eeVelocities.append(eeVel.data())  # only saving transitional vel, not orientation

			# Joint State
			jointPositions.append(msg.position)#temp_jointState.get_positions())
			jointVelocities.append(msg.velocity)


		# Reshape lists
		pose2save = np.array(eePositions)
		twist2save = np.array(eeVelocities)
		jointPos2save = np.array(jointPositions)
		jointVel2save = np.array(jointVelocities)
		print("Saving file " + str(count) + " of  " + str(numberOfFiles) + ": " + save_dir+"_eePosition.txt")
		np.savetxt(save_dir+"_eePosition.txt", pose2save, delimiter=",")
		np.savetxt(save_dir+"_eeVelocity.txt", twist2save, delimiter=",")
		np.savetxt(save_dir + "_jointPositions.txt", jointPos2save, delimiter=",")
		np.savetxt(save_dir + "_jointVelocities.txt", jointVel2save, delimiter=",")

		bag.close()

	print("Done reading all " + str(numberOfFiles) + " bag files.")


if __name__ == '__main__':

	if len(sys.argv) >= 2:
		user_number = sys.argv[1]
	else:
		user_number = 0

	if isinstance(user_number, str):
		print("Processing rosbags for User_"+user_number)
		process_user_rosbags(user_number)
	else:
		print("Processing rosbags for User_1 \n")
		print("To process other user, provide user number as sole argument: python3 bag2csv.py 2")
		process_user_rosbags()


import rosbag
import sys
import os
import numpy as np
import glob

from robot_model import Model, InverseKinematicsParameters, QPInverseVelocityParameters
import state_representation as sr


def process_user_rosbags(user_num='1'):

	# Set up data path
	data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data"

	# Set up user dir
	subject_dir = os.path.join(data_dir, "User_"+user_num)
	rosbag_dir = os.path.join(subject_dir, "rosbags")
	csv_dir = os.path.join(subject_dir, "csv")

	# import panda urdf
	urdf_path = "/home/ros/ros_ws/src/learning_safety_margin/urdf/panda_arm.urdf"
	robot = Model("franka", urdf_path)

	# Verify directory
	listOfBagFiles = glob.glob(rosbag_dir + '/**/*.bag', recursive=True)

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
		if "safe" in bagFile:
			countSafe += 1
			save_dir = os.path.join(csv_dir, "safe", str(countSafe))
		if "unsafe" in bagFile:
			countUnsafe += 1
			save_dir = os.path.join(csv_dir, "unsafe", str(countUnsafe))
		if "daring" in bagFile:
			countDaring += 1
			save_dir = os.path.join(csv_dir, "daring", str(countDaring))

		eePositions = []
		eeVelocities = []
		temp_jointState = sr.JointState("franka", robot.get_joint_frames())

		# access bag
		bag = rosbag.Bag(bagFile)

		for topic, msg, t in bag.read_messages():

			# positions
			# must convert to sr to compute forward kinematics
			jointPos = sr.JointPositions("franka", msg.position)
			eePos = robot.forward_kinematics(jointPos)  # outputs cartesian pose (7d)
			# convert back from sr to save to list
			eePositions.append(eePos.data()[:3]) # only saving transitional pos, not orientation

			# velocities
			# must convert to sr Joint state to compute forward velocities
			temp_jointState.set_velocities(msg.velocity)
			temp_jointState.set_positions(msg.position)
			eeVel = robot.forward_velocity(temp_jointState)  # outputs cartesian twist (6d)
			# convert back from sr to save to list
			eeVelocities.append(eeVel.data()[:3])  # only saving transitional vel, not orientation

		# Reshape lists
		pose2save = np.reshape(np.array(eePositions), (3, -1))
		twist2save = np.reshape(np.array(eeVelocities), (3, -1))
		print("Saving file " + str(count) + " of  " + str(numberOfFiles) + ": " + save_dir+"_eePosition.txt")
		np.savetxt(save_dir+"_eePosition.txt", pose2save, delimiter=",")
		np.savetxt(save_dir+"_eeVelocity.txt", twist2save, delimiter=",")

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


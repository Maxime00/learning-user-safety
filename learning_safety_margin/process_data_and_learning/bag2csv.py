import pandas as pd
import rosbag
import sys
import os
import numpy as np
import glob
import pickle
import time

from robot_model import Model, InverseKinematicsParameters, QPInverseVelocityParameters
import state_representation as sr
from scipy import signal
import matplotlib.pyplot as plt
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


def find_closest_time(df, time):
    dist = (df['time'] - time).abs()
    return df.loc[dist.idxmin()]


def process_user_rosbags(user_num='0'):

	# Set up data path
	data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data"

	# Set up user dir
	subject_dir = os.path.join(data_dir, "User_"+user_num)
	# subject_dir = os.path.join(data_dir, "example_traj_to_replay")
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
		if "/safe/" in bagFile:
			countSafe += 1
			save_dir = os.path.join(csv_dir, "safe", str(countSafe))
		elif "/unsafe/" in bagFile:
			countUnsafe += 1
			save_dir = os.path.join(csv_dir, "unsafe", str(countUnsafe))
		elif "/daring/" in bagFile:
			countDaring += 1
			save_dir = os.path.join(csv_dir, "daring", str(countDaring))
		else:
			save_dir = os.path.join(csv_dir, str(count))

		eePositions = []
		eeVelocities = []
		jointPositions = []
		jointVelocities = []
		jointTorques = []
		time_idx = []
		temp_jointState = sr.JointState("franka", robot.get_joint_frames())

		# access bag
		bag = rosbag.Bag(bagFile)

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
			jointPositions.append(np.array(msg.position))#temp_jointState.get_positions())
			jointVelocities.append(np.array(msg.velocity))
			jointTorques.append(np.array(msg.effort))
			time_idx.append(t.to_sec())


		# Reshape lists
		pose2save = np.array(eePositions)
		twist2save = np.array(eeVelocities)

		jointVel2save = np.array(jointVelocities)
		jointTorq2save = np.array(jointTorques)
		# make time relative to traj
		time_idx = np.array(time_idx)
		time_idx = time_idx - time_idx[0]


		# Filter velocities and torques
		smoothTorques = np.zeros(jointTorq2save.shape)
		smoothVel = np.zeros(jointVel2save.shape)
		for i in range(0,len(jointTorq2save[0,:])):
			smoothTorques[:,i] = signal.savgol_filter(x=jointTorq2save[:,i], window_length=100, polyorder = 3)
			smoothVel[:, i] = signal.savgol_filter(x=jointVel2save[:, i], window_length=100, polyorder=3)

		# plot velocities
		# plt.figure()
		# plt.plot(time_idx, jointVel2save)
		# plt.title("raw")
		#
		# # plot velocities smooth
		# plt.figure()
		# plt.plot(time_idx, smoothVel)
		# plt.title("smooth")
		# plt.show()


		# convert to pandas
		traj_dict = {'time' : time_idx ,'position': jointPositions, 'velocity': smoothVel.tolist(), 'torques': smoothTorques.tolist()}
		trajectory_df = pd.DataFrame.from_dict(data=traj_dict)

		# Save to file
		print("Saving file " + str(count) + " of  " + str(numberOfFiles) + ": " + save_dir+"_eePosition.txt")
		np.savetxt(save_dir+"_eePosition.txt", pose2save, delimiter=",")
		np.savetxt(save_dir+"_eeVelocity.txt", twist2save, delimiter=",")
		trajectory_df.to_pickle(path = save_dir+"_jointState.pkl")

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


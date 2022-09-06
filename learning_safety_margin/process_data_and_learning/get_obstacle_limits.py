#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
import time
import rosbag

from bag2csv import process_user_rosbags
from robot_model import Model, InverseKinematicsParameters, QPInverseVelocityParameters
import state_representation as sr

def read_bag(bagFile):
    # import panda urdf
    urdf_path = "/home/ros/ros_ws/src/learning_safety_margin/urdf/panda_arm.urdf"
    robot = Model("franka", urdf_path)

    eePositions = []
    eeVelocities = []
    jointPositions = []
    jointVelocities = []
    jointTorques = []
    time_idx = []
    temp_jointState = sr.JointState("franka", robot.get_joint_frames())

    bag = rosbag.Bag(bagFile)
    for topic, msg, t in bag.read_messages():
        # positions
        # must convert to sr to compute forward kinematics
        # print(topic)
        jointPos = sr.JointPositions("franka", np.array(msg.position))
        eePos = robot.forward_kinematics(jointPos, 'panda_link8')  # outputs cartesian pose (7d)
        # convert back from sr to save to list
        eePositions.append(eePos.data())  # only saving transitional pos, not orientation

        # velocities
        # must convert to sr Joint state to compute forward velocities
        temp_jointState.set_velocities(np.array(msg.velocity))
        temp_jointState.set_positions(np.array(msg.position))
        eeVel = robot.forward_velocity(temp_jointState, 'panda_link8')  # outputs cartesian twist (6d)
        # convert back from sr to save to list
        eeVelocities.append(eeVel.data())  # only saving transitional vel, not orientation

        # Joint State
        jointPositions.append(np.array(msg.position))  # temp_jointState.get_positions())
        jointVelocities.append(np.array(msg.velocity))
        jointTorques.append(np.array(msg.effort))
        time_idx.append(t.to_sec())

    # Reshape lists
    pose2save = np.array(eePositions)
    twist2save = np.array(eeVelocities)

    return pose2save, twist2save


def get_obstacle_limits(bag="/home/ros/ros_ws/src/learning_safety_margin/data/obstacle-boundaries.bag"):
    pos, _ = read_bag(bag)
    print(pos.shape)
    obs_upper_limits = np.amax(pos, axis=0)
    obs_lower_limits = np.amin(pos, axis=0)
    obs_limits = np.vstack((obs_lower_limits, obs_upper_limits)).T
    print("Obstacle Limits: {}, \n Lower Limits: {} \n Upper Limits: {}".format(obs_limits, obs_lower_limits, obs_upper_limits))
    return obs_limits

def get_obstacle_limits(bag="/home/ros/ros_ws/src/learning_safety_margin/data/obstacle-boundaries.bag"):
    pos, _ = read_bag(bag)
    print(pos.shape)
    obs_upper_limits = np.amax(pos, axis=0)
    obs_lower_limits = np.amin(pos, axis=0)
    obs_limits = np.vstack((obs_lower_limits, obs_upper_limits)).T
    print("Obstacle Limits: {}, \n Lower Limits: {} \n Upper Limits: {}".format(obs_limits, obs_lower_limits, obs_upper_limits))
    return obs_limits
def parser():
    # Parse arguments calls functions accordingly

    parser = argparse.ArgumentParser(description="Process rosbags into csv and gets limits")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", default=True)
    group.add_argument("-q", "--quiet", action="store_true", default = False)
    parser.add_argument("-p", "--plot", action="store_true", default=False)
    parser.add_argument("-s", "--to_smooth", type=str, choices=['0', '1'], default='1', nargs='?', help="option to smooth cartesian velocities")

    args = parser.parse_args()

    get_obstacle_limits()


if __name__ == "__main__":
    parser()

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


def set_up_dir(user_number):
    # Check directories and create them if needed
    data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_"+user_number+"/csv"

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if not os.path.isdir(os.path.join(data_dir, "daring")):
        os.mkdir(os.path.join(data_dir, "daring"))

    if not os.path.isdir(os.path.join(data_dir, "safe")):
        os.mkdir(os.path.join(data_dir, "safe"))

    if not os.path.isdir(os.path.join(data_dir, "unsafe")):
        os.mkdir(os.path.join(data_dir, "unsafe"))

def bags_need_processing(user_number):

    data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_"+user_number

    count_csv = sum(len(files) for _, _, files in os.walk(data_dir + "/csv"))
    count_rosbags= sum(len(files) for _, _, files in os.walk(data_dir + "/rosbags"))

    if count_csv >= 3*count_rosbags:
        return False
    elif count_csv < 3*count_rosbags:
        return True

def get_user_limits(user_number):
    # benchmark
    start_time = time.time()

    # Set up data path
    data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_" + user_number + "/"
    csv_dir = data_dir + 'csv/'
    rosbag_dir = data_dir + "rosbags/"
    fig_path = '../franka_env/figures/vel_lim/'
    save = False

    nSafe = len(os.listdir(rosbag_dir + "safe"))
    nUnsafe = len(os.listdir(rosbag_dir + "unsafe"))
    nDaring = len(os.listdir(rosbag_dir + "daring"))
    print("# Safe Demos: {}, # Unsafe Demos: {}, # Daring Demos: {}".format(nSafe, nUnsafe, nDaring))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    safe_traj = []
    for i in range(0, nSafe):
        fname = csv_dir + 'safe/' + str(i + 1) + '_eePosition.txt'
        if os.path.exists(fname):
            pos = np.loadtxt(fname, delimiter=',')[:,0:3]
            safe_traj.append(pos)
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'g')
        else:
            print("Safe Demo {} File Path does not exist: {}".format(i, fname))

    unsafe_traj = []
    tte_list = []
    for i in range(0, nUnsafe):
        fname = csv_dir + 'unsafe/' + str(i + 1) + '_eePosition.txt'
        if os.path.exists(fname):
            pos = np.loadtxt(fname, delimiter=',')[:,0:3]
            tte = np.expand_dims(np.ones(pos.shape[0]), axis=1)
            for j in range(pos.shape[0]):
                tte[j] = float((pos.shape[0] - (j + 1)) / pos.shape[0])
            unsafe_traj.append(pos)
            tte_list.append(tte)
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'r')
        else:
            print("Unsafe Demo {} File Path does not exist: {}".format(i, fname))

    daring_traj = []
    for i in range(0, nDaring):
        fname = csv_dir + 'daring/' + str(i + 1) + '_eePosition.txt'
        if os.path.exists(fname):
            pos = np.loadtxt(fname, delimiter=',')[:, 0:3]
            daring_traj.append(pos)
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b')
        else:
            print("Daring Demo {} File Path does not exist: {}".format(i, fname))

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title('Franka CBF Vel_lim: Position Data')
    if save: plt.savefig(fig_path + 'demonstration_data.pdf')
    # plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    safe_vel = []
    nInvalidSafe = 0
    for i in range(0, nSafe):
        fname = csv_dir + 'safe/' + str(i+1) + '_eeVelocity.txt'
        if os.path.exists(fname):
            vel = np.loadtxt(fname, delimiter=',')[:,0:3]
            safe_vel.append(vel)
            ax.plot(vel[:,0], vel[:,1], vel[:,2], 'g')
        else:
            print("Safe Demo {} File Path does not exist: {}".format(i, fname))
            nInvalidSafe += 1

    unsafe_vel = []
    nInvalidUnsafe = 0
    for i in range(0, nUnsafe):
        fname = csv_dir + 'unsafe/' + str(i+1) + '_eeVelocity.txt'
        if os.path.exists(fname):
            vel = np.loadtxt(fname, delimiter=',')[:,0:3]
            unsafe_vel.append(vel)
            ax.plot(vel[:,0], vel[:,1], vel[:,2], 'r')
        else:
            print("Unsafe Demo {} File Path does not exist: {}".format(i, fname))
            nInvalidUnsafe += 1

    daring_vel = []
    nInvalidDaring = 0
    for i in range(0, nDaring):
        fname = csv_dir + 'daring/' + str(i+1) + '_eeVelocity.txt'
        if os.path.exists(fname):
            vel = np.loadtxt(fname, delimiter=',')[:,0:3]
            daring_vel.append(vel)
            ax.plot(vel[:,0], vel[:,1], vel[:,2], 'b')
        else:
            print("Daring Demo {} File Path does not exist: {}".format(i, fname))
            nInvalidDaring += 1

    safe_acc = []
    for i in range(0, nSafe):
        fname = csv_dir + 'safe/' + str(i+1) + '_eeAcceleration.txt'
        if os.path.exists(fname):
            acc = np.loadtxt(fname, delimiter=',')[:,0:3]
            safe_acc.append(acc)
            ax.plot(acc[:,0], acc[:,1], acc[:,2], 'b')
        else:
            print("Safe Demo {} File Path does not exist: {}".format(i, fname))

    unsafe_acc = []
    for i in range(0, nUnsafe):
        fname = csv_dir + 'unsafe/' + str(i+1) + '_eeAcceleration.txt'
        if os.path.exists(fname):
            acc = np.loadtxt(fname, delimiter=',')[:,0:3]
            unsafe_acc.append(acc)
            ax.plot(acc[:,0], acc[:,1], acc[:,2], 'b')
        else:
            print("Safe Demo {} File Path does not exist: {}".format(i, fname))

    daring_acc = []
    for i in range(0, nDaring):
        fname = csv_dir + 'daring/' + str(i+1) + '_eeAcceleration.txt'
        if os.path.exists(fname):
            acc = np.loadtxt(fname, delimiter=',')[:,0:3]
            daring_acc.append(acc)
            ax.plot(acc[:,0], acc[:,1], acc[:,2], 'b')
        else:
            print("Daring Demo {} File Path does not exist: {}".format(i, fname))

    nSafe = nSafe-nInvalidSafe
    nUnsafe = nUnsafe-nInvalidUnsafe
    nDaring = nDaring-nInvalidDaring
    print(nSafe, nUnsafe, nDaring)
    xtraj = np.hstack((safe_traj[0], safe_vel[0]))
    safe_pts = xtraj
    safe_u = safe_acc[0]
    for i in range(1, nSafe):
        xtraj = np.hstack((safe_traj[i], safe_vel[i]))
        safe_pts = np.vstack((safe_pts, xtraj))
        safe_u = np.vstack((safe_u, safe_acc[i]))

    xtraj = np.hstack((unsafe_traj[0], unsafe_vel[0]))
    unsafe_pts = xtraj
    unsafe_ttelist = tte_list[0]
    unsafe_u = unsafe_acc[0]
    for i in range(1, nUnsafe):
        xtraj = np.hstack((unsafe_traj[i], unsafe_vel[i]))
        unsafe_pts = np.vstack((unsafe_pts, xtraj))
        unsafe_ttelist = np.vstack((unsafe_ttelist, tte_list[i]))
        unsafe_u = np.vstack((unsafe_u, unsafe_acc[i]))


    xtraj = np.hstack((daring_traj[0], daring_vel[0]))
    semisafe_pts = xtraj
    semisafe_u = daring_acc[0]
    for i in range(1, nDaring):
        xtraj = np.hstack((daring_traj[i], daring_vel[i]))
        semisafe_pts = np.vstack((semisafe_pts, xtraj))
        semisafe_u = np.vstack((semisafe_u, daring_acc[i]))

    print(safe_pts.shape, semisafe_pts.shape, unsafe_pts.shape)
    all_pts = np.vstack((safe_pts, semisafe_pts, unsafe_pts))

    print(safe_u.shape, semisafe_u.shape, unsafe_u.shape)
    all_u = np.vstack((safe_u, semisafe_u, unsafe_u))

    xsafe_ulim = np.amax(safe_pts, axis=0)
    xdaring_ulim = np.amax(semisafe_pts, axis=0)
    xunsafe_ulim = np.amax(unsafe_pts, axis=0)
    x_max = np.amax(all_pts, axis=0)
    print("User {} X Upper Limits: \n Safe Limits: {} \n Daring Limits: {} \n Unsafe Limits: {} \n All Limits: {}".format(user_number, xsafe_ulim, xdaring_ulim, xunsafe_ulim, x_max))

    xsafe_llim = np.amin(safe_pts, axis=0)
    xdaring_llim = np.amin(semisafe_pts, axis=0)
    xunsafe_llim = np.amin(unsafe_pts, axis=0)
    x_min = np.amin(all_pts, axis=0)
    print("User {} X Lower Limits: \n Safe Limits: {} \n Daring Limits: {} \n Unsafe Limits: {} \n All Limits: {}".format(user_number, xsafe_llim, xdaring_llim, xunsafe_llim, x_min))

    usafe_ulim = np.amax(safe_u, axis=0)
    udaring_ulim = np.amax(semisafe_u, axis=0)
    uunsafe_ulim = np.amax(unsafe_u, axis=0)
    u_max = np.amax(all_u, axis=0)
    print("User {} U Upper Limits: \nSafe Limits: {} \n Daring Limits: {} \n Unsafe Limits: {} \n All Limits: {}".format(user_number, usafe_ulim, udaring_ulim, uunsafe_ulim, u_max))

    usafe_llim = np.amin(safe_u, axis=0)
    udaring_llim = np.amin(semisafe_u, axis=0)
    uunsafe_llim = np.amin(unsafe_u, axis=0)
    u_min = np.amin(all_u, axis=0)
    print("User {} U Lower Limits: \n Safe Limits: {} \n Daring Limits: {} \n Unsafe Limits: {} \n All Limits: {}".format(user_number, usafe_llim, udaring_llim, uunsafe_llim, u_min))

    return x_max, x_min, u_max, u_min

def get_workspace_limits(bag="/home/ros/ros_ws/src/learning_safety_margin/data/workspace_boundaries.bag"):
    pos, _ = read_bag(bag)
    print(pos.shape)
    ws_upper_limits = np.amax(pos, axis=0)
    ws_lower_limits = np.amin(pos, axis=0)
    ws_limits = np.vstack((ws_lower_limits, ws_upper_limits)).T
    print(ws_upper_limits.shape, ws_lower_limits.shape, ws_limits.shape)
    print("Workspace Limits: {}, \n Lower Limits: {} \n Upper Limits: {}".format(ws_limits, ws_lower_limits, ws_upper_limits))
    return ws_limits

def parser():
    # Parse arguments calls functions accordingly

    parser = argparse.ArgumentParser(description="Process rosbags into csv and gets limits")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", default=True)
    group.add_argument("-q", "--quiet", action="store_true", default = False)
    parser.add_argument("-p", "--plot", action="store_true", default=False)
    parser.add_argument("-u", "--user_number", type=str, default='0', nargs='?', help="The user data to process")
    parser.add_argument("-l", "--learning_algorithm", type=str, choices=['pos', 'vel'], default = 'pos',  nargs='?', help="The learning algorithm to run")
    parser.add_argument("-s", "--to_smooth", type=str, choices=['0', '1'], default='1', nargs='?', help="option to smooth cartesian velocities")

    args = parser.parse_args()

    set_up_dir(args.user_number)

    if bags_need_processing(args.user_number):
        print("Processing rosbags for User_" + args.user_number)
        process_user_rosbags(args.user_number, args.to_smooth)
        print("Finished processing rosbags for User_" + args.user_number)
    else:
        print("Rosbags already processed for User_"+ args.user_number)

    get_user_limits(args.user_number)
    get_workspace_limits()
    get_workspace_limits(bag="/home/ros/ros_ws/src/learning_safety_margin/data/task-space-boundaries.bag")


if __name__ == "__main__":
    parser()

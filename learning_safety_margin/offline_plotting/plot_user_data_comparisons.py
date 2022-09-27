#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
import time
import rosbag
import seaborn as sns

from bag2csv import process_user_rosbags
from robot_model import Model, InverseKinematicsParameters, QPInverseVelocityParameters
import state_representation as sr
from get_statespace_limits import *

def extract_user_data(user_number):
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

    safe_traj = []
    for i in range(0, nSafe):
        fname = csv_dir + 'safe/' + str(i + 1) + '_eePosition.txt'
        if os.path.exists(fname):
            pos = np.loadtxt(fname, delimiter=',')[:,0:3]
            safe_traj.append(pos)
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
        else:
            print("Unsafe Demo {} File Path does not exist: {}".format(i, fname))

    daring_traj = []
    for i in range(0, nDaring):
        fname = csv_dir + 'daring/' + str(i + 1) + '_eePosition.txt'
        if os.path.exists(fname):
            pos = np.loadtxt(fname, delimiter=',')[:, 0:3]
            daring_traj.append(pos)
        else:
            print("Daring Demo {} File Path does not exist: {}".format(i, fname))


    safe_vel = []
    nInvalidSafe = 0
    for i in range(0, nSafe):
        fname = csv_dir + 'safe/' + str(i+1) + '_eeVelocity.txt'
        if os.path.exists(fname):
            vel = np.loadtxt(fname, delimiter=',')[:,0:3]
            safe_vel.append(vel)
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
        else:
            print("Daring Demo {} File Path does not exist: {}".format(i, fname))
            nInvalidDaring += 1

    safe_acc = []
    for i in range(0, nSafe):
        fname = csv_dir + 'safe/' + str(i+1) + '_eeAcceleration.txt'
        if os.path.exists(fname):
            acc = np.loadtxt(fname, delimiter=',')[:,0:3]
            safe_acc.append(acc)
        else:
            print("Safe Demo {} File Path does not exist: {}".format(i, fname))

    unsafe_acc = []
    for i in range(0, nUnsafe):
        fname = csv_dir + 'unsafe/' + str(i+1) + '_eeAcceleration.txt'
        if os.path.exists(fname):
            acc = np.loadtxt(fname, delimiter=',')[:,0:3]
            unsafe_acc.append(acc)
        else:
            print("Safe Demo {} File Path does not exist: {}".format(i, fname))

    daring_acc = []
    for i in range(0, nDaring):
        fname = csv_dir + 'daring/' + str(i+1) + '_eeAcceleration.txt'
        if os.path.exists(fname):
            acc = np.loadtxt(fname, delimiter=',')[:,0:3]
            daring_acc.append(acc)
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

    return safe_pts, unsafe_pts, semisafe_pts, semisafe_u

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

    ws_lims = get_workspace_limits(bag="/home/ros/ros_ws/src/learning_safety_margin/data/task-space-boundaries.bag")
    print(ws_lims, ws_lims.shape)
    safe_data = []
    unsafe_data = []
    daring_data = []
    daring_u = []


    for i in range(2, 9):
        args.user_number = str(i)
        set_up_dir(args.user_number)
        if bags_need_processing(args.user_number):
            print("Processing rosbags for User_" + args.user_number)
            process_user_rosbags(args.user_number, args.to_smooth)
            print("Finished processing rosbags for User_" + args.user_number)
        else:
            print("Rosbags already processed for User_" + args.user_number)

        safe_pts, unsafe_pts, semisafe_pts, semisafe_u = extract_user_data(str(i))
        safe_data.append(safe_pts)
        unsafe_data.append(unsafe_pts)
        daring_data.append(semisafe_pts)
        daring_u.append(semisafe_u)
    print(len(safe_data))
    palette = sns.color_palette("husl", len(safe_data))
    print(palette)

    figx, axx = plt.subplots()
    figy, axy = plt.subplots()
    figz, axz = plt.subplots()

    figx_s, axx_s = plt.subplots()
    figy_s, axy_s = plt.subplots()
    figz_s, axz_s = plt.subplots()
    fig_s = plt.figure()
    ax_s = fig_s.add_subplot(111, projection='3d')

    figx_u, axx_u = plt.subplots()
    figy_u, axy_u = plt.subplots()
    figz_u, axz_u = plt.subplots()
    fig_u = plt.figure()
    ax_u = fig_u.add_subplot(111, projection='3d')

    figx_d, axx_d = plt.subplots()
    figy_d, axy_d = plt.subplots()
    figz_d, axz_d = plt.subplots()
    fig_d = plt.figure()
    ax_d = fig_d.add_subplot(111, projection='3d')


    for i in range(len(safe_data)):
        val = i

        safe_pts = safe_data[i]
        ax_s.plot(safe_pts[:,0], safe_pts[:,1],safe_pts[:,2], marker='.', color=palette[i])
        axx_s.plot(safe_pts[:,0], np.zeros_like(safe_pts[:,0]) + val, '.', color=palette[i])
        axy_s.plot(safe_pts[:,1], np.zeros_like(safe_pts[:,1]) + val, '.', color=palette[i])
        axz_s.plot(safe_pts[:,2], np.zeros_like(safe_pts[:,2]) + val, '.', color=palette[i])

        unsafe_pts = unsafe_data[i]
        ax_u.plot(unsafe_pts[:,0], unsafe_pts[:,1], unsafe_pts[:,2], marker='.', color=palette[i])
        axx_u.plot(unsafe_pts[:,0], np.zeros_like(unsafe_pts[:,0]) + val, '.', color=palette[i])
        axy_u.plot(unsafe_pts[:,1], np.zeros_like(unsafe_pts[:,1]) + val, '.', color=palette[i])
        axz_u.plot(unsafe_pts[:,2], np.zeros_like(unsafe_pts[:,2]) + val, '.', color=palette[i])

        semisafe_pts = daring_data[i]
        ax_d.plot(semisafe_pts[:,0], semisafe_pts[:,1],semisafe_pts[:,2], marker='.', color=palette[i])
        axx_d.plot(semisafe_pts[:,0], np.zeros_like(semisafe_pts[:,0]) + val, '.', color=palette[i])
        axy_d.plot(semisafe_pts[:,1], np.zeros_like(semisafe_pts[:,1]) + val, '.', color=palette[i])
        axz_d.plot(semisafe_pts[:,2], np.zeros_like(semisafe_pts[:,2]) + val, '.', color=palette[i])

    ax_s.set_title('User Data: Safe Trajectories')
    # ax_s.set_xlim(ws_lims[0])
    # ax_s.set_ylim(ws_lims[1])
    # ax_s.set_zlim(ws_lims[2])
    axx_s.set_title('User Data: Safe X')
    # axx_s.set_xlim(ws_lims[0])
    axy_s.set_title('User Data: Safe Y')
    # axy_s.set_xlim(ws_lims[1])
    axz_s.set_title('User Data: Safe Z')
    # axz_s.set_xlim(ws_lims[2])

    ax_u.set_title('User Data: Unsafe Trajectories')
    # ax_u.set_xlim(ws_lims[0])
    # ax_u.set_ylim(ws_lims[1])
    # ax_u.set_zlim(ws_lims[2])
    axx_u.set_title('User Data: Unsafe X')
    # axx_u.set_xlim(ws_lims[0])
    axy_u.set_title('User Data: Unsafe Y')
    # axy_u.set_xlim(ws_lims[1])
    axz_u.set_title('User Data: Unsafe Z')
    # axz_u.set_xlim(ws_lims[2])

    ax_d.set_title('User Data: SemiSafe Trajectories')
    # ax_d.set_xlim(ws_lims[0])
    # ax_d.set_ylim(ws_lims[1])
    # ax_d.set_zlim(ws_lims[2])
    axx_d.set_title('User Data: SemiSafe X')
    # axx_d.set_xlim(ws_lims[0])
    axy_d.set_title('User Data: SemiSafe Y')
    # axy_d.set_xlim(ws_lims[1])
    axz_d.set_title('User Data: SemiSafe Z')
    # axz_d.set_xlim(ws_lims[2])
    plt.show()



if __name__ == "__main__":
    parser()

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from learning_safety_margin.vel_control_utils import *

def plot_ref_vs_rec(user, traj, rec):

    traj_name =" MPC traj #"+traj+" of User_"+user

    # Formatting for ease
    rec_time = rec[:, 0]

    rec_pos = rec[:, 1:4]
    rec_ref_pos = rec[:, 10:13]
    error_pos = rec_pos - rec_ref_pos

    rec_vel = rec[:, 4:7]
    rec_ref_vel = rec[:, 13:16]
    error_vel = rec_vel - rec_ref_vel

    rec_acc = rec[:, 7:10]
    rec_ref_acc = rec[:, 16:19]
    error_acc = rec_acc - rec_ref_acc

    rec_ort = rec[:, 26:30]
    rec_ref_ort = rec[:, 30:34]

    rec_ang_vel = rec[:, 34:37]
    rec_ref_ang_vel = rec[:, 37:40]

    rec_tor = rec[:, 19:26]

    print("PLOTTING REFERENCE VS RECORDED TRAJECTORY")

    # 3D position plot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.plot(rec_pos[:, 0], rec_pos[:, 1], rec_pos[:, 2])
    plt.plot(rec_ref_pos[:, 0], rec_ref_pos[:, 1], rec_ref_pos[:, 2])
    plt.xlim(ws_lim[0])
    plt.ylim(ws_lim[1])
    ax.set_zlim(ws_lim[2])
    fig.suptitle("EE pos vs ref of '{}'".format(traj_name))
    fig.legend(labels=['Recorded', 'Reference'])

    # 3D velocity plot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.plot(rec_vel[:, 0], rec_vel[:, 1], rec_vel[:, 2])
    plt.plot(rec_ref_vel[:, 0], rec_ref_vel[:, 1], rec_ref_vel[:, 2])
    plt.xlim(np.array(vdot_lim)/2)
    plt.ylim(np.array(vdot_lim)/2)
    ax.set_zlim(np.array(vdot_lim)/2)
    fig.suptitle("EE vel vs ref of '{}'".format(traj_name))
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot reference and state
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE Position and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time[:], rec_pos[:, i])
        ax.plot(rec_time[:], rec_ref_pos[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot position error
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE position error of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, error_pos[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")

    # Plot velocities and reference
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE Velocity and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, rec_vel[:, i])
        ax.plot(rec_time, rec_ref_vel[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot velocity error
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE Velocity error of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, error_vel[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")

    # Plot acc and reference - no acceleration recorded
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE acceleration and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, rec_acc[:, i])
        ax.plot(rec_time, rec_ref_acc[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot orientation and reference
    fig, axs = plt.subplots(4, 1)
    fig.suptitle("EE Orientation and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, rec_ort[:, i])
        ax.plot(rec_time, rec_ref_ort[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot angular velocity and reference
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE angular velocity and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, rec_ang_vel[:, i])
        ax.plot(rec_time, rec_ref_ang_vel[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot torques
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint Torques of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, rec_tor[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")

    plt.show()


def parser():
    # Parse arguments calls functions accordingly

    parser = argparse.ArgumentParser(description="Plot the reference vs recorded joint State of a replayed kinesthetic demonstration")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", default=True)
    group.add_argument("-q", "--quiet", action="store_true", default = False)
    parser.add_argument("-u", "--user_number", type=str, default='1', nargs='?', help="The user data to plot")
    parser.add_argument("-s", "--safety", type=str, choices=['safe', 'daring', 'unsafe'], default='safe',  nargs='?', help="The safety margin of the trajectory")
    parser.add_argument("-t", "--trajectory_number", type=str, default='1', nargs='?', help="The trajectory to plot")

    args = parser.parse_args()

    return args.user_number, args.safety, args.trajectory_number


if __name__ == "__main__":

    user_nbr, safety, traj_nbr = parser()

    # Check directories and create them if needed
    data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_" + user_nbr + "/MPC/"
    fn = data_dir + traj_nbr+"_"+safety+"_MPC_eeState.npy"

    print("Plotting recorded vs reference for User_", user_nbr)
    print("Trajectory name : ", os.path.split(fn)[1], "\n")

    plannerTraj = np.load(fn)#, allow_pickle=True)

    plot_ref_vs_rec(user_nbr, traj_nbr, plannerTraj)
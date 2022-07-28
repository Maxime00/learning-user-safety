import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from learning_safety_margin.vel_control_utils import *

def pandas_to_np(df):

    temp_time = df['time'].to_numpy()
    temp_pos = df['position'].to_numpy()
    temp_vel = df['velocity'].to_numpy()
    temp_torques = df['torques'].to_numpy()
    reference_arr = np.zeros((len(temp_time), 1 + 7 * 3))
    reference_arr[:, 0] = temp_time
    for i in range(0, (len(temp_pos))):
        reference_arr[i, 1:8] = temp_pos[i]
        reference_arr[i, 8:15] = temp_vel[i]
        reference_arr[i, 15:22] = temp_torques[i]

    return reference_arr

def plot_ref_vs_rec(user, traj, df_rec):

    traj_name ="traj #"+traj+" of User_"+user

    # Formatting for ease
    rec = pandas_to_np(df_rec)

    # Formatting for ease
    rec_time = rec[:, 0]

    rec_pos = rec[:, 1:8]

    rec_vel = rec[:, 8:15]

    rec_tor = rec[:, 15:22]


    # 3D position plot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.plot(rec_pos[:, 0], rec_pos[:, 1], rec_pos[:, 2])
    # plt.xlim(ws_lim[0])
    # plt.ylim(ws_lim[1])
    # ax.set_zlim(ws_lim[2])
    fig.suptitle("EE pos vs ref of '{}'".format(traj_name))
    fig.legend(labels=['Recorded', 'Reference'])

    # 3D velocity plot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.plot(rec_vel[:, 0], rec_vel[:, 1], rec_vel[:, 2])
    # plt.xlim(np.array(vdot_lim) / 2)
    # plt.ylim(np.array(vdot_lim) / 2)
    # ax.set_zlim(np.array(vdot_lim) / 2)
    fig.suptitle("EE vel vs ref of '{}'".format(traj_name))
    fig.legend(labels=['Recorded', 'Reference'])
    #
    # # Plot reference and state
    # fig, axs = plt.subplots(4, 2)
    # fig.suptitle("Joint state and reference of '{}'".format(traj_name))
    #
    # for i, ax in enumerate(axs.ravel()[:-1]):
    #     ax.plot(rec_time[:], rec_pos[:, i])
    #     ax.plot(rec_time[:], ref_pos[:, i])
    #     ax.set(ylabel="joint {}".format(i))
    #     ax.set(xlabel="Time [sec]")
    # fig.legend(labels=['Recorded', 'Reference'], loc=(.6,.15))
    #
    # # Plot position error
    # fig, axs = plt.subplots(4, 2)
    # fig.suptitle("Joint position error of '{}'".format(traj_name))
    #
    # for i, ax in enumerate(axs.ravel()[:-1]):
    #     ax.plot(rec_time, error_pos[:, i])
    #     ax.set(ylabel="joint {}".format(i))
    #     ax.set(xlabel="Time [sec]")
    #
    #
    # # Plot velocities and reference
    # fig, axs = plt.subplots(4, 2)
    # fig.suptitle("Joint Velocities and reference of '{}'".format(traj_name))
    #
    # for i, ax in enumerate(axs.ravel()[:-1]):
    #     ax.plot(rec_time, rec_vel[:, i])
    #     ax.plot(rec_time, ref_vel[:, i])
    #     ax.set(ylabel="joint {}".format(i))
    #     ax.set(xlabel="Time [sec]")
    # fig.legend(labels=['Recorded', 'Reference'], loc=(.6, .15))
    #
    # # Plot velocity error
    # fig, axs = plt.subplots(4, 2)
    # fig.suptitle("Joint Velocity error of '{}'".format(traj_name))
    #
    # for i, ax in enumerate(axs.ravel()[:-1]):
    #     ax.plot(rec_time, error_vel[:, i])
    #     ax.set(ylabel="joint {}".format(i))
    #     ax.set(xlabel="Time [sec]")
    #
    # # Plot torques and reference
    # fig, axs = plt.subplots(4, 2)
    # fig.suptitle("Joint Torques and reference of '{}'".format(traj_name))
    #
    # for i, ax in enumerate(axs.ravel()[:-1]):
    #     ax.plot(rec_time, rec_tor[:, i])
    #     ax.plot(rec_time, ref_tor[:, i])
    #     ax.set(ylabel="joint {}".format(i))
    #     ax.set(xlabel="Time [sec]")
    # fig.legend(labels=['Recorded', 'Reference'], loc=(.6, .15))
    #
    # # Plot torque error
    # fig, axs = plt.subplots(4, 2)
    # fig.suptitle("Joint Torques error of '{}'".format(traj_name))
    #
    # for i, ax in enumerate(axs.ravel()[:-1]):
    #     ax.plot(rec_time, error_tor[:, i])
    #     ax.set(ylabel="joint {}".format(i))
    #     ax.set(xlabel="Time [sec]")

    plt.show()


def parser():
    # Parse arguments calls functions accordingly

    parser = argparse.ArgumentParser(description="Plot the reference vs recorded joint State of a replayed kinesthetic demonstration")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", default=True)
    group.add_argument("-q", "--quiet", action="store_true", default = False)
    parser.add_argument("-u", "--user_number", type=str, default='1', nargs='?', help="The user data to plot")
    parser.add_argument("-s", "--safety", type=str, choices=['safe', 'daring', 'unsafe'], default = 'safe',  nargs='?', help="The safety margin of the trajectory")
    parser.add_argument("-t", "--trajectory_number", type=str, default='1', nargs='?', help="The trajectory to plot")

    args = parser.parse_args()

    return args.user_number, args.safety, args.trajectory_number


if __name__ == "__main__":

    user_nbr, safety, traj_nbr = parser()

    # Check directories and create them if needed
    data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_" + user_nbr + "/csv/" + safety +"/"
    fn = data_dir + traj_nbr+"_jointState.pkl"

    kineTraj = pd.read_pickle(fn)#, allow_pickle=True)

    plot_ref_vs_rec(user_nbr, traj_nbr, kineTraj)
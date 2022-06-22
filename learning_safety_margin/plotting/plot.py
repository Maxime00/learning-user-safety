import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

def plot_ref_vs_rec(user, traj, rec):

    traj_name ="traj #"+traj+" of User_"+user

    # Formatting for ease
    rec_time = rec[:, 0]

    rec_pos = rec[:, 1:8]
    ref_pos = rec[:, 22:29]
    error_pos = rec_pos - ref_pos

    rec_vel = rec[:, 8:15]
    ref_vel = rec[:,29:36]
    error_vel = rec_vel - ref_vel

    rec_tor = rec[:, 15:22]
    ref_tor = rec[:, 36:43]
    error_tor = rec_tor - ref_tor

    print("PLOTTING REFERENCE VS RECORDED TRAJECTORY")

    # Plot reference and state
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint state and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time[:], rec_pos[:, i])
        ax.plot(rec_time[:], ref_pos[:, i])
        ax.set(ylabel="joint {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'], loc=(.6,.15))

    # Plot position error
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint position error of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, error_pos[:, i])
        ax.set(ylabel="joint {}".format(i))
        ax.set(xlabel="Time [sec]")


    # Plot velocities and reference
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint Velocities and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, rec_vel[:, i])
        ax.plot(rec_time, ref_vel[:, i])
        ax.set(ylabel="joint {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'], loc=(.6, .15))

    # Plot velocity error
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint Velocity error of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, error_vel[:, i])
        ax.set(ylabel="joint {}".format(i))
        ax.set(xlabel="Time [sec]")

    # Plot torques and reference
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint Torques and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, rec_tor[:, i])
        ax.plot(rec_time, ref_tor[:, i])
        ax.set(ylabel="joint {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'], loc=(.6, .15))

    # Plot torque error
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint Torques error of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, error_tor[:, i])
        ax.set(ylabel="joint {}".format(i))
        ax.set(xlabel="Time [sec]")

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
    fn = data_dir + traj_nbr+"_replay_jointState.npy"

    # data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/example_traj_to_replay/csv/"
    # fn = data_dir + "recorded_jointState.npy"

    replayedTraj = np.load(fn)#, allow_pickle=True)

    plot_ref_vs_rec(user_nbr, traj_nbr, replayedTraj)
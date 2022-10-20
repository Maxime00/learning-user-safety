import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import pickle
from learning_safety_margin.vel_control_utils import *

def plot_ref_vs_rec(user, safety, traj_df):

    # Formatting for ease
    x_list = traj_df['X'].to_numpy()
    labels = traj_df['Labels'].to_numpy()

    if 'Start points' in traj_df.columns:
        start_list = traj_df['Start points'].to_numpy()
        end_list = traj_df['End points'].to_numpy()

    # count number of safe and unsafe trajectories
    nbr_safe = len(traj_df.index[traj_df['Labels'] == 'safe'])
    nbr_unsafe = len(traj_df.index[traj_df['Labels'] == 'unsafe'])

    # 3D position plot
    # # Plot reference and state
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if safety == 'both':
        cmap_safe = plt.cm.get_cmap('Greens')
        cmap_unsafe = plt.cm.get_cmap('Reds')
        crange_safe = np.linspace(0.1, 1, nbr_safe + 2)
        crange_unsafe = np.linspace(0.1, 1, nbr_unsafe + 2)
    else :
        cmap_safe = plt.cm.get_cmap('tab20')
        cmap_unsafe = plt.cm.get_cmap('tab20')
        crange_safe = np.linspace(0, 1, nbr_safe + 2)
        crange_unsafe = np.linspace(0, 1, nbr_unsafe + 2)


    for i in range(len(labels)):
        if labels[i] == 'safe' and (safety =='safe' or safety=='both'):
            # Plot Trajectories
            rgba = cmap_safe(crange_safe[i])
            plt.plot(x_list[i][:, 0], x_list[i][:, 1], x_list[i][:, 2], color=rgba, label=f'Safe #{i + 1} [{i}]')

            # Plot Start and End Points
            if 'Start points' in traj_df.columns:
                ax.scatter(start_list[i][0], start_list[i][1], start_list[i][2], s=3, c=rgba)
                ax.scatter(end_list[i][0], end_list[i][1], end_list[i][2], '*', s=6, c=rgba)

        if labels[i] == 'unsafe' and (safety =='unsafe' or safety=='both'):
            # Plot Trajectories
            rgba = cmap_unsafe(crange_unsafe[i - nbr_safe])
            plt.plot(x_list[i][:, 0], x_list[i][:, 1], x_list[i][:, 2], color=rgba,
                     label=f'Unsafe #{i - nbr_safe  + 1} [{i}]')

            # Plot Start and End Points
            if 'Start points' in traj_df.columns:
                ax.scatter(start_list[i][0], start_list[i][1], start_list[i][2], s=3, c=rgba)
                ax.scatter(end_list[i][0], end_list[i][1], end_list[i][2], '*', s=5, c=rgba)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    fig.legend()
    fig.suptitle("Planned trajectories for User_"+user)

    plt.show()


def remove_trajectories(df_traj, idx_to_remove):

    # REMOVE TRAJECTORIES
    cleaned_df = df_traj.drop(idx_to_remove)

    new_nbr_of_traj = len(cleaned_df.index)

    cleaned_df.to_pickle(path=data_dir+ str(new_nbr_of_traj)+"_planned_trajectories.pkl",  )

    return cleaned_df


def parser():
    # Parse arguments calls functions accordingly

    parser = argparse.ArgumentParser(description="Plot the reference vs recorded joint State of a replayed kinesthetic demonstration")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", default=True)
    group.add_argument("-q", "--quiet", action="store_true", default = False)
    parser.add_argument("-u", "--user_number", type=str, default='1', nargs='?', help="The user data to plot")
    parser.add_argument("-s", "--safety", type=str, default='both', choices=['safe', 'unsafe', 'both'], nargs='?', help="The safety trajectories to plot")
    parser.add_argument("-r", "--remove", type=str, default='False', choices=['True', 'False'], nargs='?',
                        help="Whether to remove trajectories or not")

    args = parser.parse_args()
    print(args)

    return args.user_number, args.safety, eval(args.remove)


if __name__ == "__main__":

    user_nbr, safety, remove = parser()

    # Index list of trajectories to be removed form dict
    traj_to_remove = [4,18,22,23]

    # Check directories and create them if needed
    data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_" + user_nbr +"/"
    fn = data_dir + "30_planned_trajectories_19-10.pkl"

    with open(fn, 'rb') as f:
        plannedTraj = pickle.load(f)

    traj_df = pd.DataFrame.from_dict(plannedTraj)

    if remove:
        traj_df = remove_trajectories(traj_df, traj_to_remove)

    plot_ref_vs_rec(user_nbr, safety, traj_df)
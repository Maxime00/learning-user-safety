from process_and_learn import process_and_learn
import shutil
import os
import pickle
from learning_safety_margin.cbf_traj_generator import trajGenerator

data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/" #User_" + user_number + "/csv"

#For each user
for user in range(2,10):

    user_number =str(user)
    # Remove all csv repos
    csv_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_" + user_number + "/csv"
    if os.path.isdir(csv_dir):
        shutil.rmtree(csv_dir)

    # process and learn
    process_and_learn(user_number, "vel", "1")

    # PLAN trajectories
    # Set up directories
    user_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_" + user_number + "/"

    ## Set up Trajectory Planner
    data = pickle.load(open(user_dir + "vel_data_dict.p", "rb"))
    params = data["theta"]
    bias_param = data["bias"]
    # slack_param = data["unsafe_slack"]
    centers = data["rbf_centers"]
    stds = data["rbf_stds"]
    if bias_param is None: bias_param = 0.1

    # Traj planner parameters
    freq = 300
    dt = 0.1  # 1./freq
    n_steps = 50
    nbr_demos = 15

    # Init Trajectory Generation
    print("Generating trajectories for User_"+ user_number)
    traj_generator = trajGenerator(centers, stds, params, bias_param, dt=dt, n_steps=n_steps, r_gains=0.01,
                                   zero_acc_start=True)

    cbf_traj = traj_generator.generate_all_trajectories(
        num_demos=nbr_demos)  # , init_guess_list=fpath_list)#, plot_debug=show_plots)
    print("Finished generating trajectories for User_"+ user_number)
    with open(user_dir + str(nbr_demos * 2) + '_planned_trajectories.pkl', 'wb') as f:
        pickle.dump(cbf_traj, f)


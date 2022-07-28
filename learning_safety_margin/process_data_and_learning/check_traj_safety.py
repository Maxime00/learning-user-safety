import numpy as np
from learning_safety_margin.cbf_mpc_vel_planner import *
from learning_safety_margin.vel_control_utils import *
import sys
import pickle
import os
# Check passed argument - User number
if len(sys.argv) >= 2:
    user_number = sys.argv[1]
else:
    user_number = '0'


# Get data
data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_" +user_number +"/"

save_dir = data_dir + "MPC/"
if not os.path.isdir(save_dir):
    print("Trajectory Data Not Found")

eeTraj = np.load(save_dir + "3_MPC_eeState.npy")
refTraj = np.load(save_dir + "3_MPC_refState.npy")

print(eeTraj.shape, refTraj.shape)

data = pickle.load(open(data_dir + "vel_data_dict.p", "rb"))
params = data["theta"]
bias_param = data["bias"]
slack_param = data["unsafe_slack"]
centers = data['rbf_centers']
stds = data["rbf_stds"]
bias_param = 0.1

# SET PARAMETERS FOR MPC CONTROL
freq = 100
dt = 0.15  # 1./freq
n_steps = 50

mpc_planner = CBFMPC_Controller(centers, stds, params, bias_param, dt=dt, n_steps=n_steps, r_gains=200,
                                zero_acc_start=True)

safety = mpc_planner.check_safety(refTraj)
print("Trajectory Safe: ", safety)
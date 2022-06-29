import numpy as np
import pickle
import rospy
import time
import sys
import os
import matplotlib.pyplot as plt
import casadi


from learning_safety_margin.cbf_mpc_vel_planner import *
from learning_safety_margin.vel_control_utils import *

# Generate Casadi Functions for h(x) functions
class trajGenerator():

    def __init__(self, centers, stds, theta, bias, daring_offset=0.1, unsafe_offset=0.3, dt=0.1, n_steps=50):

        self.centers = centers
        self.stds = stds
        self.theta = theta
        self.bias = bias
        self.daring_offset=daring_offset
        self.unsafe_offset = unsafe_offset

        self.daring_bias = self.bias + self.daring_offset
        self.unsafe_bias = self.bias = self.unsafe_offset

        self.dt = dt
        self.n_steps = n_steps


        # Generate H function(s) for Casadi
        c = self.centers[0]
        s = self.stds[0]
        self.phi = casadi.exp(-1 / (2 * s ** 2) * casadi.norm_2(self.x - c) ** 2)
        for i in range(1, len(self.centers)):
            c = self.centers[i]
            s = self.stds[i]
            rbf = casadi.exp(-1 / (2 * s ** 2) * casadi.norm_2(self.x - c) ** 2)
            self.phi = casadi.horzcat(self.phi, rbf)
        self.h = casadi.mtimes(self.phi, self.theta) + self.bias
        self.h_fun = casadi.Function('h_fun', [self.x], [self.h])

        # Daring
        self.h_daring = casadi.mtimes(self.phi, self.theta) + self.daring_bias
        self.h_daring_fun = casadi.Function('h_daring_fun', [self.x], [self.h_daring])

        # Unsafe
        self.h_unsafe = casadi.mtimes(self.phi, self.theta) + self.unsafe_bias
        self.h_unsafe_fun = casadi.Function('h_unsafe_fun', [self.x], [self.h_unsafe])

        # Generate Planners
        self.safe_mpc_planner = CBFMPC_Controller(self.centers, self.stds, self.theta, self.bias, dt=self.dt, n_steps=self.n_steps)
        self.daring_mpc_planner = CBFMPC_Controller(self.centers, self.stds, self.theta, self.daring_bias, dt=self.dt, n_steps=self.n_steps)
        self.unsafe_mpc_planner = CBFMPC_Controller(self.centers, self.stds, self.theta, self.unsafe_bias, dt=self.dt, n_steps=self.n_steps)

    def generate_safe_traj(self, start, target):
        X, U, T = self.safe_mpc_planner.control(start, target)
        return X,U,T

    def generate_daring_traj(self, start, target):
        X, U, T = self.daring_mpc_planner.control(start, target)

        #check if daring
        hvals = self.h.map(X.T)
        if np.any(hvals < 0.):
            print('Unsafe Trajectory')
            pass
        elif np.any( 0. <=hvals <= 0.1)
            print('Daring Trajectory')
            return X,U,T
        else:
            print('Safe Trajectory')
            pass

    def generate_unsafe_traj(self, start, target):
        X, U, T = self.unsafe_mpc_planner.control(start, target)
        # Check if unsafe
        hvals = self.h.map(X.T)
        if np.any(hvals < 0.):
            print('Unsafe Trajectory')
            return X, U, T
        else:
            print('Safe Trajectory')
            pass

# Get data
user_number = '0'

data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_"+user_number+"/"

data = pickle.load(open(data_dir + "vel_data_dict.p", "rb"))
params = data["theta"]
bias_param = data["bias"]
slack_param = data["unsafe_slack"]
bias_param = data['bias']#0.1
centers = data['rbf_centers']
stds = data['rbf_stds']

# SET PARAMETERS FOR MPC CONTROL
dt = 0.1  # 1./freq
n_steps = 50

# Init Trajectory Generation
print("Instantiating MPC Planner, should take around %s seconds... \n"%(n_steps/2))
instantiate_start = time.time()
traj_generator = trajGenerator(centers, stds, params, bias_param, dt=dt, n_steps=n_steps)
print("FINISH INSTANTIATING MPC PLANNER IN %s seconds \n" % (time.time() - instantiate_start))



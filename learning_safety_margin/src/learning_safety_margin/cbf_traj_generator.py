import numpy as np
import pickle
import rospy
import time
import sys
import os
import matplotlib.pyplot as plt
import casadi
import random

from learning_safety_margin.cbf_mpc_vel_planner import *
from learning_safety_margin.vel_control_utils import *

class trajGenerator():

    def __init__(self, centers, stds, theta, bias, daring_offset=0.1, unsafe_offset=30., dt=0.1, n_steps=50, r_gains = 1, zero_acc_start = False):

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
        self.r_gains = r_gains
        self.zero_acc_start = zero_acc_start

        # Generate H function(s) for Casadi

        # Set up variables
        self.x = casadi.SX.sym('x', 6)

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
        self.safe_mpc_planner = CBFMPC_Controller(self.centers, self.stds, self.theta, self.bias, dt=self.dt, n_steps=self.n_steps,r_gains = self.r_gains, zero_acc_start=self.zero_acc_start)
        self.daring_mpc_planner = CBFMPC_Controller(self.centers, self.stds, self.theta, self.daring_bias, dt=self.dt, n_steps=self.n_steps, r_gains = self.r_gains, zero_acc_start=self.zero_acc_start)
        self.unsafe_mpc_planner = CBFMPC_Controller(self.centers, self.stds, self.theta, self.unsafe_bias, dt=self.dt, n_steps=self.n_steps, r_gains = self.r_gains, zero_acc_start=self.zero_acc_start)

    def generate_safe_traj(self, start, target):
        X, U, T = self.safe_mpc_planner.control(start, target)
        # convert into np arrays for reading ease
        X = np.array(X)  # pos + vel  (n_steps, 6)
        U = np.array(U)  # accel (n_steps, 3)
        T = np.array(T)
        # check if safe
        hvals = np.zeros(X.shape[0])
        for i in range(len(X)):
            hvals[i] = self.h_fun(X[i])
        # hvals = self.h_fun.map(X.T)
        if np.any(hvals <= 0.):
            print('Unsafe Trajectory')
            return None
        else:
            return [X,U,T]

    def generate_daring_traj(self, start, target):
        X, U, T = self.daring_mpc_planner.control(start, target)
        # convert into np arrays for reading ease
        X = np.array(X)  # pos + vel  (n_steps, 6)
        U = np.array(U)  # accel (n_steps, 3)
        T = np.array(T)

        #check if daring
        hvals = np.zeros(X.shape[0])
        for i in range(len(X)):
            hvals[i] = self.h_fun(X[i])
        print(hvals)
        print(hvals >= 0. and hvals <= 0.1)
        print(0. <= hvals <= 0.1)
        # hvals = self.h.map(X.T)
        if np.any(hvals < 0.):
            print('Unsafe Trajectory')
            return None
        elif np.all(hvals >= 0.) and np.any(hvals <= 0.1):
            print('Daring Trajectory')
            return [X,U,T]
        else:
            print('Safe Trajectory')
            return None


    def generate_unsafe_traj(self, start, target):
        X, U, T = self.unsafe_mpc_planner.control(start, target)
        # convert into np arrays for reading ease
        X = np.array(X)  # pos + vel  (n_steps, 6)
        U = np.array(U)  # accel (n_steps, 3)
        T = np.array(T)

        # Check if unsafe
        hvals = np.zeros(X.shape[0])
        for i in range(len(X)):
            hvals[i] = self.h_fun(X[i])

        # hvals = self.h.map(X.T)
        if np.any((hvals < 0.)):
            print('Unsafe Trajectory')
            return [X, U, T]
        else:
            print('Safe Trajectory')
            return None

    def make_trial_conditions(self):
        # Generate starting pose

        inRange = False

        while not inRange:
            x = np.random.uniform(x_lim[0], x_lim[1])
            # print("x:", x)
            if 0.3 <= x <= 0.7:
                if random.random() < 0.5:
                    y = np.random.uniform(0.3, 0.45)
                    # print("y :", y)
                else:
                    y = np.random.uniform(-0.3, -0.45)
                xt = x + np.random.uniform(0.3-x, 0.7-x)
                yt = -y
                # print("x, y : ", x, y, xt, yt)
            else:
                y = np.random.uniform(y_lim[0], y_lim[1])
                if x < 0.3:
                    xt = 0.7 + (0.3-x)
                else:
                    xt = x - 0.45
                yt = y + np.random.uniform(y_lim[0]-y, y_lim[1]-y)

            # check coordinates are reachable by robot
            if np.linalg.norm([x,y]) <= .75: # .8 should do it
                inRange = True

        z = 0.2
        xdot = 0
        ydot = 0
        zdot = 0
        print("x, y and xt, yt : ", x, y, xt, yt)
        x0 = np.hstack((x, y, z, xdot, ydot, zdot))
        xt = np.hstack((xt, yt, z, xdot, ydot, zdot))

        return x0, xt

    def generate_all_trajectories(self, num_demos=5):
        x_list = []
        u_list = []
        t_list = []
        labels = []

        num_safe = 0
        while num_safe < num_demos:
            x, xt = self.make_trial_conditions()
            res = self.generate_safe_traj(x, xt)
            if res is not None:
                x_list.append(res[0])
                u_list.append(res[1])
                t_list.append(res[2])
                labels.append('safe')
                num_safe += 1

        print("Generated Safe Trajectories")

        #num_daring = 0
        #while num_daring < num_demos:
        #    x, xt = self.make_trial_conditions()
        #    res = self.generate_daring_traj(x, xt)
        #    if res is not None:
        #        x_daring.append(res[0])
        #        u_daring.append(res[1])
        #        t_daring.append(res[2])
        #        labels.append('daring')
        #        num_daring += 1
        # print("Generated Daring Trajectories")
        x_unsafe = []
        u_unsafe = []
        t_unsafe = []

        #num_unsafe = 0
        #while num_unsafe < num_demos:
        #    x, xt = self.make_trial_conditions()
        #    res = self.generate_unsafe_traj(x, xt)
        #    if res is not None:
        #        x_unsafe.append(res[0])
        #        u_unsafe.append(res[1])
        #        t_unsafe.append(res[2])
        #        num_unsafe += 1
        #print("Generated Unsafe Trajectories")
        cbf_traj_data = {
            'X': x_list,
            'U': u_list,
            'T': t_list,
            'Labels': labels
        }
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(len(x_list)):
            plt.plot(x_list[i][:,0],x_list[i][:,1], x_list[i][:,2])
            print("start: ", x_list[i][0,0:3])
            print("end: ", x_list[i][-1, 0:3])
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        plt.show()

        return cbf_traj_data

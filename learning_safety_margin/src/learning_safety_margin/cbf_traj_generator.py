import matplotlib.colors
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

    def __init__(self, centers, stds, theta, bias, daring_offset=1.0, unsafe_offset=30., dt=0.1, n_steps=50, r_gains = 1, zero_acc_start = False):

        self.centers = centers
        self.stds = stds
        self.theta = theta
        self.bias = bias
        # print("TRAJ BIAS:", bias, self.bias)
        self.safe = True
        self.unsafe = True
        self.semisafe = False
        if self.semisafe: self.daring_offset = daring_offset
        if self.unsafe: self.unsafe_offset = unsafe_offset
        #
        if self.semisafe: self.daring_bias = self.bias + self.daring_offset
        if self.unsafe: self.unsafe_bias = self.bias + self.unsafe_offset

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

        # Safe
        # if self.safe:
        self.h = casadi.mtimes(self.phi, self.theta) + self.bias
        self.h_fun = casadi.Function('h_fun', [self.x], [self.h])

        # Daring
        if self.semisafe:
            self.h_daring = casadi.mtimes(self.phi, self.theta) + self.daring_bias
            self.h_daring_fun = casadi.Function('h_daring_fun', [self.x], [self.h_daring])

        # Unsafe
        if self.unsafe:
            self.h_unsafe = casadi.mtimes(self.phi, self.theta) + self.unsafe_bias
            self.h_unsafe_fun = casadi.Function('h_unsafe_fun', [self.x], [self.h_unsafe])

        # Generate Planners
        if self.safe: self.safe_mpc_planner = CBFMPC_Controller(self.centers, self.stds, self.theta, self.bias, dt=self.dt, n_steps=self.n_steps, pos_gains=2, r_gains = self.r_gains, zero_acc_start=self.zero_acc_start)
        if self.semisafe: self.daring_mpc_planner = CBFMPC_Controller(self.centers, self.stds, self.theta, self.daring_bias, dt=self.dt, n_steps=self.n_steps, r_gains = self.r_gains, zero_acc_start=self.zero_acc_start)
        if self.unsafe: self.unsafe_mpc_planner = CBFMPC_Controller(self.centers, self.stds, self.theta, self.unsafe_bias, dt=self.dt, n_steps=self.n_steps, r_gains = self.r_gains, zero_acc_start=self.zero_acc_start)

    def generate_safe_traj(self, start, target, ig_time=None, ig_pos=None, ig_vel=None, ig_acc=None, num_safe=None, plot_debug=False):
        X, U, T = self.safe_mpc_planner.control(start, target, ig_time=ig_time, ig_pos=ig_pos, ig_vel=ig_vel, ig_acc=ig_acc, count=num_safe, plot_vel_acc=plot_debug)
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
            print("Check if safe: ", self.safe_mpc_planner.check_safety(X))
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
        # print(hvals)
        # print(hvals >= 0. and hvals <= 0.1)
        # print(0. <= hvals <= 0.1)
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
            if 0.35 <= x <= 0.66:
                if random.random() < 0.5:
                    y = np.random.uniform(0.3, 0.45)
                else:
                    y = np.random.uniform(-0.3, -0.45)
                xt = x + np.random.uniform(0.35 - x, 0.66 - x)
                yt = -y
            else:
                y = np.random.uniform(y_lim[0], y_lim[1])
                if x < 0.35:
                    xt = 0.66 + (0.35 - x)
                else:
                    xt = x - 0.45
                yt = y + np.random.uniform(y_lim[0] - y, y_lim[1] - y)

            z0 = np.random.uniform(0.05, 0.3)
            zt = np.random.uniform(0.05, 0.3)

            # check coordinates are reachable by robot
            if .3 <= np.linalg.norm([x, y, z0]) <= .77 and .3 <= np.linalg.norm([xt, yt, zt]) <= .77:  # .8 should do it
                inRange = True

        xdot = 0
        ydot = 0
        zdot = 0
        print(f"start: [{x:.2f},{y:.2f},{z0:.2f}] \t end: [{xt:.2f},{yt:.2f},{zt:.2f}] ")
        x0 = np.hstack((x, y, z0, xdot, ydot, zdot))
        xt = np.hstack((xt, yt, zt, xdot, ydot, zdot))

        return x0, xt

    def generate_all_trajectories(self, num_demos=5, init_guess_list=None, plot_debug=False):
        labels = []
        x_list = []
        u_list = []
        t_list = []

        start_list = []
        end_list = []
        ref_traj = []

        num_safe = 0
        num_daring = 0
        num_unsafe = 0

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        if self.safe:
            while num_safe < num_demos:
                if init_guess_list is None:
                    x, xt = self.make_trial_conditions()
                    res = self.generate_safe_traj(x, xt)
                elif init_guess_list is not None:
                    # grab file
                    fn_acc, fn_pos, fn_time, fn_vel = init_guess_list[num_safe]
                    initial_guess_time = np.loadtxt(fn_time, delimiter=",")
                    initial_guess_pos = np.loadtxt(fn_pos, delimiter=",")
                    initial_guess_vel = np.loadtxt(fn_vel, delimiter=",")
                    initial_guess_acc = np.loadtxt(fn_acc, delimiter=",")
                    x = np.concatenate((initial_guess_pos[0, 0:3], initial_guess_vel[0, 0:3]))
                    xt = np.concatenate((initial_guess_pos[-1, 0:3], initial_guess_vel[0, 0:3]))
                    res = self.generate_safe_traj(x, xt, ig_time=initial_guess_time, ig_pos=initial_guess_pos,
                                                  ig_vel=initial_guess_vel, ig_acc=initial_guess_acc, num_safe=num_safe,
                                                  plot_debug=plot_debug)
                if res is not None:
                    x_list.append(res[0])
                    u_list.append(res[1])
                    t_list.append(res[2])
                    labels.append('safe')
                    start_list.append(x)
                    end_list.append(xt)
                    if init_guess_list is not None:
                        ref_traj.append(initial_guess_pos)
                    num_safe += 1


            print("Generated Safe Trajectories")

            cmap = plt.cm.get_cmap('Greens')
            crange = np.linspace(0,1,len(x_list) + 2)
            for i in range(len(labels)):
                if labels[i] == 'safe':
                    if init_guess_list is not None:
                        plt.plot(ref_traj[i][:, 0], ref_traj[i][:, 1], ref_traj[i][:, 2], label=f'initial guess #{i + 1}')
                    # Plot Trajectories
                    rgba = cmap(crange[i + 1])
                    plt.plot(x_list[i][:, 0], x_list[i][:, 1], x_list[i][:, 2], c=rgba, label=f'Safe #{i + 1}')

                    # Plot Start and End Points
                    ax.scatter(start_list[i][0], start_list[i][1], start_list[i][2], s=3, c=rgba)
                    ax.scatter(end_list[i][0], end_list[i][1], end_list[i][2], '*', s=5, c=rgba)

        if self.semisafe:
            while num_daring < num_demos:
               x, xt = self.make_trial_conditions()
               res = self.generate_daring_traj(x, xt)
               if res is not None:
                   x_list.append(res[0])
                   u_list.append(res[1])
                   t_list.append(res[2])
                   labels.append('daring')
                   start_list.append(x)
                   end_list.append(xt)

                   num_daring += 1
            print("Generated Daring Trajectories")

            cmap = plt.cm.get_cmap('Blues')
            crange = np.linspace(0, 1, len(x_list) + 2)
            for i in range(len(labels)):
                if labels[i] == 'daring':
                    # Plot Trajectories
                    rgba = cmap(crange[i + 1])
                    plt.plot(x_list[i][:, 0], x_list[i][:, 1], x_list[i][:, 2], c=rgba,
                             label=f'Daring #{i -num_safe+ 1}')

                    # Plot Start and End Points
                    ax.scatter(start_list[i][0], start_list[i][1], start_list[i][2], s=3, c=rgba)
                    ax.scatter(end_list[i][0], end_list[i][1], end_list[i][2], '*', s=5, c=rgba)

        if self.unsafe:
            while num_unsafe < num_demos:
               x, xt = self.make_trial_conditions()
               res = self.generate_unsafe_traj(x, xt)
               if res is not None:
                   x_list.append(res[0])
                   u_list.append(res[1])
                   t_list.append(res[2])
                   labels.append('unsafe')
                   start_list.append(x)
                   end_list.append(xt)
                   num_unsafe += 1
            print("Generated Unsafe Trajectories")
            cmap = plt.cm.get_cmap('Reds')
            crange = np.linspace(0, 1, len(x_list) + 2)
            for i in range(len(labels)):
                if labels[i] == 'unsafe':
                    # Plot Trajectories
                    rgba = cmap(crange[i + 1])
                    plt.plot(x_list[i][:, 0], x_list[i][:, 1], x_list[i][:, 2], c=rgba,
                             label=f'Unsafe #{i -num_safe-num_daring + 1}')

                    # Plot Start and End Points
                    ax.scatter(start_list[i][0], start_list[i][1], start_list[i][2], s=3, c=rgba)
                    ax.scatter(end_list[i][0], end_list[i][1], end_list[i][2], '*', s=5, c=rgba)

        cbf_traj_data = {
            'X': x_list,
            'U': u_list,
            'T': t_list,
            'Labels': labels
        }
        # print(ref_traj)

        # cmap = plt.cm.get_cmap('Greens')
        # crange = np.linspace(0,1,len(safe_x_list) + 2)
        # for i in range(len(safe_x_list)):
        #     if init_guess_list is not None:
        #         plt.plot(ref_traj[i][:, 0], ref_traj[i][:, 1], ref_traj[i][:, 2], label=f'initial guess #{i + 1}')
        #     # Plot Trajectories
        #     rgba = cmap(crange[i + 1])
        #     plt.plot(safe_x_list[i][:, 0], safe_x_list[i][:, 1], safe_x_list[i][:, 2], c=rgba, label=f'Safe #{i + 1}')
        #
        #     # Plot Start and End Points
        #     ax.scatter(safe_start_list[i][0], safe_start_list[i][1], safe_start_list[i][2], s=3, c=rgba)
        #     ax.scatter(safe_end_list[i][0], safe_end_list[i][1], safe_end_list[i][2], '*', s=5, c=rgba)
        #
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)

        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        fig.legend()
        fig.suptitle("Planned trajectories")

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # for i in range(len(safe_x_list)):
        #     if init_guess_list is not None:
        #         plt.plot(ref_traj[i][:,0], ref_traj[i][:,1], ref_traj[i][:,2], label =f'initial guess #{i+1}')
        #     # Plot Trajectories
        #     plt.plot(safe_x_list[i][:,0],safe_x_list[i][:,1], safe_x_list[i][:,2], label=f'planned #{i+1}')
        #
        #     # Plot Start and End Points
        #     ax.scatter(safe_start_list[i][0],safe_start_list[i][1], safe_start_list[i][2], s=3)
        #     ax.scatter(safe_end_list[i][0], safe_end_list[i][1], safe_end_list[i][2], s=3)
        #
        #     ## PLOT RBF centers
        #     if plot_debug :
        #         # colors = np.zeros(self.centers[:, 3:].shape)
        #         # colors[:, 0] = (self.centers[:, 3] - ws_lim[3, 0]) / (ws_lim[3, 1] - ws_lim[3, 0])
        #         # colors[:, 1] = (self.centers[:, 4] - ws_lim[4, 0]) / (ws_lim[4, 1] - ws_lim[4, 0])
        #         # colors[:, 2] = (self.centers[:, 5] - ws_lim[5, 0]) / (ws_lim[5, 1] - ws_lim[5, 0])
        #         # ax.scatter(self.centers[:, 0], self.centers[:, 1], self.centers[:, 2], c=colors, alpha=0.5)
        #         colors = np.zeros(self.centers.shape[0])
        #         for i in range(len(self.centers)):
        #             colors[i] = self.h_fun(self.centers[i])
        #         print("CENTERS SHAPE: ", self.centers.shape, self.centers[0], colors[0], self.theta[0], self.bias)
        #
        #         divnorm = matplotlib.colors.TwoSlopeNorm(vmin=-1,vcenter=0.,vmax=1.)
        #         im = ax.scatter(self.centers[:, 0], self.centers[:, 1], self.centers[:, 2], c=colors, alpha=0.5, norm=divnorm, cmap="RdBu")
        #         fig.colorbar(im)
        #
        #     # print("start: ", x_list[i][0,0:3])
        #     # print("end: ", x_list[i][-1, 0:3])
        # ax.set_xlim(x_lim)
        # ax.set_ylim(y_lim)
        # ax.set_zlim(z_lim)
        # ax.set_xlabel("$x$")
        # ax.set_ylabel("$y$")
        # ax.set_zlabel("$z$")
        # fig.legend()
        # fig.suptitle("Planned trajectories")

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # print(ref_traj)
        # for i in range(len(x_list)):
        #     plt.plot(x_list[i][:,3],x_list[i][:,4], x_list[i][:,5], label='planned vel')
        # ax.set_xlim(xdot_lim)
        # ax.set_ylim(ydot_lim)
        # ax.set_zlim(zdot_lim)
        # ax.set_xlabel("$x$")
        # ax.set_ylabel("$y$")
        # ax.set_zlabel("$z$")
        # fig.legend()
        # fig.suptitle("Planned velocities")
        # plt.show()

        ## PLOTS DEBUG
        if plot_debug:
            for i in range(len(x_list)):
                fig = plt.figure()
                ax = plt.axes()
                plt.plot(t_list[i], x_list[i][:, 3], label='vx')
                plt.plot(t_list[i], x_list[i][:, 4], label='vy')
                plt.plot(t_list[i], x_list[i][:, 5], label='vz')
                fig.legend()
                fig.suptitle(f"Planned velocity #{i+1}")

                fig = plt.figure()
                ax = plt.axes()
                plt.plot(t_list[i][:-1], u_list[i][:, 0], label='ax')
                plt.plot(t_list[i][:-1], u_list[i][:, 1], label='ay')
                plt.plot(t_list[i][:-1], u_list[i][:, 2], label='az')
                fig.legend()
                fig.suptitle(f"Planned acceleration #{i+1}")
        plt.show()

        return cbf_traj_data

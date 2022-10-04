#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
import sys
import pickle

import casadi

from learning_safety_margin.vel_control_utils import *

def rbf_means_stds(X, X_lim, n, k, set_means='uniform', fixed_stds=True, std=0.1, nCenters=None):
    """ Generates means and standard deviations for Gaussian RBF kernel

    Arguments:
        X {numpy array, None} -- Mxn array of inputs (M: number of pts, n: dim of workspace);
                                 None allowed if set_means = {'uniform', 'random'}
        X_lim {numpy array} -- nx2 array of limits (max,min) of workspace dimension (n: dim of workspace)
        n {int} -- Number of workspace dimensions
        k {int} -- Number of kernels
        set_means {string} -- string representing method of determining means.
            Options: {'uniform'(default), 'random', 'inputs'}.
            'uniform': equally spaced k points across workspace
            'random': randomly generated k points across workspace
            'input': directly use the first k input points (data points) as means (ideally k=M)
            TODO: 'kmeans': use kmeans on input points (data points) to generate
        fixed_stds {bool} -- set if fixed for all means or randomized

    Returns:
        means- numpy array -- A kxn array of final means/centers
        stds - numpy array -- A kx1 array of final stds
    """
    set_means_options = ['uniform', 'random', 'inputs']
    assert set_means in set_means_options, "Invalid option for set_means"

    # Generate means
    if set_means == 'uniform':
        if n == 1:
            means = np.linspace(start=X_lim[0], stop=X_lim[1],
                                num=k, endpoint=True)
        elif n == 2:
            x = np.linspace(start=X_lim[0, 0], stop=X_lim[0, 1],
                            num=k, endpoint=True)
            y = np.linspace(start=X_lim[1, 0], stop=X_lim[1, 1],
                            num=k, endpoint=True)
            XX, YY = np.meshgrid(x, y)
            means = np.array([XX.flatten(), YY.flatten()]).T
        else:
            pts = []
            for i in range(X_lim.shape[0]):
                pts.append(np.linspace(start=X_lim[i, 0], stop=X_lim[i, 1], num=k, endpoint=True))
            pts = np.array(pts)
            pts = tuple(pts)
            D = np.meshgrid(*pts)
            means = np.array([D[i].flatten() for i in range(len(D))]).T

    if set_means == 'random':
        # key = random.PRNGKey(0)
        means = np.random.uniform(low=X_lim[:, 0], high=X_lim[:, 1], size=(nCenters, X_lim.shape[0]))
    if set_means == 'inputs':
        assert X is not None, 'X invalid data input. Cannot be None-type'
        assert k == X.shape[0], 'Set_means inputs, num kernels must equal num of data points'
        means = X.copy()

    # Generate stds
    if fixed_stds == True:
        stds = np.ones(means.shape[0]) * std
    else:
        #         stds = np.random.uniform(low = 0.0001, high = std, size=(k**n,1))
        stds = random.uniform(rng.next(), minval=0.0001, maxval=std, shape=(k ** n, 1))
    stds = np.squeeze(stds)
    return means, stds

# Generate Casadi Functions for h(x) functions

class PlotCBF():
    def __init__(self, theta, bias, centers=None, stds =None):
        self.theta = theta
        self.bias = bias
        # Set up variables
        self.x = casadi.SX.sym('x', 3)
        self.u = casadi.SX.sym('u', 3)
        if centers is None:
            self.centers, self.stds = rbf_means_stds(X=None, X_lim=np.array([x_lim, y_lim, z_lim, vdot_lim, vdot_lim, vdot_lim]),
                                           n=x_dim, k=n_dim_features, fixed_stds=True, std=rbf_std)

        else:
            self.centers = centers
            self.stds = stds
        # Generate Casadi Functions for h(x) functions
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
        self.norm = colors.TwoSlopeNorm(vmin=-1,vcenter=0.,vmax=1.)


    def plot_xy_pos(self, z=0.25, xdot=0.1, ydot=0.1, zdot=0.1, num_pts=11):
        x = np.linspace(x_lim[0], x_lim[1], num=num_pts)
        y = np.linspace(y_lim[0], y_lim[1], num=num_pts)

        xx, yy = np.meshgrid(x, y)
        hvals = np.zeros(xx.shape)
        xvec = xx.ravel()
        yvec = yy.ravel()
        hvec = hvals.ravel()
        for i in range(len(xvec)):
            hvec[i] = self.h_fun([xvec[i],yvec[i],z])
        hvals = hvec.reshape((num_pts, num_pts))
        # divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))
        divnorm = colors.TwoSlopeNorm(vmin=-3,vcenter=0.,vmax=3.)

        fig,ax = plt.subplots()
        im = ax.imshow(hvals, extent=[x_lim[0],x_lim[1], y_lim[0], y_lim[1]], origin='lower',norm=self.norm,cmap=cm.coolwarm_r)
        fig.colorbar(im)
        ax.set_xlabel('$x$', fontsize=18)
        ax.set_ylabel('$y$', fontsize=18)
        plt.show()

    def plot_xz_pos(self, y=0., xdot=0.1, ydot=0.1, zdot=0.1, num_pts=11):
        x = np.linspace(x_lim[0], x_lim[1], num=num_pts)
        z = np.linspace(z_lim[0], z_lim[1], num=num_pts)

        xx, zz = np.meshgrid(x, z)
        hvals = np.zeros(xx.shape)
        xvec = xx.ravel()
        zvec = zz.ravel()
        hvec = hvals.ravel()
        for i in range(len(xvec)):
            hvec[i] = self.h_fun([xvec[i],y,zvec[i]])
        hvals = hvec.reshape((num_pts, num_pts))
        # divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))

        fig,ax = plt.subplots()
        im = ax.imshow(hvals, extent=[x_lim[0],x_lim[1], z_lim[0], z_lim[1]], origin='lower',norm=self.norm,cmap=cm.coolwarm_r)
        fig.colorbar(im)
        ax.set_xlabel('$x$', fontsize=18)
        ax.set_ylabel('$z$', fontsize=18)
        plt.show()

    def plot_yz_pos(self, x=0.5, xdot=0.1, ydot=0.1, zdot=0.1, num_pts=11):
        y = np.linspace(y_lim[0], y_lim[1], num=num_pts)
        z = np.linspace(z_lim[0], z_lim[1], num=num_pts)

        yy, zz = np.meshgrid(y, z)
        hvals = np.zeros(yy.shape)
        yvec = yy.ravel()
        zvec = zz.ravel()
        hvec = hvals.ravel()
        for i in range(len(yvec)):
            hvec[i] = self.h_fun([x, yvec[i],zvec[i]])
        hvals = hvec.reshape((num_pts, num_pts))
        # divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))
        divnorm = colors.TwoSlopeNorm(vmin=-3,vcenter=0.,vmax=3.)

        fig,ax = plt.subplots()
        im = ax.imshow(hvals, extent=[y_lim[0],y_lim[1], z_lim[0], z_lim[1]], origin='lower',norm=self.norm,cmap=cm.coolwarm_r)
        fig.colorbar(im)
        ax.set_xlabel('$y$', fontsize=18)
        ax.set_ylabel('$z$', fontsize=18)
        plt.show()


    def plot_xyz(self, xdot = .1, ydot=0.1, zdot=0.1, num_pts = 11):
        x1 = np.linspace(x_lim[0], x_lim[1], num=num_pts)
        x2 = np.linspace(y_lim[0], y_lim[1], num=num_pts)
        x3 = np.linspace(z_lim[0], z_lim[1], num=num_pts)

        xx, yy, zz = np.meshgrid(x1, x2, x3)
        hvals = np.zeros(xx.shape)
        xvec = xx.ravel()
        yvec = yy.ravel()
        zvec = zz.ravel()
        hvec = hvals.ravel()
        for i in range(len(xvec)):
            hvec[i] = self.h_fun([xvec[i], yvec[i], zvec[i]])

        hvals = hvec.reshape((num_pts, num_pts, num_pts))

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        for i in range(len(xvec)):
            im = ax.scatter(xvec[i], yvec[i], zvec[i], c=hvec[i], norm=self.norm, cmap=cm.coolwarm_r)  # , marker=m)

        ax.set_xlabel('$x$', fontsize=18)
        ax.set_ylabel('$y$', fontsize=18)
        ax.set_zlabel('$z$', fontsize=18)

        ax.set_title('Learned CBF')
        ax.view_init(10, 180)
        plt.show()

    def plot_xyz_neg(self, xdot = .1, ydot=0.1, zdot=0.1, num_pts = 11):
        x1 = np.linspace(x_lim[0], x_lim[1], num=num_pts)
        x2 = np.linspace(y_lim[0], y_lim[1], num=num_pts)
        x3 = np.linspace(z_lim[0], z_lim[1], num=num_pts)

        xx, yy, zz = np.meshgrid(x1, x2, x3)
        hvals = np.zeros(xx.shape)
        xvec = xx.ravel()
        yvec = yy.ravel()
        zvec = zz.ravel()
        hvec = hvals.ravel()
        for i in range(len(xvec)):
            hvec[i] = self.h_fun([xvec[i], yvec[i], zvec[i]])

        hvals = hvec.reshape((num_pts, num_pts, num_pts))

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        for i in range(len(xvec)):
            if hvec[i] < 0:
                im = ax.scatter(xvec[i], yvec[i], zvec[i], c=hvec[i], norm=self.norm, cmap=cm.coolwarm_r)  # , marker=m)
        ax.set_xlabel('$x$', fontsize=18)
        ax.set_ylabel('$y$', fontsize=18)
        ax.set_zlabel('$z$', fontsize=18)

        ax.set_title('Learned CBF')
        ax.view_init(10, 180)

        plt.show()

if __name__ == '__main__':

    # Check passed argument - User number
    if len(sys.argv) >= 2:
        user_number = sys.argv[1]
    else:
        user_number = '0'

    print("Running CBF Plotting for User_"+user_number+"\n")

    # Get data
    data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_"+user_number+"/"

    data = pickle.load(open(data_dir + "pos_data_dict.p", "rb"))
    print(data.keys())
    # Set up data for MPC planner
    params = data["theta"]
    bias_param = data["bias"]
    slack_param = data["unsafe_slack"]
    centers = data["rbf_centers"]
    stds = data["rbf_stds"]
    bias_param = 0.1

    plotter = PlotCBF(params, bias_param, centers, stds)
    plotter.plot_xy_pos(num_pts=30)
    plotter.plot_xz_pos(num_pts=30)
    plotter.plot_yz_pos(num_pts=30)

    plotter.plot_xyz()
    plotter.plot_xyz_neg(num_pts=30)
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
import sys
import pickle
import glob
import random
# from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

import casadi

from functools import partial
# from jax import jit
# from jax import random, vmap, jit, grad, ops, lax, tree_util, device_put, device_get, jacobian, jacfwd, jacrev, jvp
# import jax.numpy as np
# import jax
from learning_safety_margin.vel_control_utils import *
# from learning_cbf_vel_lim import h_model, x_lim, y_lim, z_lim, vdot_lim

# Velocity Limits
# x_lim = [0., 1.]
# y_lim = [-0.5, 0.5]
# z_lim = [0., 0.5]
# vdot_lim = [-1., 1.]
#
# ws_lim = np.vstack((x_lim, y_lim, z_lim, vdot_lim, vdot_lim, vdot_lim))
#
# Define h model (i.e., RBF)

# def alpha(x):
#     return psi * x
#
#
# def rbf(x, c, s):
#     return np.exp(-1 / (2 * s ** 2) * np.linalg.norm(np.asarray(x) - c) ** 2)
#
# @vmap
# def phi(X):
#     a = np.array([rbf(X, c, s) for c, s, in zip(centers, stds)])
#     return a  # np.array(y)
#
# @jax.jit
# # @partial(jit, static_argnums=(0,))
# def h_model(x, theta, bias):
#     #     print(x.shape, phi(x).shape, theta.shape, phi(x).dot(theta).shape, bias.shape)
#     #     print(x.type, theta.type, bias.type)
#     return phi(x).dot(theta) + bias

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
    def __init__(self, theta, bias, centers=None, stds =None, data_dir=None):
        self.theta = theta
        self.bias = bias
        # Set up variables
        self.x = casadi.SX.sym('x', 6)
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
        self.data_dir = data_dir


    def plot_xy_pos(self, z=0.25, xdot=0.1, ydot=0.1, zdot=0.1, num_pts=11):
        x = np.linspace(x_lim[0], x_lim[1], num=num_pts)
        y = np.linspace(y_lim[0], y_lim[1], num=num_pts)

        xx, yy = np.meshgrid(x, y)
        hvals = np.zeros(xx.shape)
        xvec = xx.ravel()
        yvec = yy.ravel()
        hvec = hvals.ravel()
        for i in range(len(xvec)):
            hvec[i] = self.h_fun([xvec[i],yvec[i],z,xdot, ydot, zdot])

        print(hvec)
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
            hvec[i] = self.h_fun([xvec[i],y,zvec[i],xdot, ydot, zdot])
        hvals = hvec.reshape((num_pts, num_pts))
        # divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))

        fig,ax = plt.subplots()
        im = ax.imshow(hvals, extent=[x_lim[0],x_lim[1], z_lim[0], z_lim[1]], origin='lower',norm=self.norm,cmap=cm.coolwarm_r)
        fig.colorbar(im)
        ax.set_xlabel('$x$', fontsize=18)
        ax.set_ylabel('$z$', fontsize=18)
        plt.show()

    def plot_xz_pos_multiple(self, num_slices=9, xdot=0.1, ydot=0.1, zdot=0.1, num_pts=11):
        x = np.linspace(x_lim[0], x_lim[1], num=num_pts)
        z = np.linspace(z_lim[0], z_lim[1], num=num_pts)
        y = np.linspace(y_lim[0], y_lim[1], num=num_slices)

        xx, zz = np.meshgrid(x, z)
        hvals = np.zeros(xx.shape)
        xvec = xx.ravel()
        zvec = zz.ravel()
        hvec = hvals.ravel()

        # divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))

        fig,axs = plt.subplots(3,3)
        fig.suptitle("CBF centers as XZ slices for multiple y values")

        for i, ax in enumerate(axs.ravel()):

            for k in range(len(xvec)):
                hvec[k] = self.h_fun([xvec[k], y[i], zvec[k], xdot, ydot, zdot])
            hvals = hvec.reshape((num_pts, num_pts))

            im = ax.imshow(hvals, extent=[x_lim[0], x_lim[1], z_lim[0], z_lim[1]], origin='lower', norm=self.norm,
                           cmap=cm.coolwarm_r)
            ax.set_xlabel('$x$', fontsize=6)
            ax.set_ylabel('$z$', fontsize=6)
            ax.set_title("y = {:0.2f}".format(y[i]), fontsize=10)

        fig.colorbar(im)

        plt.show()

    def plot_xz_pos_animate(self, xdot=0.1, ydot=0.1, zdot=0.1, num_slices=20, num_pts=11):
        """
        Plots a loop animation of the cbf function for an xz slice in velocity space with an increasing y values,
        in a given xyz position.

        Args:
            x: Position coordinates in x
            y: Position coordinates in y
            z: Position coordinates in z
            num_slices: number steps in the animation (ie number of y values to plot)
            num_pts: number of points to plot in xz velocity space

        """
        fig, ax = plt.subplots()

        ani = FuncAnimation(fig, self.xz_pos_animate, fargs=(xdot,ydot,zdot,num_slices, num_pts, ax), frames=20, interval=500)

        plt.show()
    def xz_pos_animate(self, frame_number, xdot, ydot, zdot, num_slices, num_pts, ax):
        """
        Animation function for 'plot_xz_vel_animate'
        """
        x = np.linspace(x_lim[0], x_lim[1], num=num_pts)
        z = np.linspace(z_lim[0], z_lim[1], num=num_pts)
        y = np.linspace(y_lim[0], y_lim[1], num=num_slices)

        xx, zz = np.meshgrid(x, z)
        hvals = np.zeros(xx.shape)
        xvec = xx.ravel()
        zvec = zz.ravel()
        hvec = hvals.ravel()
        for i in range(len(xvec)):
            hvec[i] = self.h_fun([xvec[i], y[frame_number], zvec[i], xdot, ydot, zdot])
        hvals = hvec.reshape((num_pts, num_pts))

        ax.clear()
        im = ax.imshow(hvals, extent=[x_lim[0], x_lim[1], z_lim[0], z_lim[1]], origin='lower',
                       norm=self.norm, cmap=cm.coolwarm_r)
        ax.set_xlabel('$x$', fontsize=18)
        ax.set_ylabel('$z$', fontsize=18)
        # ax.set_title('XZ position slice in [%s, %s ,%s] for y_dot=%0.2f' % (x, y[frame_number], z))
        # fig.colorbar(im)

    def plot_yz_pos(self, x=0.5, xdot=0.1, ydot=0.1, zdot=0.1, num_pts=11):
        y = np.linspace(y_lim[0], y_lim[1], num=num_pts)
        z = np.linspace(z_lim[0], z_lim[1], num=num_pts)

        yy, zz = np.meshgrid(y, z)
        hvals = np.zeros(yy.shape)
        yvec = yy.ravel()
        zvec = zz.ravel()
        hvec = hvals.ravel()
        for i in range(len(yvec)):
            hvec[i] = self.h_fun([x, yvec[i],zvec[i],xdot, ydot, zdot])
        hvals = hvec.reshape((num_pts, num_pts))
        # divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))
        divnorm = colors.TwoSlopeNorm(vmin=-3,vcenter=0.,vmax=3.)

        fig,ax = plt.subplots()
        im = ax.imshow(hvals, extent=[y_lim[0],y_lim[1], z_lim[0], z_lim[1]], origin='lower',norm=self.norm,cmap=cm.coolwarm_r)
        fig.colorbar(im)
        ax.set_xlabel('$y$', fontsize=18)
        ax.set_ylabel('$z$', fontsize=18)
        plt.show()


    def plot_xy_vel(self, x=0.5, y=0., z=0.25, zdot=0., num_pts=11):
        xdot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
        ydot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)

        xx, yy = np.meshgrid(xdot, ydot)
        hvals = np.zeros(xx.shape)
        xdvec = xx.ravel()
        ydvec = yy.ravel()
        hvec = hvals.ravel()
        for i in range(len(xdvec)):
            hvec[i] = self.h_fun([x,y,z,xdvec[i], ydvec[i], zdot])
        hvals = hvec.reshape((num_pts, num_pts))
        # divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))
        divnorm = colors.TwoSlopeNorm(vmin=-3,vcenter=0.,vmax=3.)

        fig,ax = plt.subplots()
        im = ax.imshow(hvals, extent=[vdot_lim[0],vdot_lim[1], vdot_lim[0], vdot_lim[1]], origin='lower',norm=self.norm,cmap=cm.coolwarm_r)
        fig.colorbar(im)
        ax.set_xlabel('$\dot{x}$', fontsize=18)
        ax.set_ylabel('$\dot{y}$', fontsize=18)
        plt.show()

    def plot_xz_vel(self, x=0.5, y=0., z=0.25, ydot=0., num_pts=11):
        xdot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
        zdot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)

        xx, zz = np.meshgrid(xdot, zdot)
        hvals = np.zeros(xx.shape)
        xdvec = xx.ravel()
        zdvec = zz.ravel()
        hvec = hvals.ravel()
        for i in range(len(xdvec)):
            hvec[i] = self.h_fun([x,y,z,xdvec[i], ydot, zdvec[i]])
        hvals = hvec.reshape((num_pts, num_pts))
        # divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))
        divnorm = colors.TwoSlopeNorm(vmin=-3,vcenter=0.,vmax=3.)

        fig,ax = plt.subplots()
        im = ax.imshow(hvals, extent=[vdot_lim[0],vdot_lim[1], vdot_lim[0], vdot_lim[1]], origin='lower',norm=self.norm,cmap=cm.coolwarm_r)
        fig.colorbar(im)
        ax.set_xlabel('$\dot{x}$', fontsize=18)
        ax.set_ylabel('$\dot{z}$', fontsize=18)
        plt.show()

    def plot_xz_vel_animate(self, x=0.5, y=0., z=0.25, num_slices=20, num_pts=11):
        """
        Plots a loop animation of the cbf function for an xz slice in velocity space with an increasing y values,
        in a given xyz position.

        Args:
            x: Position coordinates in x
            y: Position coordinates in y
            z: Position coordinates in z
            num_slices: number steps in the animation (ie number of y values to plot)
            num_pts: number of points to plot in xz velocity space

        """
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, self.xz_vel_animate, fargs=(x,y,z,num_slices, num_pts, ax), frames=20, interval=500)
        plt.show()

    def xz_vel_animate(self, frame_number, x, y, z, num_slices, num_pts, ax):
        """
        Animation function for 'plot_xz_vel_animate'
        """
        xdot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
        zdot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
        ydot = np.linspace(y_lim[0], y_lim[1], num=num_slices)

        xx, zz = np.meshgrid(xdot, zdot)
        hvals = np.zeros(xx.shape)
        xdvec = xx.ravel()
        zdvec = zz.ravel()
        hvec = hvals.ravel()
        for i in range(len(xdvec)):
            hvec[i] = self.h_fun([x, y, z, xdvec[i], ydot[frame_number], zdvec[i]])
        hvals = hvec.reshape((num_pts, num_pts))

        ax.clear()
        im = ax.imshow(hvals, extent=[vdot_lim[0], vdot_lim[1], vdot_lim[0], vdot_lim[1]], origin='lower',
                       norm=self.norm, cmap=cm.coolwarm_r)
        ax.set_xlabel('$\dot{x}$', fontsize=18)
        ax.set_ylabel('$\dot{z}$', fontsize=18)
        ax.set_title('XZ velocity slice in [%s, %s ,%s] for y_dot=%0.2f' %(x,y,z, ydot[frame_number]))
        # fig.colorbar(im)

    def plot_yz_vel(self, x=0.5, y=0., z=0.25, xdot=0., num_pts=11):
        ydot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
        zdot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)

        yy, zz = np.meshgrid(ydot, zdot)
        hvals = np.zeros(yy.shape)
        ydvec = yy.ravel()
        zdvec = zz.ravel()
        hvec = hvals.ravel()
        for i in range(len(ydvec)):
            hvec[i] = self.h_fun([x, y, z, xdot, ydvec[i], zdvec[i]])
        hvals = hvec.reshape((num_pts, num_pts))
        # divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))
        divnorm = colors.TwoSlopeNorm(vmin=-3,vcenter=0.,vmax=3.)

        fig, ax = plt.subplots()
        im = ax.imshow(hvals, extent=[vdot_lim[0], vdot_lim[1], vdot_lim[0], vdot_lim[1]], origin='lower', norm=self.norm,
                       cmap=cm.coolwarm_r)
        fig.colorbar(im)
        ax.set_xlabel('$\dot{y}$', fontsize=18)
        ax.set_ylabel('$\dot{z}$', fontsize=18)
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
            hvec[i] = self.h_fun([xvec[i], yvec[i], zvec[i], xdot, ydot, zdot])

        hvals = hvec.reshape((num_pts, num_pts, num_pts))

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        for i in range(len(xvec)):
            im = ax.scatter(xvec[i], yvec[i], zvec[i], c=hvec[i], norm=self.norm, cmap=cm.coolwarm_r)  # , marker=m)

        # add isosurface
        # dx = x1[1] - x1[0]
        # dy = x2[1] - x2[0]
        # dz = x3[1] - x3[0]
        # verts, faces, _, _ = marching_cubes(hvals, 0, spacing=(dx, dy, dz), step_size=2)
        # # verts *= np.array([dx, dy, dz])
        # # verts -= np.array([x_lim[0], y_lim[0], z_lim[0]])
        # # add as Poly3DCollection
        # mesh = Poly3DCollection(verts[faces])
        # mesh.set_facecolor('g')
        # mesh.set_edgecolor('none')
        # mesh.set_alpha(0.3)
        # ax.add_collection3d(mesh)

        ax.set_xlabel('$x$', fontsize=18)
        ax.set_ylabel('$y$', fontsize=18)
        ax.set_zlabel('$z$', fontsize=18)
        fig.colorbar(im)
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
            hvec[i] = self.h_fun([xvec[i], yvec[i], zvec[i], xdot, ydot, zdot])

        hvals = hvec.reshape((num_pts, num_pts, num_pts))

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        for i in range(len(xvec)):
            if hvec[i] < 0:
                im = ax.scatter(xvec[i], yvec[i], zvec[i], c=hvec[i], norm=self.norm, cmap=cm.coolwarm_r)  # , marker=m)

        # add isosurface
        # dx = x1[1] - x1[0]
        # dy = x2[1] - x2[0]
        # dz = x3[1] - x3[0]
        # verts, faces, _, _ = marching_cubes(hvals, 0, spacing=(1, 1, 1), step_size=2)
        # verts *= np.array([dx, dy, dz])
        # verts += np.array([x_lim[0], y_lim[0], z_lim[0]])
        # # add as Poly3DCollection
        # mesh = Poly3DCollection(verts[faces])
        # mesh.set_facecolor('g')
        # mesh.set_edgecolor('none')
        # mesh.set_alpha(0.3)
        # ax.add_collection3d(mesh)

        ax.set_xlabel('$x$', fontsize=18)
        ax.set_ylabel('$y$', fontsize=18)
        ax.set_zlabel('$z$', fontsize=18)
        fig.colorbar(im)
        ax.set_title('Learned CBF neg')
        ax.view_init(10, 180)

        plt.show()

    def plot_xyz_neg_animate(self,  xdot=0.1, ydot=0.1, zdot=0.1, num_slices=11, num_pts=11):
        """
        Plots a loop animation of the cbf function for an xz slice in velocity space with an increasing y values,
        in a given xyz position.

        Args:
            x: Position coordinates in x
            y: Position coordinates in y
            z: Position coordinates in z
            num_slices: number steps in the animation (ie number of y values to plot)
            num_pts: number of points to plot in xz velocity space

        """
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')

        ani = FuncAnimation(fig, self.xyz_neg_animate, fargs=(xdot, ydot, zdot, num_slices, num_pts, ax), frames=20, interval=100, repeat=False)

        plt.show()
    def xyz_neg_animate(self, frame_number, xdot, ydot, zdot, num_slices, num_pts, ax):

        x1 = np.linspace(x_lim[0], x_lim[1], num=num_pts)
        x2 = np.linspace(y_lim[0], y_lim[1], num=num_pts)
        x3 = np.linspace(z_lim[0], z_lim[1], num=num_slices)

        xx, yy, zz = np.meshgrid(x1, x2, x3)
        hvals = np.zeros(xx.shape)
        xvec = xx.ravel()
        yvec = yy.ravel()
        zvec = zz.ravel()
        hvec = hvals.ravel()
        for i in range(len(xvec)):
            hvec[i] = self.h_fun([xvec[i], yvec[i], zvec[frame_number], xdot, ydot, zdot])

        # hvals = hvec.reshape((num_pts, num_pts, num_pts))
        neg_ind = np.where(hvec < 0.)
        # print(neg_ind)
        ax.clear()
        im = ax.scatter3D(xvec[neg_ind], yvec[neg_ind], zvec[neg_ind], c=hvec[neg_ind], norm=self.norm, cmap=cm.coolwarm_r)  # , marker=m)
        ax.set_xlabel('$x$', fontsize=18)
        ax.set_ylabel('$y$', fontsize=18)
        ax.set_zlabel('$z$', fontsize=18)
        # fig.colorbar(im)


    def plot_xyz_animate(self, xdot=0.1, ydot=0.1, zdot=0.1, num_slices=11, num_pts=11):
        """
        Plots a loop animation of the cbf function for an xz slice in velocity space with an increasing y values,
        in a given xyz position.

        Args:
            x: Position coordinates in x
            y: Position coordinates in y
            z: Position coordinates in z
            num_slices: number steps in the animation (ie number of y values to plot)
            num_pts: number of points to plot in xz velocity space

        """
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')

        ani = FuncAnimation(fig, self.xyz_animate, fargs=(xdot, ydot, zdot, num_slices, num_pts, ax), frames=20, interval=100, repeat=False)

        plt.show()
    def xyz_animate(self, frame_number, xdot, ydot, zdot, num_slices, num_pts, ax):

        x1 = np.linspace(x_lim[0], x_lim[1], num=num_pts)
        x2 = np.linspace(y_lim[0], y_lim[1], num=num_pts)
        x3 = np.linspace(z_lim[0], z_lim[1], num=num_slices)

        xx, yy, zz = np.meshgrid(x1, x2, x3)
        hvals = np.zeros(xx.shape)
        xvec = xx.ravel()
        yvec = yy.ravel()
        zvec = zz.ravel()
        hvec = hvals.ravel()
        for i in range(len(xvec)):
            hvec[i] = self.h_fun([xvec[i], yvec[i], zvec[frame_number], xdot, ydot, zdot])

        hvals = hvec.reshape((num_pts, num_pts, num_pts))

        ax.clear()
        im = ax.scatter3D(xvec, yvec, zvec, c=hvec, norm=self.norm, cmap=cm.coolwarm_r)  # , marker=m)
        ax.set_xlabel('$x$', fontsize=18)
        ax.set_ylabel('$y$', fontsize=18)
        ax.set_zlabel('$z$', fontsize=18)
        # fig.colorbar(im)
    def select_random_traj(self, num_traj):
        """
        Selects random trajectories from the provided User directory
        Args:
            num_traj: number of trajectories to grab per safety category

        Returns:
            A list of path to each trajectory end effector cartesian position file
        """
        traj_nbr_per_category = int(num_traj)
        category_list = ["safe/", "daring/", "unsafe/"]
        fpath_list = []

        for category in category_list:
            ## count nbr of traj in category folder, make a list
            all_fn = glob.glob(self.data_dir + "csv/" + category + "*_eePosition.txt")
            # Sample trajectories from category
            fpath_list.append(random.sample(all_fn, traj_nbr_per_category))

        # Reshape list to be 1D
        return sum(fpath_list, [])

    def plot_3D_quiver_uniform(self, num_pts=6, num_arrows=3, random_points=False, alpha=0.75):

        """
        This function plots the CBF h function over a 6D state space of positions and velocities.
        Positions are represented in a 3D plot and for each position, quiver arrows represent the velocity in different
        directions.
        The coordinates of the positions is generated using either a linspace or random normal distribution.
        The color gradient of the arrows represents the h_function in each direction

        Args:
            num_pts: number of points per position dimension -> total number of points in plot is num_pts^3
            num_arrows: number of quiver arrows per velocity dimension -> total number of arrows per point is num_arrows^3
            random_points: Whether to use random or uniform point generation for 3D positions
            alpha: alpha value of the quiver arrows, used for visibility

        Note :
            num_pts and num_arrows is identical in each dimension -> different values for each dimension might provide
            better readability in plots

        """

        # Random distribution
        if random_points:
            x_mu = (x_lim[0]+x_lim[1])/2
            x_sig = x_lim[1] - x_mu
            x1 = np.random.normal(x_mu, x_sig, num_pts)
            y_mu = (y_lim[0]+y_lim[1])/2
            y_sig = y_lim[1] - y_mu
            x2 = np.random.normal(y_mu, y_sig, num_pts)
            z_mu = (z_lim[0]+z_lim[1])/2
            z_sig = z_lim[1] - z_mu
            x3 = np.random.normal(z_mu, z_sig, num_pts)
        elif not random_points:
            x1 = np.linspace(x_lim[0], x_lim[1], num=num_pts)
            x2 = np.linspace(y_lim[0], y_lim[1], num=num_pts)
            x3 = np.linspace(z_lim[0], z_lim[1], num=num_pts)

        xdot = np.linspace(xdot_lim[0], xdot_lim[1], num=num_arrows)
        ydot = np.linspace(ydot_lim[0], ydot_lim[1], num=num_arrows)
        zdot = np.linspace(zdot_lim[0], zdot_lim[1], num=num_arrows)

        xx, yy, zz = np.meshgrid(x1, x2, x3)
        xvec = xx.ravel()
        yvec = yy.ravel()
        zvec = zz.ravel()

        xxdot, yydot, zzdot = np.meshgrid(xdot, ydot, zdot)
        hvals = np.zeros(xxdot.shape)
        xdotvec= xxdot.ravel()
        ydotvec = yydot.ravel()
        zdotvec = zzdot.ravel()
        hvec = hvals.ravel()

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        # ax = fig.add_subplot(projection='3d')
        for i in range(len(xvec)):
            xvec_rep = np.repeat(xvec[i], len(xdotvec))
            yvec_rep = np.repeat(yvec[i], len(xdotvec))
            zvec_rep = np.repeat(zvec[i], len(xdotvec))

            for k in range(len(xdotvec)):
                hvec[k] = self.h_fun([xvec[i], yvec[i], zvec[i], xdotvec[k], ydotvec[k], zdotvec[k]])

            # Flatten and normalize
            c = (hvec.ravel() - hvec.min()) / hvec.ptp()
            # Repeat for each body line and two head lines
            c = np.concatenate((c, np.repeat(c, 2)))
            # Colormap
            c = plt.cm.coolwarm_r(c)
            # Set alpha
            c[:, -1] = np.repeat(alpha, len(c[:, 0]))

            im_quiver = ax.quiver3D(xvec_rep, yvec_rep, zvec_rep, xdotvec, ydotvec, zdotvec, colors=c,
                                     normalize=True, length=0.05, cmap=cm.coolwarm_r, norm=self.norm)

        im = ax.scatter3D(xvec, yvec, zvec, s=10)

        ax.set_xlabel('$x$', fontsize=18)
        ax.set_ylabel('$y$', fontsize=18)
        ax.set_zlabel('$z$', fontsize=18)
        fig.colorbar(im_quiver)
        ax.set_title('Learned CBF with quiver')
        ax.view_init(10, 180)
        plt.show()

    def plot_3D_quiver_from_traj(self, num_pts=6, num_arrows=3, num_traj=3, alpha=0.75):#
        """
        This function plots the CBF h function over a 6D state space of positions and velocities.
        Positions are represented in a 3D plot and for each position, quiver arrows represent the velocity in different
        directions.
        The coordinates of the positions is based on the User's recorded trajectories
        The color gradient of the arrows represents the h_function in each direction.

        Args:
            num_pts: The number of positions points to use for each trajectory (points for which to plot velocities)
            num_arrows: The number of quiver arrows per velocity dimension -> total number of arrows per point is num_arrows^3
            num_traj: The number of trajectories selected per category
            alpha: the alpha values of the quiver arrows

        Note : Same as above, different num_arrows for each dimension might yield improved visibility
            Improving subsampling of points to use for quiver offline_plotting might avoid clusters/improve readability

        """
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')

        fpath_list = self.select_random_traj(num_traj)

        for fpath in fpath_list:
            traj_pos = np.loadtxt(fpath, delimiter=',')[:, 0:3]

            nb_sample_points = 100
            step = round(len(traj_pos[:,0])/nb_sample_points)
            pos_to_plot = traj_pos[::step]

            step_for_quiv = round(len(traj_pos[:,0])/num_pts)
            pos_for_mesh = traj_pos[::step_for_quiv]

            xdot = np.linspace(xdot_lim[0], xdot_lim[1], num=num_arrows)
            ydot = np.linspace(ydot_lim[0], ydot_lim[1], num=num_arrows)
            zdot = np.linspace(zdot_lim[0], zdot_lim[1], num=num_arrows)

            xvec = pos_for_mesh[:,0]
            yvec = pos_for_mesh[:,1]
            zvec = pos_for_mesh[:,2]

            xxdot, yydot, zzdot = np.meshgrid(xdot, ydot, zdot)
            hvals = np.zeros(xxdot.shape)
            xdotvec = xxdot.ravel()
            ydotvec = yydot.ravel()
            zdotvec = zzdot.ravel()
            hvec = hvals.ravel()

            for i in range(len(xvec)):
                xvec_rep = np.repeat(xvec[i], len(xdotvec))
                yvec_rep = np.repeat(yvec[i], len(xdotvec))
                zvec_rep = np.repeat(zvec[i], len(xdotvec))

                for k in range(len(xdotvec)):
                    hvec[k] = self.h_fun([xvec[i], yvec[i], zvec[i], xdotvec[k], ydotvec[k], zdotvec[k]])

                print(hvec)

                # Flatten and normalize
                # c = (hvec.ravel() - hvec.min()) / hvec.ptp()
                c = self.norm(hvec.ravel())
                # Repeat for each body line and two head lines
                c = np.concatenate((c, np.repeat(c, 2)))
                # Colormap
                c = plt.cm.coolwarm_r(c)
                # Set alpha
                c[:, -1] = np.repeat(alpha, len(c[:, 0]))

                # im_quiver = ax.quiver3D(xvec_rep, yvec_rep, zvec_rep, xdotvec, ydotvec, zdotvec, colors=c,
                #                         normalize=True, length=0.04, cmap=cm.coolwarm_r)#, norm=self.norm)
                im_quiver = ax.quiver(xvec_rep, yvec_rep, zvec_rep, xdotvec, ydotvec, zdotvec, colors=c, cmap=cm.coolwarm_r, length=0.04,normalize=True , norm=self.norm)

            # Set color
            if '/safe/' in fpath:
                traj_color = 'g'
                traj_label = 'safe'
            elif '/daring/' in fpath:
                traj_color = 'b'
                traj_label = 'daring'
            elif 'unsafe' in fpath:
                traj_color = 'r'
                traj_label = 'unsafe'
            else:
                traj_color = 'k'
                traj_label = 'unknown'
            ax.plot(pos_to_plot[:,0], pos_to_plot[:,1], pos_to_plot[:,2], traj_color, label=traj_label)

        ax.set_xlabel('$x$', fontsize=18)
        ax.set_ylabel('$y$', fontsize=18)
        ax.set_zlabel('$z$', fontsize=18)
        fig.colorbar(im_quiver)
        # Legend without duplicates
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))
        ax.set_title('Learned CBF with quiver on recorded trajectories')
        ax.view_init(10, 180)
        plt.show()

    def plot_centers_hvals(self):
        colors = np.zeros(self.centers.shape[0])
        for i in range(len(self.centers)):
            colors[i] = self.h_fun(self.centers[i])
        print("CENTERS SHAPE: ", self.centers.shape, self.centers[0], colors[0], self.theta[0], self.bias)

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # divnorm = cm.TwoSlopeNorm(vcenter=0.)
        im = ax.scatter(self.centers[:, 0], self.centers[:, 1], self.centers[:, 2], c=colors, alpha=0.5, norm=self.norm,
                        cmap="RdBu")

        # print("start: ", x_list[i][0,0:3])
        # print("end: ", x_list[i][-1, 0:3])
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        fig.legend()
        fig.colorbar(im)

# def plot_xz_vel(self, params, bias_param, x=0.5, y=0., z=0.25, ydot=0., num_pts=10):
#     xdot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
#     zdot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
#     hvals = vmap(lambda s1: vmap(lambda s2: h_model(np.array([[x, y, z, s1, ydot, s2]]), params, bias_param))(zdot))(xdot).squeeze()
#
#     divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))
#
#     fig,ax = plt.subplots()
#     im = ax.imshow(hvals.T, extent=[vdot_lim[0],vdot_lim[1], vdot_lim[0], vdot_lim[1]], origin='lower',norm=divnorm,cmap=cm.coolwarm_r)
#     fig.colorbar(im)
#     ax.set_xlabel('$\dot{x}$', fontsize=18)
#     ax.set_ylabel('$\dot{z}$', fontsize=18)
#     plt.show()
#
# def plot_yz_vel(params, bias_param, x=0.5, y=0., z=0.25, xdot=0., num_pts=10):
#     ydot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
#     zdot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
#     hvals = vmap(lambda s1: vmap(lambda s2: h_model(np.array([[x, y, z, xdot, s1, s2]]), params, bias_param))(zdot))(ydot).squeeze()
#     divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))
#
#     fig,ax = plt.subplots()
#     im = ax.imshow(hvals.T, extent=[vdot_lim[0],vdot_lim[1], vdot_lim[0], vdot_lim[1]], origin='lower',norm=divnorm,cmap=cm.coolwarm_r)
#     fig.colorbar(im)
#     ax.set_xlabel('$\dot{y}$', fontsize=18)
#     ax.set_ylabel('$\dot{z}$', fontsize=18)
#     plt.show()
#
#
# def plot_xy_pos(params, bias_param,  z=0.25, xdot=0.1, ydot=0.1, zdot=0.1, num_pts=10):
#     x = onp.linspace(x_lim[0], x_lim[1], num=num_pts)
#     y = onp.linspace(y_lim[0], y_lim[1], num=num_pts)
#     hvals = vmap(lambda s1: vmap(lambda s2: h_model(np.array([[s1, s2, z, xdot, ydot, zdot]]), params, bias_param))(y))(
#     x).squeeze()
#
#     divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))
#
#     fig,ax = plt.subplots()
#     im = ax.imshow(hvals.T, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], origin='lower', norm=divnorm,
#                cmap=cm.coolwarm_r)
#     fig.colorbar(im)
#     ax.set_xlabel('$x$', fontsize=18)
#     ax.set_ylabel('$y$', fontsize=18)
#     plt.show()
#
# def plot_yz_pos(params, bias_param,  x=0.5, xdot=0.1, ydot=0.1, zdot=0.1, num_pts=11):
#     y = np.linspace(y_lim[0], y_lim[1], num=num_pts)
#     z = np.linspace(z_lim[0], z_lim[1], num=num_pts)
#
#     hvals = vmap(lambda s1: vmap(lambda s2: h_model(np.array([[x, s1, s2, xdot, ydot, zdot]]), params, bias_param))(z))(
#     y).squeeze()
#
#     divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))
#
#     fig,ax = plt.subplots()
#     im = ax.imshow(hvals.T, extent=[y_lim[0], y_lim[1], z_lim[0], z_lim[1]], origin='lower', norm=divnorm,
#                cmap=cm.coolwarm_r)
#     fig.colorbar(im)
#     ax.set_xlabel('$x$', fontsize=18)
#     ax.set_ylabel('$y$', fontsize=18)
#     plt.show()
#
# def plot_xz_pos(theta, bias,  y=0., xdot=0.1, ydot=0.1, zdot=0.1, num_pts=11):
#     x = np.linspace(x_lim[0], x_lim[1], num=num_pts)
#     z = np.linspace(z_lim[0], z_lim[1], num=num_pts)
#
#     hvals = vmap(lambda s1: vmap(lambda s2: h_model(np.array([[s1, y, s2, xdot, ydot, zdot]]), theta, bias))(z))(
#     x).squeeze()
#
#     divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))
#
#     fig,ax = plt.subplots()
#     im = ax.imshow(hvals.T, extent=[x_lim[0], x_lim[1], z_lim[0], z_lim[1]], origin='lower', norm=divnorm,
#                cmap=cm.coolwarm_r)
#     fig.colorbar(im)
#     ax.set_xlabel('$x$', fontsize=18)
#     ax.set_ylabel('$y$', fontsize=18)
#     plt.show()
#

if __name__ == '__main__':

    # Check passed argument - User number
    if len(sys.argv) >= 2:
        user_number = sys.argv[1]
    else:
        user_number = '0'

    print("Running CBF Plotting for User_"+user_number+"\n")

    # Get data
    # data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_"+user_number+"/"
    data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/cbf_tests/"

    data = pickle.load(open(data_dir + "vel_data_dict.p", "rb"))
    print(data.keys())
    # Set up data for MPC planner
    params = data["theta"]
    bias_param = data["bias"]
    slack_param = data["unsafe_slack"]
    centers = data["rbf_centers"]
    stds = data["rbf_stds"]
    bias_param = 0.1 #0.1

    plotter = PlotCBF(params, bias_param, centers, stds, data_dir)

    # plotter.plot_xy_pos(num_pts=30)
    # plotter.plot_xz_pos(num_pts=30)
    # plotter.plot_yz_pos(num_pts=30)
    # plotter.plot_xy_vel(num_pts=30)
    # plotter.plot_xz_vel(num_pts=30)
    # plotter.plot_yz_vel(num_pts=30)
    plotter.plot_centers_hvals()
    v = 0.4
    plotter.plot_xyz_animate(xdot=v, ydot=v, zdot=v, num_slices=11, num_pts=11)
    # plotter.plot_xyz_neg_animate(num_slices=11, num_pts=11)

    # # plotter.plot_xz_pos_multiple()
    # plotter.plot_xz_pos_animate()
    #
    # plotter.plot_xz_vel_animate()
    # # plotter.plot_3D_quiver_uniform()
    # plotter.plot_3D_quiver_from_traj(num_traj=3)
    # # plotter.plot_xyz_neg()#num_pts=30)


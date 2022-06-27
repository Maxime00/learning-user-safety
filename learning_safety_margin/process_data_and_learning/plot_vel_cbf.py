
import numpy as onp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors


from jax import random, vmap, jit, grad, ops, lax, tree_util, device_put, device_get, jacobian, jacfwd, jacrev, jvp
import jax.numpy as np
from learning_cbf_vel_lim import h_model, x_lim, y_lim, z_lim, vdot_lim


def plot_xy_vel(params, bias_param, x=0.5, y=0., z=0.25, zdot=0. num=10):
    xdot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
    ydot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
    hvals = vmap(lambda s1: vmap(lambda s2: h_model(np.array([[x, y, z, s1, s2, zdot]]), params, bias_param))(ydot))(xdot).squeeze()

    divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))

    fig,ax = plt.subplots()
    im = ax.imshow(hvals.T, extent=[vdot_lim[0],vdot_lim[1], vdot_lim[0], vdot_lim[1]], origin='lower',norm=divnorm,cmap=cm.coolwarm_r)
    fig.colorbar(im)
    ax.set_xlabel('$\dot{x}$', fontsize=18)
    ax.set_ylabel('$\dot{y}$', fontsize=18)
    plt.show()

def plot_xz_vel(params, bias_param, x=0.5, y=0., z=0.25, ydot=0. num=10):
    xdot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
    zdot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
    hvals = vmap(lambda s1: vmap(lambda s2: h_model(np.array([[x, y, z, s1, ydot, s2]]), params, bias_param))(zdot))(xdot).squeeze()

    divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))

    fig,ax = plt.subplots()
    im = ax.imshow(hvals.T, extent=[vdot_lim[0],vdot_lim[1], vdot_lim[0], vdot_lim[1]], origin='lower',norm=divnorm,cmap=cm.coolwarm_r)
    fig.colorbar(im)
    ax.set_xlabel('$\dot{x}$', fontsize=18)
    ax.set_ylabel('$\dot{z}$', fontsize=18)
    plt.show()

def plot_yz_vel(params, bias_param, x=0.5, y=0., z=0.25, xdot=0. num=10):
    ydot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
    zdot = np.linspace(vdot_lim[0], vdot_lim[1], num=num_pts)
    hvals = vmap(lambda s1: vmap(lambda s2: h_model(np.array([[x, y, z, xdot, s1, s2]]), params, bias_param))(zdot))(ydot).squeeze()
    divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))

    fig,ax = plt.subplots()
    im = ax.imshow(hvals.T, extent=[vdot_lim[0],vdot_lim[1], vdot_lim[0], vdot_lim[1]], origin='lower',norm=divnorm,cmap=cm.coolwarm_r)
    fig.colorbar(im)
    ax.set_xlabel('$\dot{y}$', fontsize=18)
    ax.set_ylabel('$\dot{z}$', fontsize=18)
    plt.show()


def plot_xy_pos(params, bias_param,  z=0.25, xdot=0.1, ydot=0.1 zdot=0.1, num=10):
    x = np.linspace(x_lim[0], x_lim[1], num=num_pts)
    y = np.linspace(y_lim[0], y_lim[1], num=num_pts)
    hvals = vmap(lambda s1: vmap(lambda s2: h_model(np.array([[s1, s2, z, xdot, ydot, zdot]]), params, bias_param))(y))(
    x).squeeze()

    divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))

    fig,ax = plt.subplots()
    im = ax.imshow(hvals.T, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], origin='lower', norm=divnorm,
               cmap=cm.coolwarm_r)
    fig.colorbar(im)
    ax.set_xlabel('$x$', fontsize=18)
    ax.set_ylabel('$y$', fontsize=18)
    plt.show()

def plot_yz_pos(params, bias_param,  x=0.5, xdot=0.1, ydot=0.1 zdot=0.1, num=10):
    y = np.linspace(y_lim[0], y_lim[1], num=num_pts)
    z = np.linspace(z_lim[0], z_lim[1], num=num_pts)

    hvals = vmap(lambda s1: vmap(lambda s2: h_model(np.array([[x, s1, s2, xdot, ydot, zdot]]), params, bias_param))(z))(
    y).squeeze()

    divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))

    fig,ax = plt.subplots()
    im = ax.imshow(hvals.T, extent=[y_lim[0], y_lim[1], z_lim[0], z_lim[1]], origin='lower', norm=divnorm,
               cmap=cm.coolwarm_r)
    fig.colorbar(im)
    ax.set_xlabel('$x$', fontsize=18)
    ax.set_ylabel('$y$', fontsize=18)
    plt.show()

def plot_xz_pos(params, bias_param,  y=0., xdot=0.1, ydot=0.1 zdot=0.1, num=10):
    x = np.linspace(x_lim[0], x_lim[1], num=num_pts)
    z = np.linspace(z_lim[0], z_lim[1], num=num_pts)

    hvals = vmap(lambda s1: vmap(lambda s2: h_model(np.array([[s1, y, s2, xdot, ydot, zdot]]), params, bias_param))(z))(
    x).squeeze()

    divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))

    fig,ax = plt.subplots()
    im = ax.imshow(hvals.T, extent=[x_lim[0], x_lim[1], z_lim[0], z_lim[1]], origin='lower', norm=divnorm,
               cmap=cm.coolwarm_r)
    fig.colorbar(im)
    ax.set_xlabel('$x$', fontsize=18)
    ax.set_ylabel('$y$', fontsize=18)
    plt.show()



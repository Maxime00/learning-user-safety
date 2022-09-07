import numpy as np
import matplotlib.pyplot as plt
from vel_control_utils import *
import matplotlib.colors as colors
import pickle
# from jax import vmap

bottom_box_corners = np.array([[0.44, 0.145], [0.44, -0.12], [0.69, -0.12],[0.69, 0.145]])
bottom_box_heights = np.array([0.14, 0.23])
bbox_lims = np.array([[0.44,0.69], [-0.12, 0.145], [0.14, 0.23]])
top_box_corners = np.array([[0.44, 0.06], [0.44, -0.12], [0.69, -0.12],[0.69, 0.06]])
top_box_heights = np.array([0.23, 0.345])
tbox_lims = np.array([[0.44,0.69], [-0.12, 0.06], [0.23, 0.345]])

bbox_coords = []
for i in range(len(bottom_box_heights)):
    for j in range(len(bottom_box_corners)):
        bbox_coords.append(np.hstack(( bottom_box_corners[j], bottom_box_heights[i],)))
bbox_coords = np.array(bbox_coords)

print(bbox_coords.shape, bbox_coords)

tbox_coords = []
for i in range(len(top_box_heights)):
    for j in range(len(top_box_corners)):
        tbox_coords.append(np.hstack(( top_box_corners[j], top_box_heights[i],)))
tbox_coords = np.array(tbox_coords)

print(tbox_coords.shape, tbox_coords)
print(bbox_coords[:,0])

def check_pt_interior(pt, lims=bbox_lims):
    interior=False
    if lims[0,0] <= pt[0] <= lims[0,1] and lims[1,0] <= pt[1] <= lims[1,1] and lims[2,0] <= pt[2] <= lims[2,1]:
        interior=True
    return interior

def draw_box(coords, fig, ax, color='g' ):
    draw_bt_pts = np.vstack((coords[:4], coords[0]))
    draw_top_pts = np.vstack((coords[4:], coords[4]))
    ax.plot3D(draw_bt_pts[:, 0], draw_bt_pts[:, 1], draw_bt_pts[:, 2], color)
    ax.plot3D(draw_top_pts[:, 0], draw_top_pts[:, 1], draw_top_pts[:, 2], color)
    for i in range(4):
        line = np.vstack((coords[:4][i],coords[4:][i]))
        ax.plot3D(line[:,0], line[:,1], line[:,2], color)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# draw_box(bbox_coords, fig, ax)
# draw_box(tbox_coords, fig, ax, color='b')
#
# ax.set_xlim(x_lim)
# ax.set_ylim(y_lim)
# ax.set_zlim(z_lim)
# plt.show()

test_pts = np.random.uniform(ws_lim[:3, 0], ws_lim[:3, 1], (1000, 3))
print(test_pts.shape)
b_int = [check_pt_interior(pt, bbox_lims) for pt in test_pts]
t_int = [check_pt_interior(pt, tbox_lims) for pt in test_pts]

pt_colors = []
for i in range(len(test_pts)):
    if b_int[i] is True:
        c = 'g'
    elif t_int[i] is True:
        c='b'
    else:
        c='y'
    pt_colors.append(c)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# draw_box(bbox_coords, fig, ax)
# draw_box(tbox_coords, fig, ax, color='b')
# ax.scatter3D(test_pts[:,0], test_pts[:,1], test_pts[:,2], c=pt_colors)
# ax.set_xlim(x_lim)
# ax.set_ylim(y_lim)
# ax.set_zlim(z_lim)
# plt.show()

# Define ellipse for each obstacle

class Ellipse:

    def __init__(self, axes, center=[0,0,0]):

        self.center = center

        self.xc = center[0]
        self.yc = center[1]
        self.zc = center[2]

        self.a = axes[0]
        self.b = axes[1]
        self.c = axes[2]

    def plot_ellipse(self, fig, ax, color='g'):
        u = np.linspace(0,2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = self.a * np.outer(np.cos(u), np.sin(v)) + self.xc
        y = self.b * np.outer(np.sin(u), np.sin(v)) + self.yc
        z = self.c * np.outer(np.ones_like(u), np.cos(v)) + self.zc

        ax.plot_surface(x,y,z, rstride=4, cstride=4, color=color, alpha=0.5)


    def check_dist(self, pt):
        x = pt[0]
        y = pt[1]
        z = pt[2]
        dist = (x - self.xc) ** 2 / self.a ** 2 \
               + (y - self.yc) ** 2 / self.b ** 2 \
               + (z - self.zc) ** 2 / self.c ** 2

        return dist

    def check_collision(self, pt):
        dist = self.check_dist(pt)
        if dist <= 1: collision = True
        else: collision = False

        return dist, collision

    def cbf_val(self, pt):
        h = self.check_dist(pt) - 2
        return h



b_center = (bbox_lims[:,1]-bbox_lims[:,0])/2 + bbox_lims[:,0]
b_axes = (.15, .15, .12)
b_ell = Ellipse(b_axes, center=b_center)

t_center = (tbox_lims[:,1]-tbox_lims[:,0])/2 + tbox_lims[:,0]
t_axes = (.15, .1, .1)
t_ell = Ellipse(t_axes, center=t_center)

fig = plt.figure()
ax = plt.axes(projection='3d')
draw_box(bbox_coords, fig, ax)
draw_box(tbox_coords, fig, ax, color='b')
b_ell.plot_ellipse(fig=fig, ax=ax)
t_ell.plot_ellipse(fig=fig, ax=ax, color='b')

ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.set_zlim(z_lim)
plt.show()

# Test CBF values

fig = plt.figure()
ax = plt.axes(projection='3d')
draw_box(bbox_coords, fig, ax)
draw_box(tbox_coords, fig, ax, color='b')

h_vals = [b_ell.cbf_val(pt) + t_ell.cbf_val(pt) for pt in test_pts]

divnorm = colors.TwoSlopeNorm(vmin=-5., vcenter=0, vmax=40)
im = ax.scatter3D(test_pts[:,0], test_pts[:,1], test_pts[:,2], c=h_vals, norm=divnorm, cmap='RdBu')

ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.set_zlim(z_lim)
fig.colorbar(im)
plt.show()

### Make Manual CBF using RBF centers
def rbf(x,c,s):
    return np.exp(-1/(2 *s**2) * np.sum((x-c)**2))# np.linalg.norm(x - c) ** 2)#

def phi(x):
    #a = rbf(x,centers, stds)
    a = np.array([rbf(x, c, s) for c, s, in zip(centers, stds)])
    return a
# def _rbf(x, c, s):
#     # return np.exp(-1 / (2 * s[0] ** 2) * np.linalg.norm(x - c) ** 2)
#     return np.exp(-1 / (2 * s[0] ** 2) * np.sum((x - c) ** 2))
# rbf = vmap(_rbf, in_axes=(None, 0, 0))
#
# def phi(x):
#     # a = np.array([rbf(x, c, s) for c, s, in zip(centers, stds)])
#     a = rbf(x, centers, stds)
#     return a  # np.array(y)
# phi_vec = vmap(phi)

def h_function(x, theta, bias=0.1):
    return phi(x).dot(theta) + bias

# # Initialize RBF Parameters
# x_dim = 3
# n_features = 1000#n_dim_features**x_dim
# u_dim = 2
# rbf_std = 0.1#.1#onp.max(mu_dist) * 0.5 #0.1#1.0
# centers, stds = rbf_means_stds(X=None, X_lim = ws_lim[:3],
#                                n=x_dim, k=n_dim_features, set_means='random',fixed_stds=True, std=rbf_std, nCenters=n_features)
# print("rbf shapes", centers.shape, stds.shape)
# bias = 0.#0.1
#
# # Calculate RBF theta parameters
# theta = np.array([(b_ell.cbf_val(pt) + t_ell.cbf_val(pt))*5 for pt in centers])
#
# theta = np.clip(theta, -10, 1)
# print(theta.shape)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# draw_box(bbox_coords, fig, ax)
# draw_box(tbox_coords, fig, ax, color='b')
#
# ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], c=theta, norm=divnorm, cmap='RdBu')
# plt.show()
#
# hvals = [h_function(pt, theta, bias) for pt in centers]#test_pts]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# draw_box(bbox_coords, fig, ax)
# draw_box(tbox_coords, fig, ax, color='b')
# divnorm = colors.TwoSlopeNorm(vmin=-5., vcenter=0, vmax=1.)
#
# im= ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], c=hvals, norm=divnorm, cmap='RdBu')
# fig.colorbar(im)
# plt.show()


# hvals = [h_function(pt, theta, bias) for pt in test_pts]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# draw_box(bbox_coords, fig, ax)
# draw_box(tbox_coords, fig, ax, color='b')
# divnorm = colors.TwoSlopeNorm(vmin=-5., vcenter=0, vmax=1.)
#
# im= ax.scatter3D(test_pts[:, 0], test_pts[:, 1], test_pts[:, 2], c=hvals, norm=divnorm, cmap='RdBu')
# fig.colorbar(im)
# plt.show()
#
# is_bias = True
# data = {
#     "theta": theta,
#     "bias": bias,
#     "rbf_centers": centers,
#     "rbf_stds": stds,
#     "is_bias": is_bias,
# }
#
# data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/cbf_tests/"
#
# pickle.dump(data, open(data_dir + "manual_rbfcbf_vel_data_dict.p", "wb"))
# plt.show()

# Define Velocity CBF functions

def sigmoid(x, a):
    z = 1./((1.+np.exp(2*(x-a))))
    return z

xdot_lim = 1.
ydot_lim = 1.
zdot_lim = 1.


def xd_cbf(pt):
    h = 3*sigmoid(pt, xdot_lim) - 3*sigmoid(pt, -xdot_lim) - 1.
    return h
def yd_cbf(pt):
    h = 3*sigmoid(pt, ydot_lim) - 3*sigmoid(pt, -ydot_lim) - 1.
    return h
def zd_cbf(pt):
    h = 3*sigmoid(pt, zdot_lim) - 3*sigmoid(pt, -zdot_lim) - 1.
    return h

pts = np.linspace(-5,5, 20)
xd_pts = np.array([xd_cbf(pt) for pt in pts])
plt.plot(pts, xd_pts)

yd_pts = np.array([yd_cbf(pt) for pt in pts])
plt.plot(pts, yd_pts)

zd_pts = np.array([zd_cbf(pt) for pt in pts])
plt.plot(pts, zd_pts)

plt.show()

X, Y, Z = np.meshgrid(pts, pts, pts)

xx = X.ravel()
yy = Y.ravel()
zz = Z.ravel()
vel_pts = np.vstack((xx, yy, zz)).T
print(X.shape, Y.shape, Z.shape, xx.shape, yy.shape, zz.shape, vel_pts.shape)


def vel_cbf(pt):
    val = xd_cbf(pt[0]) + yd_cbf(pt[1]) + zd_cbf(pt[2])#(b_ell.cbf_val(pt) + t_ell.cbf_val(pt))*5 +
    return val**2

hvals = np.array([vel_cbf(pt) for pt in vel_pts])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# divnorm = colors.TwoSlopeNorm(vmin=-5., vcenter=0, vmax=1.)

im= ax.scatter3D(vel_pts[:, 0], vel_pts[:, 1], vel_pts[:, 2], c=hvals, norm=divnorm, cmap='RdBu')
fig.colorbar(im)
plt.show()

# Initialize RBF Parameters
x_dim = 6
n_features = 1000#n_dim_features**x_dim
u_dim = 2
rbf_std = 0.1#.1#onp.max(mu_dist) * 0.5 #0.1#1.0
centers, stds = rbf_means_stds(X=None, X_lim = ws_lim,
                               n=x_dim, k=n_dim_features, set_means='random',fixed_stds=True, std=rbf_std, nCenters=n_features)
print("rbf shapes", centers.shape, stds.shape)
bias = 0.#0.1

# Calculate RBF theta parameters
theta_p = np.array([(b_ell.cbf_val(pt[:3]) + t_ell.cbf_val(pt[:3]))*5 for pt in centers])
theta = np.clip(theta_p, -10, 1.)

theta_v = np.array([vel_cbf(pt[3:])for pt in centers])
theta = np.clip(theta_p, -10, 1.)

theta = np.clip(theta_p + theta_v, -10, 1.)
print(theta.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
draw_box(bbox_coords, fig, ax)
draw_box(tbox_coords, fig, ax, color='b')
plt.title('theta vals')
im = ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], c=theta)#, norm=divnorm, cmap='RdBu')
fig.colorbar(im)
plt.show()

hvals = [h_function(pt, theta, bias) for pt in centers]#test_pts]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
draw_box(bbox_coords, fig, ax)
draw_box(tbox_coords, fig, ax, color='b')
divnorm = colors.TwoSlopeNorm(vmin=-5., vcenter=0, vmax=1.)

im= ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], c=hvals)#, norm=divnorm, cmap='RdBu')
fig.colorbar(im)
plt.title('centers hvals')
plt.show()

test_pts = np.random.uniform(ws_lim[:, 0], ws_lim[:, 1], (1000, 6))
print(test_pts.shape)

hvals = [h_function(pt, theta, bias) for pt in test_pts]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
draw_box(bbox_coords, fig, ax)
draw_box(tbox_coords, fig, ax, color='b')
divnorm = colors.TwoSlopeNorm(vmin=-5., vcenter=0, vmax=1.)

im= ax.scatter3D(test_pts[:, 0], test_pts[:, 1], test_pts[:, 2], c=hvals)#, norm=divnorm, cmap='RdBu')
fig.colorbar(im)
plt.title('test pts: hvals')
plt.show()

is_bias = True
data = {
    "theta": theta,
    "bias": bias,
    "rbf_centers": centers,
    "rbf_stds": stds,
    "is_bias": is_bias,
}

data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/cbf_tests/"

pickle.dump(data, open(data_dir + "manual_rbfcbf_vel_data_dict.p", "wb"))

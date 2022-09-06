import numpy as np
import matplotlib.pyplot as plt
from vel_control_utils import *
from itertools import combinations

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
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(bbox_coords[:,0], bbox_coords[:,1],bbox_coords[:,2],  'g*')
# ax.plot3D(tbox_coords[:,0], tbox_coords[:,1], tbox_coords[:,2], 'b.')
# ax.set_xlim(x_lim)
# ax.set_ylim(y_lim)
# ax.set_zlim(z_lim)
#
# plt.show()

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

    # lines = combinations(coords, 2)
    # for ind in list(lines):
    #     line = np.vstack(ind)
    #     # print(line)
    #     ax.plot3D(line[:,0], line[:,1], line[:,2], color)

fig = plt.figure()
ax = plt.axes(projection='3d')
draw_box(bbox_coords, fig, ax)
draw_box(tbox_coords, fig, ax, color='b')

ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.set_zlim(z_lim)
plt.show()

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
fig = plt.figure()
ax = plt.axes(projection='3d')
draw_box(bbox_coords, fig, ax)
draw_box(tbox_coords, fig, ax, color='b')
ax.scatter3D(test_pts[:,0], test_pts[:,1], test_pts[:,2], c=pt_colors)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.set_zlim(z_lim)
plt.show()
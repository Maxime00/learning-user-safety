import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.colors as colors
from matplotlib import cm
from franka_pos_cbfmpc_control import *
from utils import kin_car_u


MaxSpeed = 1 
MaxAngular = 4.124/6

def alpha(x):
    return psi * x

def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * np.linalg.norm(x-c)**2)

def phi(X):
    a = np.array([rbf(X, c, s) for c, s, in zip(centers, stds)])
    return a

def h_model(x, theta, bias):
    return phi(x).dot(theta) + bias

# Set Data Path
path = os.getcwd()
print("Current Directory", path)

parent_dir = os.path.abspath(os.path.join(path, os.pardir))
print(parent_dir)

data_path = path + '/figures/oa/w_artificial_pts/artsafepts_2std/'
print(data_path)
data = pickle.load( open(data_path+ "data_dict.p", "rb" ) )
params = data["theta"]
bias_param = data["bias"] 
slack_param = data["unsafe_slack"]
bias_param=0.1

# Initialize RBF Parameters
print(ws_lim, x_lim, y_lim, z_lim, x_dim, n_dim_features, rbf_std)
centers, stds = rbf_means_stds(X=None, X_lim = np.array([x_lim,y_lim,z_lim]), 
                               n=x_dim, k=n_dim_features, fixed_stds=True, std=rbf_std)

x = np.array([0.,0.,0.])#onp.random.uniform(low=[0.6, 0.2], high=[1.,0.8], size=(3,))
print(bias_param)
print(h_model(x, params, bias_param))


class SingleIntegrator():
    def __init__(self, dt=0.1):
        self.dt = dt

    def forward_sim(self, x, u, steps=1):
        return x + u * self.dt*steps

# Set up RHC-MPC

horizon_steps = 5
dyn = SingleIntegrator()
controller = CBFMPC_Controller(centers, stds, params, bias_param, n_steps=horizon_steps)
xgoal = [0.,0.,0.]

# Plot h model
num_pts = 50
x1 = np.linspace(x_lim[0], x_lim[1], num=num_pts)
x2 = np.linspace(y_lim[0], y_lim[1], num=num_pts)
x3 = np.linspace(z_lim[0], z_lim[1], num=num_pts)

xx,yy = np.meshgrid(x1, x2)
hvals = np.zeros(xx.shape)
xvec = xx.ravel()
yvec = yy.ravel()
hvec = hvals.ravel()
for i in range(len(xvec)):
    hvec[i] = controller.h_fun([xvec[i], yvec[i], 0])

hvals = hvec.reshape((num_pts, num_pts))
divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))

fig,ax = plt.subplots()
im = ax.imshow(hvals, extent=[x_lim[0],x_lim[1], y_lim[0], y_lim[1]], origin='lower',norm=divnorm, cmap=cm.coolwarm_r)
fig.colorbar(im)
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$y$', fontsize=18)
ax.set_title('Learned CBF')

xlist = np.linspace(0.4,0.6,3)
ylist = np.linspace(-0.4,-0.2,3)
zlist = np.linspace(0,0.3,5)
for i in range(len(xlist)):
    for j in range(len(ylist)):
        x0 = [xlist[i],ylist[j], 0.1]
        max_iter = 100
        xtraj = [x0]
        utraj = []
        dist = np.inf
        iter = 0
        xcurr = x0
        while dist >= 0.01 and iter <= max_iter:
            u_des = single_integrator_u(xcurr, xgoal, limits=[MaxSpeed, MaxAngular])
            xdes = dyn.forward_sim(xcurr, u_des, steps=10).tolist()
            X, U = controller.control(xcurr, xdes)
            u_rhc = np.array(U[0,:]).squeeze()
            xnew = dyn.forward_sim(xcurr, u_rhc)
            xtraj.append(xnew)
            utraj.append(u_rhc)
            dist = np.linalg.norm(xnew-xgoal)
            xcurr = xnew
            iter = iter +1

        xtraj = np.array(xtraj)
        utraj = np.array(utraj)
        plt.plot(xtraj[:, 0], xtraj[:, 1])
plt.xlim(ws_lim[0])
plt.ylim(ws_lim[1])
plt.show()

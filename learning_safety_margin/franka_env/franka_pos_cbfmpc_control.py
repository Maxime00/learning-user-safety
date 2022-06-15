import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import matplotlib.colors as colors
from matplotlib import cm
#from casadi import *
import casadi
import os
from control_utils import *
DEBUG_FLAG = False
MaxSpeed = 1.5 # max Qolo speed: 1.51 m/s               --> Equivalent to 5.44 km/h
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
        

class CBFMPC_Controller(SingleIntegrator):

    def __init__(self, centers, stds, theta, bias, dt=0.1, n_steps=10):

        super().__init__(dt)
        # Set MPC Parameters
        self.N = n_steps
        self.dt = dt
        # CBF Parameters
        self.gamma = 0.05 # CBF Parameter
        self.centers = centers
        self.stds = stds
        self.theta = theta
        self.bias = bias
        # Optimization weights' variables
        self.Q_x = 100
        self.Q_y = 100
        self.Q_theta = 10
        self.R1 = 1
        self.R2 = 1
        self.R3 = 1

        self.Q = casadi.diagcat(self.Q_x, self.Q_y, self.Q_theta) # State Weights 
        self.R = casadi.diagcat(self.R1, self.R2, self.R3) # Control Weights
        
        # Set up variables
        self.x = casadi.SX.sym('x', 3)
        self.u = casadi.SX.sym('u', 3)

        self.n_states = self.x.numel()
        self.n_controls = self.u.numel()

        self.X = casadi.SX.sym('X',  self.n_states, self.N+1) # Horizontal vectors (nxN vector, n: dim of state, N steps in future)
        self.U = casadi.SX.sym('U', self.n_controls, self.N) # Horizontal vectors (mxN vector, m: dim of control, N steps in future)

        # coloumn vector for storing initial state and target state
        self.P = casadi.SX.sym('P', self.n_states + self.n_states)

        # Generate Casadi Functions for h(x) functions
        c = self.centers[0]
        s = self.stds[0]
        self.phi = casadi.exp(-1 / (2 * s**2) * casadi.norm_2(self.x-c)**2)
        for i in range(1,len(self.centers)):
            c = self.centers[i]
            s = self.stds[i]
            rbf = casadi.exp(-1 / (2 * s**2) * casadi.norm_2(self.x-c)**2)
            self.phi = casadi.horzcat(self.phi, rbf)
        self.h = casadi.mtimes(self.phi, self.theta)+ self.bias
        self.h_fun = casadi.Function('h_fun', [self.x],  [self.h])

        # self.dhdx = casadi.gradient(self.h, self.x)
        # self.gradh = casadi.Function('gradh', [self.x], [self.dhdx])
        # print('dhdx([0,0,0]) = ', self.gradh([0.,0.,0.]))


        # Set up Cost Function and Constraint Expressions

        self.lbx = ws_lim[:,0]#[-4,-2,-casadi.pi]
        self.ubx = ws_lim[:,1]#[2, 2, casadi.pi]
        print(self.lbx, self.ubx)

        self.lbX = [self.lbx]*(self.N+1)
        self.ubX = [self.ubx]*(self.N+1)

        self.lbu = [-1., -1., -1.]
        self.ubu = [1., 1., 1.]
        self.lbU = np.array([self.lbu]*(self.N))
        self.ubU = np.array([self.ubu]*(self.N))

        self.lb_vars = np.vstack((self.lbX, self.lbU))
        self.ub_vars = np.vstack((self.ubX, self.ubU))
 
        self.cost_fn = 0
        self.constraints = self.X[:,0] - self.P[:self.n_states] # Initialize constraint list with x0 constraint
        self.lb_con = np.array([0]*self.n_states)
        self.ub_con = np.array([0]*self.n_states)
        for i in range(self.N):
            xi = self.X[:, i]
            ui = self.U[:,i]
            xnext = self.X[:, i+1]
            # Objective Function
            self.cost_fn = self.cost_fn + (xi-self.P[self.n_states:]).T @ self.Q @ (xi-self.P[self.n_states:]) + ui.T @ self.R @ ui

            # Constraints
            xnext_sim = self.forward_sim(xi, ui)#xi + ui * self.dt # Forward Sim
            dyn_con = xnext-xnext_sim #Dynamics Constraint
            self.constraints = casadi.vertcat(self.constraints, dyn_con) # Add Dynamics Constraint
            self.lb_con=np.hstack((self.lb_con, np.array([0]*self.n_states))) # Add Equality lower bounds to lists
            self.ub_con = np.hstack((self.ub_con, [0]*self.n_states)) # Add Equality lower bounds to lists
            h_con = self.h_fun(xnext) - self.h_fun(xi) + self.gamma * self.h_fun(xi) #CBF constraint (grad(h(x)) >= -gamma * h(x) ==> h(x_{i+1}) - h(x_{i}) + gamma * h(x_{i}) >= 0
            self.constraints = casadi.vertcat(self.constraints, h_con) # Add CBF Constraint
            self.lb_con=np.hstack((self.lb_con, 0)) # Add Equality lower bounds to lists
            self.ub_con = np.hstack((self.ub_con, casadi.inf)) # Add Inequality lower bounds to lists

        # Set up NLP Problem
        self.nlp = {}

        self.nlp['x'] = casadi.vertcat(self.X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            self.U.reshape((-1, 1)))
        self.nlp['f'] = self.cost_fn
        self.nlp['g'] = self.constraints
        self.nlp['p'] = self.P

        self.opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
                },
            'print_time': 0
            }
        self.solver = casadi.nlpsol('solver', 'ipopt', self.nlp, self.opts)



    def control(self, x0, xgoal):
        # Set up arg dictionary for optimization inputs
        args = {}
        args['lbg'] = self.lb_con
        args['ubg'] = self.ub_con

        args['lbx'] = self.lb_vars.reshape((-1,1))
        args['ubx'] = self.ub_vars.reshape((-1,1))

        u0 = casadi.DM.zeros((self.n_controls, self.N))  # initial control
        X0 = casadi.repmat(x0, 1, self.N+1)         # initial state full
        params = casadi.vertcat(
            x0,    # current state
            xgoal   # target state
        )
        args['p'] = params

        args['x0'] = casadi.vertcat(
            casadi.reshape(X0, self.n_states*(self.N+1), 1),
            casadi.reshape(u0, self.n_controls*self.N, 1)
            )
        sol = self.solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']    
                )

        U = casadi.reshape(sol['x'][self.n_states * (self.N+1):], self.n_controls, self.N).T
        X = casadi.reshape(sol['x'][:self.n_states * (self.N+1)], self.n_states, self.N+1).T
        return X, U

if __name__ == "__main__": 
    controller = CBFMPC_Controller(centers, stds, params, bias_param, n_steps=50)

    num_pts = 30
    x1 = np.linspace(x_lim[0], x_lim[1], num=num_pts)
    x2 = np.linspace(y_lim[0], y_lim[1], num=num_pts)
    x3 = np.linspace(z_lim[0], z_lim[1], num=num_pts)

    xx,yy,zz = np.meshgrid(x1, x2, x3)
    hvals = np.zeros(xx.shape)
    xvec = xx.ravel()
    yvec = yy.ravel()
    zvec = zz.ravel()
    hvec = hvals.ravel()
    for i in range(len(xvec)):
        hvec[i] = controller.h_fun([xvec[i], yvec[i], zvec[i]])

    hvals = hvec.reshape((num_pts, num_pts, num_pts))

    divnorm = colors.TwoSlopeNorm(vmin=np.min(hvals), vcenter=0., vmax=np.max(hvals))

    # fig,ax = plt.subplots()
    # im = ax.imshow(hvals, extent=[x_lim[0],x_lim[1], y_lim[0], y_lim[1]], origin='lower',norm=divnorm, cmap=cm.coolwarm_r)
    # fig.colorbar(im)
    neg_pts = []
    neg_h = []

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    for i in range(len(xvec)): 
        if hvec[i] < 0: 
            im = ax.scatter(xvec[i], yvec[i], zvec[i], c=hvec[i], norm=divnorm, cmap=cm.coolwarm_r)#, marker=m)
            neg_pts.append([xvec[i], yvec[i], zvec[i]])
            neg_h.append(hvec[i])


    ax.set_xlabel('$x$', fontsize=18)
    ax.set_ylabel('$y$', fontsize=18)
    ax.set_title('Learned CBF')
    ax.view_init(10, 180)


    xgoal = [0.5,0.4,0.1]
    xlist = np.linspace(0.4,0.6,3)
    ylist = np.linspace(-0.4,-0.2,3)
    zlist = np.linspace(0,0.3,5)
    traj = []
    for i in range(len(xlist)): 
        for j in range(len(ylist)):
            x0 = [xlist[i],ylist[j], 0.1]
            if controller.h_fun(x0) > 0: 
                X, U = controller.control(x0, xgoal)
                Xmat = np.array(X)
                ax.plot(Xmat[:, 0], Xmat[:, 1], Xmat[:,2])
                traj.append(Xmat)
            else:
                print('unsafe start point! Invalid.')
    plt.xlim(ws_lim[0])
    plt.ylim(ws_lim[1])
    plt.show()

    
    yy,zz = np.meshgrid(x2, x3)
    hvals = np.zeros(yy.shape)
    yvec = yy.ravel()
    zvec = zz.ravel()
    hvec = hvals.ravel()
    for i in range(len(yvec)):
        hvec[i] = controller.h_fun([0.5, yvec[i], zvec[i]])

    hvals = hvec.reshape((num_pts, num_pts))
    
    fig,ax = plt.subplots()
    im = ax.imshow(hvals, extent=[y_lim[0],y_lim[1], z_lim[0], z_lim[1]], origin='lower',norm=divnorm, cmap=cm.coolwarm_r)
    fig.colorbar(im)

    for i in range(len(traj)):
        ax.plot(traj[i][:, 1], traj[i][:,2])
    ax.set_xlabel('$y$')
    ax.set_ylabel('$z$')
    plt.show()

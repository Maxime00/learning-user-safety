import numpy as np
import casadi
from learning_safety_margin.vel_control_utils import *


class DoubleIntegrator():
    def __init__(self, dt=0.1):
        self.dt = dt

    def forward_sim(self, x, u, steps=1):
        return x + casadi.vertcat(x[3], x[4], x[5], u[0], u[1], u[2]) * self.dt*steps
        

class CBFMPC_Controller(DoubleIntegrator):

    def __init__(self, centers, stds, theta, bias, dt=0.1, n_steps=10):

        super().__init__(dt)
        # Set MPC Parameters
        self.N = n_steps
        self.dt = dt
        self.tlist = np.linspace(0,self.N*self.dt, self.N+1)
        
        # Set up variables
        self.x = casadi.SX.sym('x', 6)
        self.u = casadi.SX.sym('u', 3)

        self.n_states = self.x.numel()
        self.n_controls = self.u.numel()

        # Initialize Trajectory State Variables
        self.X = casadi.SX.sym('X',  self.n_states, self.N+1) # Horizontal vectors (nxN vector, n: dim of state, N steps in future)
        self.U = casadi.SX.sym('U', self.n_controls, self.N) # Horizontal vectors (mxN vector, m: dim of control, N steps in future)

        # coloumn vector for storing initial state and target state
        self.P = casadi.SX.sym('P', self.n_states + self.n_states)

        # Optimization weights' variables
        self.Q_x = 100
        self.Q_y = 100
        self.Q_theta = 100
        self.Q_v = 10
        self.R1 = 1
        self.R2 = 1
        self.R3 = 1

        self.Q = casadi.diagcat(self.Q_x, self.Q_y, self.Q_theta, self.Q_v, self.Q_v, self.Q_v) # State Weights 
        self.R = casadi.diagcat(self.R1, self.R2, self.R3) # Control Weights

        # CBF Parameters
        self.gamma = 0.05 # CBF Parameter
        self.centers = centers
        self.stds = stds
        self.theta = theta
        self.bias = bias

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

        # Set up Cost Function and Constraint Expressions

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

        # Set Variable Limits
 
        self.lbx = casadi.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))
        self.ubx = casadi.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))       

        self.lbx[0: self.n_states*(self.N+1): self.n_states] = ws_lim[0,0]     # X lower bound
        self.lbx[1: self.n_states*(self.N+1): self.n_states] = ws_lim[1,0]     # Y lower bound
        self.lbx[2: self.n_states*(self.N+1): self.n_states] = ws_lim[2,0]     # Z lower bound
        self.lbx[3: self.n_states*(self.N+1): self.n_states] = ws_lim[3,0]     # VX lower bound
        self.lbx[4: self.n_states*(self.N+1): self.n_states] = ws_lim[4,0]     # VY lower bound
        self.lbx[5: self.n_states*(self.N+1): self.n_states] = ws_lim[5,0]     # VZ lower bound

        self.ubx[0: self.n_states*(self.N+1): self.n_states] = ws_lim[0,1]     # X lower bound
        self.ubx[1: self.n_states*(self.N+1): self.n_states] = ws_lim[1,1]     # Y lower bound
        self.ubx[2: self.n_states*(self.N+1): self.n_states] = ws_lim[2,1]     # Z lower bound
        self.ubx[3: self.n_states*(self.N+1): self.n_states] = ws_lim[3,1]     # VX lower bound
        self.ubx[4: self.n_states*(self.N+1): self.n_states] = ws_lim[4,1]     # VY lower bound
        self.ubx[5: self.n_states*(self.N+1): self.n_states] = ws_lim[5,1]     # VZ lower bound

        self.lbx[self.n_states*(self.N+1):] = -1.                 # v lower bound for all V
        self.ubx[self.n_states*(self.N+1):] = 1.                  # v upper bound for all V


    def control(self, x0, xgoal, t0=0):
        # Set up arg dictionary for optimization inputs
        args = {}
        args['lbg'] = self.lb_con
        args['ubg'] = self.ub_con

        args['lbx'] = self.lbx
        args['ubx'] = self.ubx

        u0 = casadi.DM.zeros((self.n_controls, self.N))  # initial control
        X0 = casadi.repmat(x0, 1, self.N+1)         # initial state full
        params = casadi.vertcat(
            x0,    # current state
            xgoal   # target state
        )
        args['p'] = params

        args['x0'] = casadi.vertcat(
            casadi.reshape(X0, self.n_states*(self.N+1), 1),
            casadi.reshape(u0, self.n_controls*(self.N), 1)
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

        T = self.tlist + t0
        return X, U, T

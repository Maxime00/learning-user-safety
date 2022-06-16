import numpy as np
import casadi
from learning_safety_margin.control_utils import *

DEBUG_FLAG = False
MaxSpeed = 1.5  # max Qolo speed: 1.51 m/s               --> Equivalent to 5.44 km/h
MaxAngular = 4.124 / 6


class SingleIntegrator():
    def __init__(self, dt=0.1):
        self.dt = dt

    def forward_sim(self, x, u, steps=1):
        return x + u * self.dt * steps

class CBFMPC_Controller(SingleIntegrator):

    def __init__(self, centers, stds, theta, bias, dt=0.1, n_steps=10):

        super().__init__(dt)
        # Set MPC Parameters
        self.N = n_steps
        self.dt = dt
        # CBF Parameters
        self.gamma = 0.05  # CBF Parameter
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

        self.Q = casadi.diagcat(self.Q_x, self.Q_y, self.Q_theta)  # State Weights
        self.R = casadi.diagcat(self.R1, self.R2, self.R3)  # Control Weights

        # Set up variables
        self.x = casadi.SX.sym('x', 3)
        self.u = casadi.SX.sym('u', 3)

        self.n_states = self.x.numel()
        self.n_controls = self.u.numel()

        self.X = casadi.SX.sym('X', self.n_states,
                               self.N + 1)  # Horizontal vectors (nxN vector, n: dim of state, N steps in future)
        self.U = casadi.SX.sym('U', self.n_controls,
                               self.N)  # Horizontal vectors (mxN vector, m: dim of control, N steps in future)

        # coloumn vector for storing initial state and target state
        self.P = casadi.SX.sym('P', self.n_states + self.n_states)

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

        # self.dhdx = casadi.gradient(self.h, self.x)
        # self.gradh = casadi.Function('gradh', [self.x], [self.dhdx])
        # print('dhdx([0,0,0]) = ', self.gradh([0.,0.,0.]))

        # Set up Cost Function and Constraint Expressions

        self.lbx = ws_lim[:, 0]  # [-4,-2,-casadi.pi]
        self.ubx = ws_lim[:, 1]  # [2, 2, casadi.pi]
        print(self.lbx, self.ubx)

        self.lbX = [self.lbx] * (self.N + 1)
        self.ubX = [self.ubx] * (self.N + 1)

        self.lbu = [-1., -1., -1.]
        self.ubu = [1., 1., 1.]
        self.lbU = np.array([self.lbu] * (self.N))
        self.ubU = np.array([self.ubu] * (self.N))

        self.lb_vars = np.vstack((self.lbX, self.lbU))
        self.ub_vars = np.vstack((self.ubX, self.ubU))

        self.cost_fn = 0
        self.constraints = self.X[:, 0] - self.P[:self.n_states]  # Initialize constraint list with x0 constraint
        self.lb_con = np.array([0] * self.n_states)
        self.ub_con = np.array([0] * self.n_states)
        for i in range(self.N):
            xi = self.X[:, i]
            ui = self.U[:, i]
            xnext = self.X[:, i + 1]
            # Objective Function
            self.cost_fn = self.cost_fn + (xi - self.P[self.n_states:]).T @ self.Q @ (
                        xi - self.P[self.n_states:]) + ui.T @ self.R @ ui

            # Constraints
            xnext_sim = self.forward_sim(xi, ui)  # xi + ui * self.dt # Forward Sim
            dyn_con = xnext - xnext_sim  # Dynamics Constraint
            self.constraints = casadi.vertcat(self.constraints, dyn_con)  # Add Dynamics Constraint
            self.lb_con = np.hstack((self.lb_con, np.array([0] * self.n_states)))  # Add Equality lower bounds to lists
            self.ub_con = np.hstack((self.ub_con, [0] * self.n_states))  # Add Equality lower bounds to lists
            h_con = self.h_fun(xnext) - self.h_fun(xi) + self.gamma * self.h_fun(
                xi)  # CBF constraint (grad(h(x)) >= -gamma * h(x) ==> h(x_{i+1}) - h(x_{i}) + gamma * h(x_{i}) >= 0
            self.constraints = casadi.vertcat(self.constraints, h_con)  # Add CBF Constraint
            self.lb_con = np.hstack((self.lb_con, 0))  # Add Equality lower bounds to lists
            self.ub_con = np.hstack((self.ub_con, casadi.inf))  # Add Inequality lower bounds to lists

        # Set up NLP Problem
        self.nlp = {}

        self.nlp['x'] = casadi.vertcat(self.X.reshape((-1, 1)),  # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
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

        args['lbx'] = self.lb_vars.reshape((-1, 1))
        args['ubx'] = self.ub_vars.reshape((-1, 1))

        u0 = casadi.DM.zeros((self.n_controls, self.N))  # initial control
        X0 = casadi.repmat(x0, 1, self.N + 1)  # initial state full
        params = casadi.vertcat(
            x0,  # current state
            xgoal  # target state
        )
        args['p'] = params

        args['x0'] = casadi.vertcat(
            casadi.reshape(X0, self.n_states * (self.N + 1), 1),
            casadi.reshape(u0, self.n_controls * self.N, 1)
        )
        sol = self.solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        U = casadi.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N).T
        X = casadi.reshape(sol['x'][:self.n_states * (self.N + 1)], self.n_states, self.N + 1).T
        return X, U



    #controller = CBFMPC_Controller(centers, stds, params, bias_param, n_steps=50)



    # xgoal = [0.5, 0.4, 0.1]
    # xlist = np.linspace(0.4, 0.6, 3)
    # ylist = np.linspace(-0.4, -0.2, 3)
    # zlist = np.linspace(0, 0.3, 5)
    # traj = []
    # for i in range(len(xlist)):
    #     for j in range(len(ylist)):
    #         x0 = [xlist[i], ylist[j], 0.1]
    #         if controller.h_fun(x0) > 0:
    #             X, U = controller.control(x0, xgoal)
    #             Xmat = np.array(X)
    #             ax.plot(Xmat[:, 0], Xmat[:, 1], Xmat[:, 2])
    #             traj.append(Xmat)
    #         else:
    #             print('unsafe start point! Invalid.')
    # plt.xlim(ws_lim[0])
    # plt.ylim(ws_lim[1])
    # plt.show()

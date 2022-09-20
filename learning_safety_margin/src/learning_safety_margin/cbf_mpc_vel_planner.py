import numpy as np
import casadi
from learning_safety_margin.vel_control_utils import *
import state_representation as sr
from dynamical_systems import create_cartesian_ds, DYNAMICAL_SYSTEM_TYPE
import matplotlib.pyplot as plt
from scipy import interpolate

class DoubleIntegrator():
    def __init__(self, dt=0.1):
        self.dt = dt

    def forward_sim(self, x, u, steps=1):
        return x + casadi.vertcat(x[3], x[4], x[5], u[0], u[1], u[2]) * self.dt*steps
        

class CBFMPC_Controller(DoubleIntegrator):

    def __init__(self, centers, stds, theta, bias, dt=0.1, n_steps=10, v_gain = 10, r_gains =1, zero_acc_start = False):

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

        # column vector for storing initial state and target state
        self.P = casadi.SX.sym('P', self.n_states + self.n_states)

        # Optimization weights' variables
        self.Q_x = 100
        self.Q_y = 100
        self.Q_theta = 100
        self.Q_v = v_gain
        self.R1 = r_gains
        self.R2 = r_gains
        self.R3 = r_gains

        self.Q = casadi.diagcat(self.Q_x, self.Q_y, self.Q_theta, self.Q_v, self.Q_v, self.Q_v) # State Weights 
        self.R = casadi.diagcat(self.R1, self.R2, self.R3) # Control Weights

        # CBF Parameters
        self.gamma = 0.1#0.05 # CBF Parameter
        self.centers = centers
        self.stds = stds
        self.theta = theta
        self.bias = bias

        # Generate Casadi Functions for h(x) functions
        c = self.centers[0]
        s = self.stds[0]
        # self.phi = casadi.exp(-1 / (2 * s**2) * casadi.norm_2(self.x-c)**2)
        self.phi = casadi.exp(-1 / (2 * s**2) * casadi.sumsqr(self.x - c))
        for i in range(1,len(self.centers)):
            c = self.centers[i]
            s = self.stds[i]
            # rbf = casadi.exp(-1 / (2 * s**2) * casadi.norm_2(self.x-c)**2)
            rbf = casadi.exp(-1 / (2 * s ** 2) * casadi.sumsqr(self.x - c))
            self.phi = casadi.horzcat(self.phi, rbf)
        self.h = casadi.mtimes(self.phi, self.theta)+ self.bias
        self.h_fun = casadi.Function('h_fun', [self.x],  [self.h])

        # Set up Cost Function and Constraint Expressions
        self.cost_fn = 0
        self.constraints = self.X[:,0] - self.P[:self.n_states] # Initialize constraint list with x0 constraint
        # adding zero acceleration at start constraint
        if zero_acc_start:
            cmd_constraint = self.U[:,0] - np.zeros(self.n_controls)
            self.constraints = casadi.vertcat(self.constraints, cmd_constraint)
            self.lb_con = np.array([0]*(self.n_states+self.n_controls))
            self.ub_con = np.array([0]*(self.n_states+self.n_controls))
        else:
            self.lb_con = np.array([0] * self.n_states)
            self.ub_con = np.array([0] * self.n_states)

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
                'acceptable_obj_change_tol': 1e-6,
            },
            'print_time': 0,
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

        self.lbx[self.n_states*(self.N+1):] = -1                 # v lower bound for all V
        self.ubx[self.n_states*(self.N+1):] = 1                # v upper bound for all V


    def control(self, x0, xgoal, t0=0, ig_time=None, ig_pos=None, ig_vel=None, ig_acc=None, count=0, plot_vel_acc=False):
        # Set up arg dictionary for optimization inputs
        args = {}
        args['lbg'] = self.lb_con
        args['ubg'] = self.ub_con

        args['lbx'] = self.lbx
        args['ubx'] = self.ubx
        print("goal: ", xgoal)

        ## NO MOVE initial guess
        # u0 = casadi.DM.zeros((self.n_controls, self.N))  # initial control
        # X0 = casadi.repmat(x0, 1, self.N+1)         # initial state full

        if ig_pos is None:
            ## Straight line initial guess
            X0 = np.linspace(x0, xgoal, self.N+1)
            u0 = np.zeros((self.N, self.n_controls))

            # Integrate for velocity
            # for i in range(self.N):
            #     X0[i, 3:6] = (X0[i+1, 0:3] - X0[i, 0:3])/ self.dt

            # DS for velocity
            ds = create_cartesian_ds(DYNAMICAL_SYSTEM_TYPE.POINT_ATTRACTOR)
            ds.set_parameter_value("gain", [50., 50., 50., 10., 10., 10.], sr.ParameterType.DOUBLE_ARRAY)
            target = sr.CartesianPose('panda_ee', xgoal[0:3], np.array([0., 1., 0., 0.]), 'panda_base')
            ds.set_parameter_value("attractor", target, sr.ParameterType.STATE, sr.StateType.CARTESIAN_POSE)
            curr_state = sr.CartesianState('panda_ee', 'panda_base')
            curr_state.set_orientation(np.array([0., 1., 0., 0.]))

            for i in range(self.N):
                curr_state.set_position(X0[i, 0:3])
                ds_twist = sr.CartesianTwist(ds.evaluate(curr_state))
                # TODO:  need clamping ?? Initial velocities are very high
                # ds_twist.clamp(.25, .5)
                X0[i, 3:6] = ds_twist.data()[0:3]

            for i in range(self.N):
                u0[i, :] = (X0[i+1, 3:6] - X0[i, 3:6]) / self.dt

        elif ig_pos is not None:
            # Demo traj initial guess
            X0 = np.zeros((self.N+1, self.n_states))

            ## Interpolate method
            if ig_time[-1] < self.tlist[-1]:  # recorded traj shorter than planner time
                ig_time = np.linspace(0, self.N*self.dt, len(ig_time))
                print("TRAJ FOR INITIAL GUESS SHORTER THAN PLANNER TIME ! \n")
            print("Time of initial guess traj : ", ig_time[-1], " sec \n")

            f_pos = interpolate.interp1d(ig_time, ig_pos[:, 0:3], axis=0)
            f_vel = interpolate.interp1d(ig_time, ig_vel[:, 0:3],  axis=0)
            f_acc = interpolate.interp1d(ig_time, ig_acc[:, 0:3], axis=0)

            # resample time to match
            sub_sampled_time = np.linspace(0, ig_time[-1], self.N+1)

            sampled_pos = f_pos(sub_sampled_time)
            sampled_vel = f_vel(sub_sampled_time)
            sampled_acc = f_acc(sub_sampled_time[:-1])

            ## Subsample method
            # step_size = round(len(ig_pos[:,0])/(self.N+1))
            # sampled_pos = ig_pos[::step_size, 0:3]
            # sampled_vel = ig_vel[::step_size, 0:3]
            #
            # print(sampled_pos.shape)
            # # if subsample off
            # while len(sampled_pos[:, 0]) < self.N+1:
            #     sampled_pos = np.append(sampled_pos, [ig_pos[-1, 0:3]], axis=0)
            #     sampled_vel = np.append(sampled_vel, [ig_vel[-1, 0:3]], axis=0)
            # while len(sampled_pos[:, 0]) > self.N + 1:
            #     sampled_pos = np.delete(sampled_pos, -1, axis=0)
            #     sampled_vel = np.delete(sampled_vel, -1, axis=0)
            #
            # step_size = round(len(ig_acc[:, 0]) / self.N)
            # sampled_acc = ig_acc[::step_size, 0:3]
            #
            # while len(sampled_acc[:, 0]) < self.N:
            #     np.append(sampled_acc, [ig_acc[-1, 0:3]], axis=0)
            # while len(sampled_acc[:, 0]) > self.N:
            #     sampled_acc = np.delete(sampled_acc, -1, axis=0)

            X0[:, 0:3] = sampled_pos
            X0[:, 3:6] = sampled_vel
            u0 = sampled_acc

        ## PLOTS DEBUG
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # plt.plot(X0[:, 0], X0[:, 1], X0[:, 2], label='position')
        # ax.set_xlim(x_lim)
        # ax.set_ylim(y_lim)
        # ax.set_zlim(z_lim)
        # ax.set_xlabel("$x$")
        # ax.set_ylabel("$y$")
        # ax.set_zlabel("$z$")
        # fig.legend()
        # fig.suptitle(f"Initial guess position #{count+1}")

        if plot_vel_acc:
            fig = plt.figure()
            ax = plt.axes()
            plt.plot(self.tlist, X0[:, 3], label='vx')
            plt.plot(self.tlist, X0[:, 4], label='vy')
            plt.plot(self.tlist, X0[:, 5], label='vz')
            fig.legend()
            fig.suptitle(f"Initial guess velocity #{count+1}")

            fig = plt.figure()
            ax = plt.axes()
            plt.plot(self.tlist[:-1], u0[:, 0], label='ax')
            plt.plot(self.tlist[:-1], u0[:, 1], label='ay')
            plt.plot(self.tlist[:-1], u0[:, 2], label='az')
            fig.legend()
            fig.suptitle(f"Initial guess acceleration #{count+1}")
            # plt.show()

        X0 = casadi.DM(X0.T)
        u0 = casadi.DM(u0.T)

        print(X0.shape, u0.shape)
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
        # print("Initial Cost: ", self.cost_fn(["x0"]))

        print("Optimal Cost: ", float(sol["f"], ))
        U = casadi.reshape(sol['x'][self.n_states * (self.N+1):], self.n_controls, self.N).T
        X = casadi.reshape(sol['x'][:self.n_states * (self.N+1)], self.n_states, self.N+1).T

        T = self.tlist + t0
        return X, U, T

    def check_safety(self, X):
        print(X.shape)
        safe=True
        hvals = np.zeros(X.shape[0])
        for i in range(len(X)):
            hvals[i] = self.h_fun(X[i])
        if np.any(hvals <= 0.):
            safe=False
            print("Traj not Safe")

        # for i in range(len(X)-1):
        #     xi = X[i]
        #     xnext = X[i+1]
        #
        #     if not (self.h_fun(xnext) - self.h_fun(xi) + self.gamma * self.h_fun(xi) >= 0.):
        #         safe=False
        #         print("Traj not Safe")
        return safe

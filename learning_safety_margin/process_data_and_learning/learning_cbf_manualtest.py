import numpy as onp
import matplotlib.pyplot as plt
import jax.numpy as np#jnp
import sys
import pickle
import os
import time
from jax import random, vmap, jit, grad, ops, lax, tree_util, device_put, device_get, jacobian, jacfwd, jacrev, jvp
import cvxpy as cp
import mosek

import seaborn as sns
sns.set_style('darkgrid')
palette = sns.color_palette()

import jax
print(jax.default_backend())
print(jax.devices())
def vel_learning(user_number):

    # benchmark
    start_time = time.time()

    # Set up data path
    # data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_"+user_number+"/"
    data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/cbf_tests/"
    fig_path = '../franka_env/figures/vel_lim/'
    save = False

    # Velocity Limits
    x_lim = [0.25, 0.75]
    y_lim = [-0.45, 0.45]
    z_lim = [0., 0.7]
    vdot_lim = [-1., 1.]
    xdot_lim = [-0.6, 1.1]#[-0.5, 0.4]
    ydot_lim = [-2., 1.5]#[-0.8, 0.8]
    zdot_lim = [-1.1, 1.1]#[-0.8, 0.8]

    ws_lim = np.vstack((x_lim, y_lim, z_lim, xdot_lim, ydot_lim, zdot_lim))  # vdot_lim, vdot_lim, vdot_lim))

    # Define Dynamical System

    eps = 1.0
    # X = [x, y, z, xdot, ydot, zdot]
    # Xdot = [xdot, ydot, zdot, xddot, yddot,  zdot] = [u1*cos(theta), u1*sin(theta), u2]

    def dynamics(x, u):
        # continuous time: xdot = f(x(t)) + g(x(t))u((t))
        # (xdot=[u1*cos(theta), u1*sin(theta), u2], x = x0 + xdot*dt)
        return np.array([x[3], x[4], x[5], u[0], u[1], u[2]])

    all_pts = onp.loadtxt(data_dir + "center_pts.txt")
    all_vals = onp.loadtxt(data_dir + "theta_hvals.txt")

    print(all_pts.shape, all_vals.shape)
    safe_condition = np.where(all_vals >= 1.)
    safe_pts = np.squeeze(all_pts[safe_condition, :])
    safe_vals = all_vals[safe_condition]
    print(safe_pts.shape, safe_vals.shape)
    semisafe_condition = np.where(np.logical_and(0. < all_vals, all_vals <= 1.))
    semisafe_pts = np.squeeze(all_pts[semisafe_condition, :])
    semisafe_vals = all_vals[semisafe_condition]
    print(semisafe_pts.shape, semisafe_vals.shape)
    unsafe_condition = np.where(all_vals <= 0.)
    unsafe_pts = np.squeeze(all_pts[unsafe_condition, :])
    unsafe_vals = all_vals[unsafe_condition]
    print(unsafe_pts.shape, unsafe_vals.shape)

    ### Set unsafe tte list to all ones (no decay to end)
    unsafe_ttelist = onp.ones(len(unsafe_pts))

    semisafe_u = np.ones((semisafe_pts.shape[0], 3))

    print(safe_pts.shape, unsafe_pts.shape, semisafe_pts.shape, semisafe_u.shape)
    print(np.min(safe_vals), np.max(safe_vals), np.min(unsafe_vals), np.max(unsafe_vals), np.min(semisafe_vals), np.max(semisafe_vals))
    # Define reward lists
    safe_rewards = safe_vals#onp.ones(len(safe_pts))*2.
    unsafe_rewards = unsafe_vals#onp.ones(len(unsafe_pts)) * -1.
    semisafe_rewards = semisafe_vals#onp.ones(len(semisafe_pts))* 0.5

    ### Set up Minimization
    # Sample Data
    n_safe_sample = 1#200#300#50
    x_safe = safe_pts[::n_safe_sample]
    safe_rewards = safe_rewards[::n_safe_sample]

    n_unsafe_sample = 1#50#50#100#20
    x_unsafe = unsafe_pts[::n_unsafe_sample]
    unsafe_rewards = unsafe_rewards[::n_unsafe_sample]
    unsafe_tte = unsafe_ttelist[::n_unsafe_sample]

    n_semisafe_sample = 1#200#300#10
    x_semisafe = semisafe_pts[::n_semisafe_sample]
    u_semisafe = semisafe_u[::n_semisafe_sample]
    semisafe_rewards = semisafe_rewards[::n_semisafe_sample]

    print(x_safe.shape, safe_rewards.shape, x_unsafe.shape, unsafe_rewards.shape, unsafe_tte.shape, x_semisafe.shape, u_semisafe.shape, semisafe_rewards.shape)

    # Initialize Data
    n_safe = len(x_safe)
    n_unsafe = len(x_unsafe)
    n_semisafe = len(x_semisafe)
    n_artificial = 0

    x_all = onp.vstack((x_safe, x_unsafe, x_semisafe))
    # print("x-all", x_all.shape)
    # x_out = onp.where((x_all[:,0]<= ws_lim[0,0]) | (ws_lim[0,1] <= x_all[:,0])) # and x_all[:,0] <= x_all[0,1])
    # y_out = onp.where((x_all[:, 1] <= ws_lim[1, 0]) | (ws_lim[1, 1] <= x_all[:, 1]))
    # z_out = onp.where((x_all[:, 2] <= ws_lim[2, 0]) | (ws_lim[2, 1] <= x_all[:, 2]))
    # xd_out = onp.where((x_all[:, 3] <= ws_lim[3, 0]) | (ws_lim[3, 1] <= x_all[:, 3]))
    # yd_out = onp.where((x_all[:, 4] <= ws_lim[4, 0]) | (ws_lim[4, 1] <= x_all[:, 4]))
    # zd_out = onp.where((x_all[:, 5] <= ws_lim[5, 0]) | (ws_lim[5, 1] <= x_all[:, 5]))
    #
    # pts_out = []
    # remove_indices = []
    # for i in range(len(ws_lim)):
    #     out = onp.where((x_all[:, i] <= ws_lim[i, 0]) | (ws_lim[i, 1] <= x_all[:, i]))
    #     pts_out.append(out)
    # remove_indices = None
    # for i in range(len(pts_out)):
    #     print(pts_out[i][0])
    #     print(len(pts_out[i][0]), pts_out[i][0].shape)
    #     if len(pts_out[i][0]) > 0:
    #         if remove_indices is None: remove_indices = pts_out[i][0]
    #         else:
    #             remove_indices = onp.hstack((remove_indices, pts_out[i][0]))
    # print(remove_indices, remove_indices is not None)
    # if remove_indices is not None: x_all = onp.delete(x_all, remove_indices, axis=0)
    # print(x_out, y_out, z_out, xd_out, yd_out, zd_out, pts_out, remove_indices)
    # print("new x_all", x_all.shape, len(x_all))


    # Define h model (i.e., RBF)

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
            means = onp.random.uniform(low=X_lim[:, 0], high=X_lim[:, 1], size=(nCenters, X_lim.shape[0]))
        if set_means == 'inputs':
            assert X is not None, 'X invalid data input. Cannot be None-type'
            assert k == X.shape[0], 'Set_means inputs, num kernels must equal num of data points'
            means = X.copy()

        # Generate stds
        if fixed_stds == True:
            stds = onp.ones(means.shape[0]) * std
        else:
            #         stds = np.random.uniform(low = 0.0001, high = std, size=(k**n,1))
            stds = random.uniform(rng.next(), minval=0.0001, maxval=std, shape=(k ** n, 1))
        stds = np.squeeze(stds)
        return means, stds


    # def alpha(x):
    #     return psi * x


    def _rbf(x, c, s):
        # return np.exp(-1 / (2 * s[0] ** 2) * np.linalg.norm(x - c) ** 2)
        return np.exp(-1 / (2 * s[0] ** 2) * np.sum((x - c) ** 2))

    rbf = vmap(_rbf,in_axes=(None, 0, 0))
    def phi(x):
        # a = np.array([rbf(x, c, s) for c, s, in zip(centers, stds)])
        a = rbf(x,centers, stds)
        return a  # np.array(y)
    phi_vec = vmap(phi)
    # # @jax.jit
    # def h_model(x, theta, bias):
    #
    #     #     print(x.shape, phi(x).shape, theta.shape, phi(x).dot(theta).shape, bias.shape)
    #     #     print(x.type, theta.type, bias.type)
    #     return phi(x).dot(theta) + bias
    # Initialize RBF Parameters
    n_dim_features = 4
    x_dim = 6
    n_features = 1000#n_dim_features**x_dim
    u_dim = 2
    psi = 1.0
    dt = 0.1
    dist_eps = 0.01#0.05
    mu_dist = (ws_lim[:, 1]-ws_lim[:,0])/n_dim_features
    print("Check: ", mu_dist, onp.max(mu_dist)*0.5)
    rbf_std = 0.2#.1#onp.max(mu_dist) * 0.5 #0.1#1.0
    print(rbf_std)
    # centers, stds = rbf_means_stds(X=None, X_lim = ws_lim,
    #                                n=x_dim, k=n_dim_features, set_means='random',fixed_stds=True, std=rbf_std, nCenters=n_features)
    # print("rbf shapes", centers.shape, stds.shape)
    ## Trying RBF centers at data pts


    centers = onp.array(x_all)
    # # centers = onp.array([x_all[0]])
    # for i in range(len(x_all)):
    #     dist = np.linalg.norm(centers - x_all[i], axis=1)
    #     # print(i, centers.shape, dist.shape)
    #     if np.all(dist > 0.01):#rbf_std):
    #         centers = onp.vstack((centers, x_all[i]))
    # stds = np.ones(centers.shape[0])*rbf_std
    # print("Centers: {}".format(centers.shape))

    # pts = []
    # for i in range(ws_lim.shape[0]):
    #     pts.append(np.linspace(start=ws_lim[i, 0], stop=ws_lim[i, 1], num=5, endpoint=True))
    # pts = np.array(pts)
    # pts = tuple(pts)
    # D = np.meshgrid(*pts)
    # means = onp.array([D[i].flatten() for i in range(len(D))]).T
    # for i in range(len(means)):
    #     dist = np.linalg.norm(centers - means[i], axis=1)
    #     # print(i, centers.shape, dist.shape)
    #     if np.all(dist > 0.5):##2*rbf_std):
    #         centers = onp.vstack((centers, means[i]))

    stds = onp.ones((centers.shape[0], 1)) * rbf_std
    n_features = centers.shape[0]
    print("Centers: {}, Stds: {}, # Features: {}".format(centers.shape, stds.shape, n_features))
    # print("RBF CHECK", phi(np.ones(6)).shape, phi_vec(np.ones((1,6))).shape)


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # colors = onp.zeros(centers[:, 3:].shape)
    # colors[:,0] = (centers[:,3]-ws_lim[3,0])/(ws_lim[3,1]-ws_lim[3,0])
    # colors[:,1] = (centers[:,4]-ws_lim[4,0])/(ws_lim[4,1]-ws_lim[4,0])
    # colors[:,2] = (centers[:,5]-ws_lim[5,0])/(ws_lim[5,1]-ws_lim[5,0])
    # print(onp.amin(colors), onp.amax(colors))
    # ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=colors)
    # plt.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # colors = onp.zeros(centers[:, :3].shape)
    # colors[:,0] = (centers[:,0]-ws_lim[0,0])/(ws_lim[0,1]-ws_lim[0,0])
    # colors[:,1] = (centers[:,1]-ws_lim[1,0])/(ws_lim[1,1]-ws_lim[1,0])
    # colors[:,2] = (centers[:,2]-ws_lim[2,0])/(ws_lim[2,1]-ws_lim[2,0])
    # print(onp.amin(colors), onp.amax(colors))
    # ax.scatter(centers[:, 3], centers[:, 4], centers[:, 5], c=colors)
    # plt.show()
    # print(x_all.shape, x_safe.shape, x_unsafe.shape, x_semisafe.shape)
    # art_safe_pts = []
    # print(range(len(centers)))
    # for i in range(len(centers)):
    #     dist = np.linalg.norm(x_all-centers[i], axis=1)
    #     if np.all(dist > 2*rbf_std):
    #         art_safe_pts.append(centers[i])
    # art_safe_pts = np.array(art_safe_pts)
    # print(art_safe_pts.shape)



    # Initialize variables
    is_bias = False
    is_slack_both = False
    is_slack_safe = False
    is_semisafe = True
    is_artificial = False
    theta = cp.Variable(n_features)  # , nonneg=True)
    print(theta.shape)
    assert not (is_slack_both and is_slack_safe), "Slack bool cannot be both and safe only"
    if is_bias:
        bias = cp.Variable()
    else:
        bias = 0.1

    if is_slack_both:
        safe_slack = cp.Variable(n_safe)
        unsafe_slack = cp.Variable(n_unsafe)
    elif is_slack_safe:
        safe_slack = cp.Variable(n_safe)
    else:
        unsafe_slack = cp.Variable(n_unsafe)

    # Initialize reward parameters
    r_scaling = 1.
    safe_val = safe_rewards#np.ones(n_safe) * 2.  # np.array(onp.squeeze(safe_rewards)*r_scaling)#np.ones(n_safe)*0.3
    unsafe_val = unsafe_rewards#np.ones(n_unsafe) * -1.0  # *-0.5#*-0.1
    # if is_semisafe:
    semisafe_val = semisafe_rewards#np.ones(n_semisafe) * 0.5  # np.array(onp.squeeze(semisafe_rewards)*r_scaling)#np.ones(n_semisafe)*0.
    gamma_dyn = np.ones(n_semisafe) * 0.1
    # unsafe_tte = unsafe_ttelist ** 3

    if is_artificial:
        art_safe_val = np.ones(len(art_safe_pts)) * 0.1  # *0.5
        n_artificial = len(art_safe_pts)
    print(safe_val.shape, unsafe_val.shape , semisafe_val.shape)#, art_safe_val.shape)
    print(unsafe_val[0])

    # Initialize cost
    h_cost = 0
    param_cost = 0
    slack_cost = 0

    ## Define Constraints
    constraints = []

    start_constraints = time.time()

    # Safe Constraints
    print("SAFE CONSTRAINTS")
    phis_safe = phi_vec(x_safe)#[phi(x) for x in x_safe]
    print("PHI CHECK", x_safe.shape, phis_safe.shape)
    if is_slack_both or is_slack_safe:
        print("Adding safe slack constraints")
        for i in range(n_safe):
            h_cost += cp.sum_squares(theta @ phis_safe[i] + bias)  # cost of norm(alpha_i * phi(x,xi) + b)
            constraints.append((theta @ phis_safe[i] + bias) >= safe_val[i] + safe_slack[i])
            constraints.append(safe_slack[i] <= 0.)
    else:
        print("Adding no slack constraints")
        for i in range(n_safe):
            h_cost += cp.sum_squares(theta @ phis_safe[i] + bias)  # cost of norm(alpha_i * phi(x,xi) + b)
            constraints.append((theta @ phis_safe[i] + bias) >= safe_val[i])

    if is_artificial:
        print("ADDING ARTIFICIAL SAFE PTS")
        phis_artificial = phi_vec(art_safe_pts)#[phi(x) for x in art_safe_pts]
        for i in range(len(art_safe_pts)):
            h_cost += cp.sum_squares(
                theta @ phis_artificial[i] + bias)  # cost of norm(alpha_i * phi(x,xi) + b)
            constraints.append((theta @ phis_artificial[i] + bias) >= art_safe_val[i])
    # Unsafe Constraints
    print("UNSAFE CONSTRAINTS")
    phis_unsafe = phi_vec(x_unsafe)#[phi(x) for x in x_unsafe]
    if is_slack_both or not is_slack_safe:
        print("Adding unsafe slack constraints")
        for i in range(n_unsafe):
            h_cost += cp.sum_squares(theta @ phis_unsafe[i] + bias)  # cost of norm(alpha_i * phi(x,xi) + b)
            constraints.append(
                (theta @ phis_unsafe[i] + bias) <= unsafe_val[i] + unsafe_tte[i] * unsafe_slack[i])
            # constraints.append(unsafe_slack[i] >= 0.)

            dist = np.linalg.norm(x_unsafe[i]-x_safe, axis=1)
            if np.any(dist) <= dist_eps:
                constraints.append(unsafe_slack[i] >= 0.)
            else:
                constraints.append(unsafe_slack[i] == 0)
                print('no slack allowed on neg constraint')
    else:
        print("adding no slack constraints")
        for i in range(n_unsafe):
            h_cost += cp.sum_squares(theta @ phis_unsafe[i] + bias)  # cost of norm(alpha_i * phi(x,xi) + b)
            constraints.append((theta @ phis_unsafe[i] + bias) <= unsafe_val[i])

    if is_semisafe:
        # Boundary Constraints: TODO: need to include semisafe control u values
        print("BOUNDARY CONSTRAINTS")

        phis_semisafe = phi_vec(x_semisafe)#[phi(x) for x in x_semisafe]
        print("SEMISAFE H CONSTRAINTS")
        for i in range(n_semisafe):
            h_cost += cp.sum_squares(theta @ phis_semisafe[i] + bias)  # cost of norm(alpha_i * phi(x,xi) + b)
            constraints.append((theta @ phis_semisafe[i] + bias) >= semisafe_val[i])

        print("SEMISAFE DERIVATIVE CONSTRAINTS")

        #     def q(x, u):
        #         dh = grad(h_model, argnums=0)(x, theta, bias)
        #         return np.dot(dh, dynamics(x, u)) + h_model(x, theta, bias)

        #     qs = q(x_semisafe, u_semisafe)
        #     for i in range(n_semisafe):
        #         constraints.append((qs[i] >= gamma_dyn[i]))
        # print(phi(x_semisafe[0]), x_semisafe[0], jacfwd(phi, argnums=0)(x_semisafe[0]).shape, jacfwd(phi, argnums=0)(x_semisafe[0]) )
        print(dynamics(x_semisafe[0], u_semisafe[0]).shape)
        Dphixdots = device_get(
            vmap(lambda x, u: np.dot(jacfwd(phi, argnums=0)(x), dynamics(x, u)),
                 in_axes=(0, 0))(x_semisafe, u_semisafe))
        # print(x_semisafe.shape, u_semisafe.shape, Dphixdots.shape)
        gamma_xu_fillers = gamma_dyn * np.ones((x_semisafe.shape[0],))
        for i, (this_phi, this_Dphixdot, this_gamma) in enumerate(zip(phis_semisafe, Dphixdots, gamma_xu_fillers)):
            if np.any(np.isnan(this_Dphixdot)):
                print("NANANANANANA", i, this_Dphixdot, x_semisafe[i], u_semisafe[i])
                input()
            #     constraints.append((theta.T * (this_Dphixdot.T + this_phi) + bias) >= this_gamma)
            constraints.append((theta @ this_Dphixdot+ theta @ this_phi + bias) >= this_gamma)
        # print(this_Dphixdot.shape)
    print('All CONSTRAINTS DEFINED')
    print("TIME TO SET CONSTRAINTS : ", time.time() -start_constraints ," seconds \n")

    # # Add Constraints on parameter values
    # for i in range(theta.shape[0]):
    #     constraints.append(cp.abs(theta[i]) <= 5.)
    #
    ## Define Objective

    param_cost = cp.sum_squares(theta) #+ bias ** 2 # l2-norm on parameters
    param_norm = n_features
    if is_slack_both: slack_cost = cp.sum_squares(safe_slack) + cp.sum_squares(unsafe_slack)
    elif is_slack_safe: slack_cost = cp.sum_squares(safe_slack) # Norm on slack variables
    else: slack_cost = cp.sum_squares(unsafe_slack) # Norm on slack variables
    slack_weight = 10.
    slack_norm = n_unsafe
    h_weight = 1.
    h_norm = n_safe + n_unsafe + n_semisafe + n_artificial
    param_weight = 1.#500.
    obj = cp.Minimize(param_weight * param_cost/param_norm + slack_weight * slack_cost / slack_norm + h_weight * h_cost / h_norm)
    print('All costs defined')

    params = None
    bias_param = None
    prob = cp.Problem(obj, constraints)
    mosek_params = {'mosek.iparam.intpnt_max_iterations': 1000000
                    }
    result = prob.solve(solver=cp.MOSEK, verbose=True, mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.primal}, enforce_dpp=True)

    params = theta.value
    if is_bias: bias_param = bias.value
    else: bias_param = None

    if is_slack_both:
        safe_slack_param = safe_slack.value
        unsafe_slack_param  = unsafe_slack.value
    elif is_slack_safe:
        safe_slack_param = safe_slack.value
        unsafe_slack_param  = None
    else:
        safe_slack_param = None
        unsafe_slack_param  = unsafe_slack.value

    if is_semisafe:
        data = {
           "theta": params,
            "bias": bias_param,
            "safe_slack": safe_slack_param,
            "unsafe_slack": unsafe_slack_param,
            "rbf_centers": centers,
            "rbf_stds": stds,
            "x_safe": x_safe,
            "x_unsafe": x_unsafe,
            "x_semisafe": x_semisafe,
            "u_semisafe": u_semisafe,
            "is_bias": is_bias,
            "is_semisafe": is_semisafe,
            "is_slack_safe": is_slack_safe,
            "semisafe_demos_u": semisafe_u,
            "safe_reward_values": safe_val,
            "unsafe_reward_values": unsafe_val,
            "semisafe_reward_values": semisafe_val,
            "gamma_dyn": gamma_dyn,
        }
    else:
        data = {
           "theta": params,
            "bias": bias_param,
            "safe_slack": safe_slack_param,
            "unsafe_slack": unsafe_slack_param,
            "rbf_centers": centers,
            "rbf_stds": stds,
            "x_safe": x_safe,
            "x_unsafe": x_unsafe,
            "is_bias": is_bias,
            "is_semisafe": is_semisafe,
            "is_slack_safe": is_slack_safe,
            "safe_reward_values": safe_val,
            "unsafe_reward_values": unsafe_val,
            }

    pickle.dump(data, open(data_dir+"vel_data_dict.p", "wb"))
    print("COMPLETE LEARNING TIME : ", time.time() - start_time, " seconds \n")
    plt.show()

if __name__ == '__main__':

    if len(sys.argv) >= 2:
        user_number = sys.argv[1]
    else:
        user_number = 0

    if isinstance(user_number, str):
        print("Learning velocity cbf for User_"+user_number)
        vel_learning(user_number)
    else:
        print("Learning velocity cbf for User_0 \n")
        print("To process other user, provide user number as sole argument: python3 bag2csv.py 2")
        vel_learning("0")

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
    data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_"+user_number+"/"
    csv_dir = data_dir + 'csv/'
    rosbag_dir = data_dir + "rosbags/"
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
    # Xdot = [xdot, ydot, zdot, xddot, yddot,  zddot] = [u1*cos(theta), u1*sin(theta), u2]

    def dynamics(x, u):
        # continuous time: xdot = f(x(t)) + g(x(t))u((t))
        # (x = x0 + xdot*dt)
        return np.array([x[3], x[4], x[5], u[0], u[1], u[2]])


    nSafe = len(os.listdir(rosbag_dir+"safe"))
    nUnsafe = len(os.listdir(rosbag_dir+"unsafe"))
    nDaring = len(os.listdir(rosbag_dir+"daring"))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    safe_traj = []
    nInvalidSafe = 0
    for i in range(0, nSafe):
        fname = csv_dir + 'safe/' + str(i + 1) + '_eePosition.txt'
        if os.path.exists(fname):
            pos = onp.loadtxt(fname, delimiter=',')[:,0:3]
            safe_traj.append(pos)
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'g')
        else:
            print("Safe Demo {} File Path does not exist: {}".format(i, fname))
            nInvalidSafe += 1

    unsafe_traj = []
    tte_list = []
    nInvalidUnsafe = 0
    for i in range(0, nUnsafe):
        fname = csv_dir + 'unsafe/' + str(i + 1) + '_eePosition.txt'
        if os.path.exists(fname):
            pos = onp.loadtxt(fname, delimiter=',')[:,0:3]
            tte = onp.expand_dims(onp.ones(pos.shape[0]), axis=1)
            for j in range(pos.shape[0]):
                tte[j] = float((pos.shape[0] - (j + 1)) / pos.shape[0])
            unsafe_traj.append(pos)
            tte_list.append(tte)
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'r')
        else:
            print("Unsafe Demo {} File Path does not exist: {}".format(i, fname))
            nInvalidUnsafe += 1


    daring_traj = []
    nInvalidDaring = 0
    for i in range(0, nDaring):
        fname = csv_dir + 'daring/' + str(i + 1) + '_eePosition.txt'
        if os.path.exists(fname):
            pos = onp.loadtxt(fname, delimiter=',')[:, 0:3]
            daring_traj.append(pos)
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b')
        else:
            print("Daring Demo {} File Path does not exist: {}".format(i, fname))
            nInvalidDaring += 1


    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title('Franka CBF Vel_lim: Position Data')
    if save: plt.savefig(fig_path + 'demonstration_data.pdf')
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    safe_vel = []
    for i in range(0, nSafe):
        fname = csv_dir + 'safe/' + str(i+1) + '_eeVelocity.txt'
        if os.path.exists(fname):
            vel = onp.loadtxt(fname, delimiter=',')[:,0:3]
            safe_vel.append(vel)
            ax.plot(vel[:,0], vel[:,1], vel[:,2], 'g')
        else:
            print("Safe Demo {} File Path does not exist: {}".format(i, fname))

    unsafe_vel = []
    for i in range(0, nUnsafe):
        fname = csv_dir + 'unsafe/' + str(i+1) + '_eeVelocity.txt'
        if os.path.exists(fname):
            vel = onp.loadtxt(fname, delimiter=',')[:,0:3]
            unsafe_vel.append(vel)
            ax.plot(vel[:,0], vel[:,1], vel[:,2], 'r')
        else:
            print("Unsafe Demo {} File Path does not exist: {}".format(i, fname))

    daring_vel = []
    for i in range(0, nDaring):
        fname = csv_dir + 'daring/' + str(i+1) + '_eeVelocity.txt'
        if os.path.exists(fname):
            vel = onp.loadtxt(fname, delimiter=',')[:,0:3]
            daring_vel.append(vel)
            ax.plot(vel[:,0], vel[:,1], vel[:,2], 'b')
        else:
            print("Daring Demo {} File Path does not exist: {}".format(i, fname))

    daring_acc = []
    for i in range(0, nDaring):
        fname = csv_dir + 'daring/' + str(i+1) + '_eeAcceleration.txt'
        if os.path.exists(fname):
            acc = onp.loadtxt(fname, delimiter=',')[:,0:3]
            daring_acc.append(acc)
            # ax.plot(acc[:,0], acc[:,1], acc[:,2], 'b')
        else:
            print("Daring Demo {} File Path does not exist: {}".format(i, fname))

    ax.set_xlabel('$\dot{x}$')
    ax.set_ylabel('$\dot{y}$')
    ax.set_zlabel('$\dot{z}$')
    ax.set_title('Franka CBF Velocity Limits: Velocity Data')
    # plt.show()

    nSafe = nSafe-nInvalidSafe
    nUnsafe = nUnsafe-nInvalidUnsafe
    nDaring = nDaring-nInvalidDaring

    xtraj = onp.hstack((safe_traj[0], safe_vel[0]))
    safe_pts = xtraj
    for i in range(1, nSafe):
        xtraj = onp.hstack((safe_traj[i], safe_vel[i]))
        safe_pts = onp.vstack((safe_pts, xtraj))

    xtraj = onp.hstack((unsafe_traj[0], unsafe_vel[0]))
    unsafe_pts = xtraj
    unsafe_ttelist = tte_list[0]
    for i in range(1, nUnsafe):
        xtraj = onp.hstack((unsafe_traj[i], unsafe_vel[i]))
        unsafe_pts = onp.vstack((unsafe_pts, xtraj))
        unsafe_ttelist = onp.vstack((unsafe_ttelist, tte_list[i]))

    ### Set unsafe tte list to all ones (no decay to end)
    unsafe_ttelist = onp.ones(len(unsafe_pts))

    xtraj = onp.hstack((daring_traj[0], daring_vel[0]))
    semisafe_pts = xtraj
    semisafe_u = daring_acc[0]
    for i in range(1, nDaring):
        xtraj = onp.hstack((daring_traj[i], daring_vel[i]))
        semisafe_pts = onp.vstack((semisafe_pts, xtraj))
        semisafe_u = onp.vstack((semisafe_u, daring_acc[i]))

    print(safe_pts.shape, unsafe_pts.shape, semisafe_pts.shape, semisafe_u.shape)

    # Define reward lists
    safe_rewards = onp.ones(len(safe_pts))*2.
    unsafe_rewards = onp.ones(len(unsafe_pts)) * -1.
    semisafe_rewards = onp.ones(len(semisafe_pts))*0.# 0.5

    ### Set up Minimization
    # Sample Data
    n_safe_sample = 200#300#50
    x_safe = safe_pts[::n_safe_sample]
    safe_rewards = safe_rewards[::n_safe_sample]

    n_unsafe_sample = 50#50#100#20
    x_unsafe = unsafe_pts[::n_unsafe_sample]
    unsafe_rewards = unsafe_rewards[::n_unsafe_sample]
    unsafe_tte = unsafe_ttelist[::n_unsafe_sample]

    n_semisafe_sample = 200#300#10
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

    ## Check if any points are outside workspace and remove if so
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
    # psi = 1.0
    dt = 0.1
    dist_eps = 0.05
    mu_dist = (ws_lim[:, 1]-ws_lim[:,0])/n_dim_features
    print("Check: ", mu_dist, onp.max(mu_dist)*0.5)
    rbf_std = 0.1#onp.max(mu_dist) * 0.5

    ### Define RBF Centers/Sigmas manually over the statespace (uniformly or randomly)
    # centers, stds = rbf_means_stds(X=None, X_lim = ws_lim,
    #                                n=x_dim, k=n_dim_features, set_means='random',
    #                                fixed_stds=True, std=rbf_std, nCenters=n_features)

    ### Define RBF centers at data pts
    centers = onp.array(x_all)
    # Add RBF centers in uniform grid over workspace that are epsilon-distance away from any data
    pts = []
    for i in range(ws_lim.shape[0]):
        pts.append(np.linspace(start=ws_lim[i, 0], stop=ws_lim[i, 1], num=5, endpoint=True))
    pts = np.array(pts)
    pts = tuple(pts)
    D = np.meshgrid(*pts)
    means = onp.array([D[i].flatten() for i in range(len(D))]).T
    for i in range(len(means)):
        dist = np.linalg.norm(centers - means[i], axis=1)
        if np.all(dist > 0.25):##2*rbf_std):
            centers = onp.vstack((centers, means[i]))


    stds = onp.ones((centers.shape[0], 1)) * rbf_std
    n_features = centers.shape[0]
    print("Centers: {}, Stds: {}, # Features: {}".format(centers.shape, stds.shape, n_features))

    ## Generate Artificial Safe Pts (far from Data)
    # art_safe_pts = []
    # print(range(len(centers)))
    # for i in range(len(centers)):
    #     dist = np.linalg.norm(x_all-centers[i], axis=1)
    #     if np.all(dist > 2*rbf_std):
    #         art_safe_pts.append(centers[i])
    # art_safe_pts = np.array(art_safe_pts)
    # print(art_safe_pts.shape)

    ## Add negative constraints for obstacle DS

    obs_unsafe_pts = []
    # TODO: Check which centers are inside the ellipsoid of the unsafe ds
    # TODO: Add all centers which are to list of obs_unsafe_pts
    obs_unsafe_pts = np.array(obs_unsafe_pts)


    # Initialize variables
    is_bias = False
    is_slack_both = False
    is_slack_safe = False
    is_semisafe = True
    is_artificial = False

    theta = cp.Variable(n_features)
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
    safe_val = safe_rewards
    unsafe_val = unsafe_rewards
    semisafe_val = semisafe_rewards
    gamma_dyn = np.ones(n_semisafe) * 0.1

    if is_artificial:
        art_safe_val = np.ones(len(art_safe_pts)) * 0.1
        n_artificial = len(art_safe_pts)

    # Initialize cost
    h_cost = 0
    param_cost = 0
    slack_cost = 0

    ## Define Constraints
    start_constraints = time.time()

    constraints = []
    # Safe Constraints
    print("SAFE CONSTRAINTS")
    phis_safe = phi_vec(x_safe)
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

            # Only add unsafe slack if there is a safe point epsilon-close to unsafe point
            dist = np.linalg.norm(x_unsafe[i]-x_safe, axis=1)
            if np.any(dist) <= dist_eps:
                constraints.append(unsafe_slack[i] >= 0.)
            else:
                constraints.append(unsafe_slack[i] == 0)
    else:
        print("adding no slack constraints")
        for i in range(n_unsafe):
            h_cost += cp.sum_squares(theta @ phis_unsafe[i] + bias)  # cost of norm(alpha_i * phi(x,xi) + b)
            constraints.append((theta @ phis_unsafe[i] + bias) <= unsafe_val[i])

    # Usafe Obstacle Pts --> TODO: TEST THIS
    print("UNSAFE CONSTRAINTS")
    phis_obs_unsafe = phi_vec(obs_unsafe_pts)
    for i in range(len(phis_obs_unsafe)):
        h_cost += cp.sum_squares(theta @ phis_obs_unsafe[i] + bias)  # cost of norm(alpha_i * phi(x,xi) + b)
        constraints.append((theta @ phis_obs_unsafe[i] + bias) <= unsafe_val[i])

    if is_semisafe:
        print("BOUNDARY CONSTRAINTS")
        phis_semisafe = phi_vec(x_semisafe)#[phi(x) for x in x_semisafe]
        print("SEMISAFE H CONSTRAINTS")
        for i in range(n_semisafe):
            h_cost += cp.sum_squares(theta @ phis_semisafe[i] + bias)  # cost of norm(alpha_i * phi(x,xi) + b)
            constraints.append((theta @ phis_semisafe[i] + bias) >= semisafe_val[i])

        print("SEMISAFE DERIVATIVE CONSTRAINTS")
        Dphixdots = device_get(
            vmap(lambda x, u: np.dot(jacfwd(phi, argnums=0)(x), dynamics(x, u)),
                 in_axes=(0, 0))(x_semisafe, u_semisafe))
        gamma_xu_fillers = gamma_dyn * np.ones((x_semisafe.shape[0],))
        for i, (this_phi, this_Dphixdot, this_gamma) in enumerate(zip(phis_semisafe, Dphixdots, gamma_xu_fillers)):
            if np.any(np.isnan(this_Dphixdot)):
                print("Invalid Dphixdot", i, this_Dphixdot, x_semisafe[i], u_semisafe[i])
                input()
            constraints.append((theta @ this_Dphixdot+ theta @ this_phi + bias) >= this_gamma)
    print('All CONSTRAINTS DEFINED')
    print("TIME TO SET CONSTRAINTS : ", time.time() -start_constraints ," seconds \n")

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
    else: bias_param = bias

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
    # plt.show()

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

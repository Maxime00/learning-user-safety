import numpy as np

dt = 0.1

x_lim = [0., 1.]
y_lim = [-0.5, 0.5]
z_lim = [0., 0.5]
vdot_lim = [-1., 1.]

ws_lim = np.vstack((x_lim, y_lim, z_lim))

# Discrete time Dynamics
def dt_dynamics(x, u):
    # print(x,u)
    # continuous time: xdot = f(x(t)) + g(x(t))u((t))
    # (xdot=[u1*cos(theta), u1*sin(theta), u2], x = x0 + xdot*dt)
    return np.array(x) + np.array([u[0], u[1], u[2]])

def dt_dynamics_f(x):
    # xdot = f(x(t)) + g(x(t))u((t))
    f = np.eye(3)
    return f#onp.diag(np.array([0.,0.]))

def dt_dynamics_g(x, dt=dt):
    # xdot = f(x(t)) + g(x(t))u((t))
    return np.eye(3)*dt

# RBF Utils

# Initialize RBF Parameters
n_dim_features = 10
x_dim = 3
n_features = n_dim_features**x_dim
u_dim = 3
psi = 1.0
rbf_std=0.05

# Define h model (i.e., RBF)

def rbf_means_stds(X, X_lim, n, k, set_means = 'uniform', fixed_stds=True, std=0.1):
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
            x = np.linspace(start=X_lim[0,0], stop=X_lim[0,1], 
                            num=k, endpoint=True)
            y = np.linspace(start=X_lim[1,0], stop=X_lim[1,1], 
                            num=k, endpoint=True)
            XX, YY = np.meshgrid(x,y)
            means = np.array([XX.flatten(), YY.flatten()]).T
        else: 
            pts = []
            for i in range(X_lim.shape[0]): 
                pts.append(np.linspace(start=X_lim[i,0], stop=X_lim[i,1], num=k, endpoint=True))
            pts = np.array(pts)
            pts = tuple(pts)
            D = np.meshgrid(*pts)
            means = np.array([D[i].flatten() for i in range(len(D))]).T
            
    if set_means == 'random':
        means = np.random.uniform(low=X_lim[:,0], high=X_lim[:,1], size=(k**(1/2),n))
    if set_means =='inputs':
        assert X is not None, 'X invalid data input. Cannot be None-type'
        assert k==X.shape[0], 'Set_means inputs, num kernels must equal num of data points'
        means = X.copy()
    
    # Generate stds    
    if fixed_stds == True: 
        stds = np.ones((k**n,1))*std
    else: 
#         stds = np.random.uniform(low = 0.0001, high = std, size=(k**n,1))
        stds = random.uniform(rng.next(), minval=0.0001, maxval= std, shape=(k**n,1))
    stds = np.squeeze(stds)
    return means, stds

### DS-Based Control

# Simulation parameters
t = 0.
tf = 3.#5.#10.
dt = 0.1

def single_integrator_u(X, goal, limits=None):
    if limits==None:
        max_ang_vel = np.inf
        max_vel = np.inf
    else: 
       max_vel = limits[0]

    print(X, goal)

    x = X[0]
    y = X[1]
    z = X[2]

    e_hat = [x-goal[0], y-goal[1], z-goal[2]]
    ux = e_hat[0]
    uy = e_hat[1]
    uz = e_hat[2]
    # print(u_w, u_speed)
    if ux > max_vel:
        ux = max_vel
    elif ux < -max_vel:
        ux = -max_vel

    if uy > max_vel:
        uy = max_vel
    elif uy < -max_vel:
        uy = -max_vel

    if u_w > max_vel:
        u_w = max_vel
    elif u_w < -max_vel:
        u_w = -max_vel;
    print(ux, uy, uz)
    inputs = np.array([-ux, -uy, -uz])
    return inputs

import numpy as np
import casadi

dt = 0.01

x_lim = [0.2, 0.8]#[0., 1.]
y_lim = [-0.4,0.5]#[-0.5, 0.5]
z_lim = [0.1, 0.6]#[0., 1.]
vdot_lim = [-1., 1.]
xdot_lim = [-0.5, 0.4]
ydot_lim = [-0.8,0.8]
zdot_lim = [-0.8,0.8]

ws_lim = np.vstack((x_lim, y_lim, z_lim, xdot_lim, ydot_lim, zdot_lim))#vdot_lim, vdot_lim, vdot_lim))

# RBF Utils

# Initialize RBF Parameters
n_dim_features = 3
x_dim = 6
n_features = n_dim_features**x_dim            
u_dim = 3
psi = 1.0

mu_dist = (ws_lim[:, 1]-ws_lim[:,0])/n_dim_features

rbf_std=np.max(mu_dist)*0.5

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

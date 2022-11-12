import numpy as onp
import matplotlib.pyplot as plt
from learning_safety_margin.vel_control_utils import *
from matplotlib import cm
import matplotlib.colors as colors

class Obs_Cbf():
    def __init__(self):
        self.center =  [self.xc, self.yc, self.zc] = [0.55, 0.0, 0.035]
        self.axes =  [self.a, self.b, self.c] = [.10, .12, .12 ]
        self.mag=1.
        self.pow=1.
        self.ellipse_lims = [[0.4, 0.7], [-0.2, 0.2], [0., 0.3]]
        self.e_lims = onp.vstack((self.ellipse_lims, xdot_lim, ydot_lim, zdot_lim))
        print(self.e_lims.shape, self.e_lims)

    def check_ellipse_interior(self, x, y, z):
        dist = ((x - self.xc)/self.a)**2 + \
                   ((y-self.yc)/self.b)**2 + \
                   ((z-self.zc)/self.c)**2
        if dist <= 1.:
            in_interior = True
        else:
            in_interior = False
        return in_interior, dist


    def ellipse_cbf_surface(self, x,y,z):
        dist = ((x - self.xc)/self.a)**2 + \
               ((y-self.yc)/self.b)**2 + \
               ((z-self.zc)/self.c)**2
        cbf_val = self.mag * (dist-1)**self.pow
        return cbf_val

    def obs_cbf(self, pt):
        val = self.ellipse_cbf_surface(pt[0], pt[1], pt[2])
        return val

    def plot_cbf(self, num_pts=20):
        xlist = onp.linspace(x_lim[0], x_lim[1], num_pts)
        ylist = onp.linspace(y_lim[0], y_lim[1], num_pts)
        zlist = onp.linspace(z_lim[0], z_lim[1], num_pts)

        XX, YY, ZZ = o.meshgrid(xlist, ylist, zlist)
        pts = onp.array([onp.ravel(XX), onp.ravel(YY), onp.ravel(ZZ)]).T
        vals = onp.array(list(map(self.obs_cbf, pts)))
        print(pts.shape, vals.shape, onp.min(vals), onp.max(vals))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        divnorm = colors.TwoSlopeNorm(vmin=onp.min(vals), vcenter=0, vmax=onp.max(vals))
        im = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=vals, norm=divnorm, cmap=cm.coolwarm_r)
        fig.colorbar(im)
        plt.show()

    def cbf_grad(self, pt):
        x = pt[0]
        y = pt[1]
        z = pt[2]
        gradx = onp.array(
            [(self.pow * 2 * x * ((((x - self.xc)/self.a)**2 + ((y - self.yc)/self.b)**2 + ((z - self.zc)/self.c)**2) - 1)**(self.pow - 1))/self.a**2, #df/dx
             (self.pow * 2 * y * ((((x - self.xc)/self.a)**2 + ((y - self.yc)/self.b)**2 + ((z - self.zc)/self.c)**2) - 1)**(self.pow - 1))/self.b**2, #df/dy
             (self.pow * 2 * z * ((((x - self.xc)/self.a)**2 + ((y - self.yc)/self.b)**2 + ((z - self.zc)/self.c)**2) - 1)**(self.pow - 1))/self.c**2, #df/dz
             0., #df/dxdot
             0., #df/dydot
             0.  #df/dzdot
             ])
        return gradx

    def generate_cbf_values(self, num_pts=100, near_ellipse=False, plot=False):
        # print(ws_lim.shape, ws_lim[0].shape, ws_lim[:,0].shape)
        # print(self.e_lims, ws_lim, z_lim)
        if near_ellipse:
            pts = onp.random.uniform(low=self.e_lims[:, 0], high=self.e_lims[:, 1], size=(num_pts, 6))
        else:
            pts = onp.random.uniform(low=ws_lim[:,0], high=ws_lim[:,1], size=(num_pts, 6))
        print(pts.shape)
        vals = onp.array(list(map(self.obs_cbf, pts)))

        neg_idx = onp.where(vals < 0.)
        print(neg_idx[0].shape, neg_idx)
        neg_vals = vals[neg_idx]
        neg_pts = pts[neg_idx]
        print(neg_vals.shape)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            divnorm = colors.TwoSlopeNorm(vmin=onp.min(vals), vcenter=0, vmax=onp.max(vals))
            im = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=vals, norm=divnorm, cmap=cm.coolwarm_r)
            fig.colorbar(im)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            divnorm = colors.TwoSlopeNorm(vmin=onp.min(vals), vcenter=0, vmax=onp.max(vals))
            im = ax.scatter(neg_pts[:,0], neg_pts[:,1], neg_pts[:,2], c=neg_vals, norm=divnorm, cmap=cm.coolwarm_r)
            fig.colorbar(im)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            plt.show()

        return pts, vals, neg_pts, neg_vals



if __name__ == "__main__":
    cbf = Obs_Cbf()
    cbf.plot_cbf()
    cbf.generate_cbf_values(num_pts=1000, near_ellipse=True)











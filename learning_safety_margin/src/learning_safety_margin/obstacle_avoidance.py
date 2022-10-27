import numpy as np

class Ellipsoid():

    def __init__(self, center_position, axes_lengths, sensitivity=1):

        self.center = np.array(center_position)
        self.axes = np.array(axes_lengths)
        self.rho = sensitivity

    def getDistance(self, cart_pos_world_frame):
        # Reframe cartesian position around obstacle center
        cart_pos = np.array(cart_pos_world_frame) - self.center

        distance = np.sqrt((cart_pos[0]/self.axes[0])**2 + (cart_pos[1]/self.axes[1])**2 + (cart_pos[2]/self.axes[2])**2)

        return distance

    def getGammaDistance(self, cart_pos_world_frame):

        # Reframe cartesian position around obstacle center
        cart_pos = np.array(cart_pos_world_frame) - self.center

        gamma = 1 + np.abs(np.sqrt(np.sum(np.square(cart_pos))) - np.sqrt(np.sum(np.square(cart_pos))/
                          ((cart_pos[0]/self.axes[0])**2 + (cart_pos[1]/self.axes[1])**2 + (cart_pos[2]/self.axes[2])**2)))

        return gamma


    def getGammaGradient(self, cart_pos_world_frame):

        # Reframe cartesian position around obstacle center
        cart_pos = np.array(cart_pos_world_frame) - self.center

        gradientGamma = 2*cart_pos / np.square(self.axes)

        return gradientGamma


def doModulation(obstacle, cart_pos, unmodulated_linear_velocity):

    # Returns modulated linear velocity to avoid obstacle

    gamma = obstacle.getGammaDistance(cart_pos)
    gamma = np.abs(gamma) ** (1/obstacle.rho)

    diagonal = [1 - 1/gamma, 1 + 1/gamma, 1 + 1/gamma]
    D = np.diag(diagonal)

    normal = obstacle.getGammaGradient(cart_pos) / np.linalg.norm(obstacle.getGammaGradient(cart_pos))
    tangent1 = np.array([0 , -normal[2], normal[1]])
    tangent2 = np.cross(normal, tangent1)
    E = np.matrix([normal, tangent1, tangent2]).transpose()

    M = E.dot(D).dot(np.linalg.inv(E))

    modulated_vel = M.dot(unmodulated_linear_velocity)

    return np.array(modulated_vel.A1)

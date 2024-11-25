import casadi as ca
import numpy as np

from traj_opt.models.terrain.base import TerrainBase

STATE_LIMIT_TOL_M = 1

class HillyTerrain(TerrainBase):
    """
    A hilly terrain based on an implicit 3D surface:

        cos(X) + cos(Y) - Z = 0
    """
    def __init__(self):
        # Define symbolic variables using MX
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        z = ca.MX.sym('z')

        # Define the SDF
        sdf_expr = -(np.cos(x) + np.cos(y) - z)
        self.sdf_func = ca.Function('hilly_sdf', [x, y, z], [sdf_expr], ['x', 'y', 'z'], ['sdf'])

        # Calculate the gradient (Jacobian) of the sdf function
        sdf_gradient = ca.jacobian(sdf_expr, ca.vertcat(x, y, z))
        self.sdf_gradient_func = ca.Function('hilly_sdf_gradient', [x, y, z], [sdf_gradient], ['x', 'y', 'z'], ['gradient'])

    def sdf(self, position):
        """
        Evaluate the SDF value at a given position.
        """
        return self.sdf_func(x=position[0], y=position[1], z=position[2])['sdf']

    def normal_vector(self, position):
        """
        Evaluate the gradient of the SDF at the given position to compute the surface normal.
        """
        gradient = self.sdf_gradient_func(x=position[0], y=position[1], z=position[2])['gradient'].T

        normal = gradient / ca.norm_2(gradient)  # Normalize to unit vector
        return normal

    def plot_func(self, X, Y):
        return np.cos(X) + np.cos(Y)
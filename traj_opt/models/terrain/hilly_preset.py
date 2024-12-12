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
        # Define the SDF
        self.sdf_expr = -(np.cos(self.x) + np.cos(self.y) - self.z)

        super().__init__()

    def plot_func(self, X, Y):
        return np.cos(X) + np.cos(Y)
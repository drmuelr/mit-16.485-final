import casadi as ca
import math

from traj_opt.models.terrain.base import TerrainBase

INCLINE_ANGLE_DEG = 20

class InclinedPlaneTerrain(TerrainBase):
    """
    The flat terrain preset. 

    The terrain is a flat plane, with z=0 everywhere.
    """
    def __init__(self):
        # Convert the incline angle to radians
        self.theta = math.radians(INCLINE_ANGLE_DEG)

        # Define the signed distance function for inclined plane terrain
        self.sdf_expr = self.z - ca.tan(self.theta) * self.x 

        super().__init__()

    def plot_func(self, X, Y):
        return math.tan(self.theta)*X

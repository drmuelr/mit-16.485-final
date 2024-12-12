import casadi as ca

from traj_opt.models.terrain.base import TerrainBase

class FlatTerrain(TerrainBase):
    """
    The flat terrain preset. 

    The terrain is a flat plane, with z=0 everywhere.
    """
    def __init__(self):
        # Define the signed distance function for flat terrain (z=0)
        self.sdf_expr = self.z  # Since it's flat at z=0, the SDF is just the z-coordinate

        super().__init__()

    def plot_func(self, X, Y):
        return 0*X

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
        # Define symbolic variables for the position
        x_pos = ca.SX.sym('x_pos')
        y_pos = ca.SX.sym('y_pos')
        z_pos = ca.SX.sym('z_pos')

        # Convert the incline angle to radians
        self.theta = math.radians(INCLINE_ANGLE_DEG)

        # Define the signed distance function for inclined plane terrain
        sdf_expr = z_pos - ca.tan(self.theta) * x_pos 
        self.sdf_func = ca.Function('flat_sdf', [x_pos, y_pos, z_pos], [sdf_expr], ['x', 'y', 'z'], ['sdf'])

        # Calculate the gradient (Jacobian) of the sdf function
        sdf_gradient = ca.jacobian(sdf_expr, ca.vertcat(x_pos, y_pos, z_pos))
        self.sdf_gradient_func = ca.Function('flat_sdf_gradient', [x_pos, y_pos, z_pos], [sdf_gradient], ['x', 'y', 'z'], ['gradient'])

    def sdf(self, position):
        # Evaluate the SDF at the given position
        return self.sdf_func(x=position[0], y=position[1], z=position[2])['sdf']
    
    def normal_vector(self, position):
        # Evaluate the gradient of the SDF at the given position to get the normal
        gradient = self.sdf_gradient_func(x=position[0], y=position[1], z=position[2])['gradient'].T
        
        # Normalize the gradient to get a unit normal vector
        normal = gradient / ca.norm_2(gradient)
        return normal
    
    def plot_func(self, X, Y):
        return math.tan(self.theta)*X

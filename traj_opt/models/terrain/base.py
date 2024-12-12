from abc import ABC, abstractmethod
import casadi as ca

class TerrainBase(ABC):
    """
    Defines the interface for all terrain classes.

    Defines two abstract methods: 
    
        - sdf(): which returns the signed distance function of the terrain as
            a casadi Function given an (x, y, z) point as input.

        - normal_vector():  returns the normal vector of the surface at a given
            (x, y, z) point on the terrain.

    Defines x, y, z ca.MX.sym variables to be used in the derived classes.

    Requires an sdf_expr attribute to be defined in the derived class __init__,
    as a symbolic expression in terms of x, y, and z.
    """

    # Define symbolic variables using MX
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    z = ca.MX.sym('z')

    def __init__(self):
        # Define the SDF casadi function that operates on symbolic MX variables
        self.sdf_func = ca.Function('mesh_sdf', [self.x, self.y, self.z], [self.sdf_expr], ['x', 'y', 'z'], ['sdf'])

        # Calculate the gradient (Jacobian) of the sdf function
        sdf_gradient = ca.jacobian(self.sdf_expr, ca.vertcat(self.x, self.y, self.z))
        self.sdf_gradient_func = ca.Function('flat_sdf_gradient', [self.x, self.y, self.z], [sdf_gradient], ['x', 'y', 'z'], ['gradient'])

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

    
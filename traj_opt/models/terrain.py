from abc import ABC, abstractmethod

import casadi as ca

class TerrainBase(ABC):
    """
    Defines the interface for all terrain classes.

    Defines two abstract methods: 
    
        - sdf(): which returns the signed distance function of the terrain as
            a casadi Function given an (x, y, z) point as input.

        - sdf_jacob():  returns the jacobian of the signed distance function 
            as a casadi Function given an (x, y, z) point as input.
    """

    @abstractmethod
    def sdf(self, position):
        pass

    @abstractmethod
    def normal_vector(self, position):
        pass

    
class FlatTerrain(TerrainBase):
    """
    The flat terrain preset. 

    The terrain is a flate plane, with z=0 everywhere.
    """
    def __init__(self):
        x_pos = ca.SX.sym('x_pos', 1)
        y_pos = ca.SX.sym('x_pos', 1)
        z_pos = ca.SX.sym('x_pos', 1)

        self.sdf_func = ca.Function('flat_sdf', [x_pos, y_pos, z_pos], [z_pos], ['x', 'y', 'z'], ['sdf'])
        self.jacobian = self.sdf_func.jacobian()

    def sdf(self, position):
        return self.sdf_func(x=position[0], y=position[1], z=position[2])['sdf']
    
    def normal_vector(self, position):
        jacob = self.jacobian(x=position[0], y=position[1], z=position[2])

        return ca.vertcat(
            jacob["jac_sdf_x"],
            jacob["jac_sdf_y"],
            jacob["jac_sdf_z"]
        )

    



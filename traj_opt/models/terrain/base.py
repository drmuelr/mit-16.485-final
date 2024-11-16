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

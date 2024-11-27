from abc import ABC, abstractmethod

class TerrainBase(ABC):
    """
    Defines the interface for all terrain classes.

    Defines two abstract methods: 
    
        - sdf(): which returns the signed distance function of the terrain as
            a casadi Function given an (x, y, z) point as input.

        - normal_vector():  returns the normal vector of the surface at a given
            (x, y, z) point on the terrain.
    """

    @abstractmethod
    def sdf(self, position):
        pass

    @abstractmethod
    def normal_vector(self, position):
        pass

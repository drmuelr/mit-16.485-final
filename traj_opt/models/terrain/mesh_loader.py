import casadi as ca
import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_sdf
from pathlib import Path

from traj_opt.models.terrain.base import TerrainBase

PADDING = 5
NUM_POINTS = 50

class MeshLoader(TerrainBase):
    """
    Creates a terrain model by making an SDF lookup table 
    from a mesh file (.obj).

    Parameters
    ----------
    mesh_file
        Filename of the .obj file to open.
    
    """
    def __init__(self, mesh_file: str):
        if not Path(mesh_file).suffix == ".obj":
            raise ValueError("Mesh file must be an .obj file.")
        
        # Load mesh and extract bounds
        self.mesh = trimesh.load(mesh_file)
        xmin, ymin, zmin = self.mesh.bounds[0]
        xmax, ymax, zmax = self.mesh.bounds[1]

        # Define X, Y, Z grids
        X = np.linspace(xmin-PADDING, xmax+PADDING, NUM_POINTS)
        Y = np.linspace(ymin-PADDING, ymax+PADDING, NUM_POINTS)
        Z = np.linspace(zmin-PADDING, zmax+PADDING, NUM_POINTS)

        X_grid, Y_grid, Z_grid = np.meshgrid(X, Y, Z, indexing='ij')

        # Flatten data and stack
        X_grid = X_grid.ravel(order="F")
        Y_grid = Y_grid.ravel(order="F")
        Z_grid = Z_grid.ravel(order="F")
        grid = np.vstack([X_grid, Y_grid, Z_grid]).T

        # Compute SDF at each grid point
        sdf = mesh_to_sdf(self.mesh, grid, sign_method='depth')

        # Create a CasADi interpolant for the SDF
        sdf_lut = ca.interpolant('sdf_lut','linear', [X, Y, Z], sdf)

        # Define symbolic variables using MX
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        z = ca.MX.sym('z')
        sdf_expr = sdf_lut(ca.vertcat(x, y, z))

        # Define the SDF casadi function that operates on symbolic MX variables
        self.sdf_func = ca.Function('mesh_sdf', [x, y, z], [sdf_expr], ['x', 'y', 'z'], ['sdf'])

        # Calculate the gradient (Jacobian) of the sdf function
        sdf_gradient = ca.jacobian(sdf_expr, ca.vertcat(x, y, z))
        self.sdf_gradient_func = ca.Function('mesh_sdf_gradient', [x, y, z], [sdf_gradient], ['x', 'y', 'z'], ['gradient'])

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
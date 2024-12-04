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
        self.sdf_expr = sdf_lut(ca.vertcat(self.x, self.y, self.z))

        super().__init__()

    def plot_surface(self, ax):
        """
        Plot the mesh on the given MPL axes.
        """
        ax.plot_trisurf(
            self.mesh.vertices[:, 0], 
            self.mesh.vertices[:,1], 
            triangles=self.mesh.faces, 
            Z=self.mesh.vertices[:,2]
        ) 
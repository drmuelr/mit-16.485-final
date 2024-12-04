import casadi as ca
import numpy as np
from scipy.interpolate import griddata
from pathlib import Path

from traj_opt.models.terrain.base import TerrainBase
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NUM_POINTS = 40

class VoxbloxSdfLoader(TerrainBase):
    """
    Creates a terrain model by making an SDF lookup table 
    from an SDF saved from voxblox.

    Parameters
    ----------
    sdf_file
        Filename of the .npy file to open.
    
    """
    def __init__(self, sdf_file: str):
        if not Path(sdf_file).suffix == ".npy":
            raise ValueError("Mesh file must be an .obj file.")
    
        # Load the saved .npy file
        data = np.load(sdf_file, allow_pickle=True).item()

        # Extract x,y,z coordinates
        points = data['points']

        # Extract SDF values associated with points
        distances = data['distances']

        # Create a regular grid from the sparse points and distances
        self.grid_sdf, self.grid_x, self.grid_y, self.grid_z = self.create_sdf_grid(points, distances)
        
        # Create casadi lookup table from regular grid
        sdf_lut = ca.interpolant(
            'SDF', 'linear', [self.grid_x, self.grid_y, self.grid_z], self.grid_grid_sdf.flatten(order='F')
        )

        self.sdf_expr = sdf_lut(ca.vertcat(self.x, self.y, self.z))

        super().__init__()

    def create_sdf_grid(self, points, distances):
        """
        Interpolates sparse SDF data onto a regular grid.
        """
        # Extract min and max values for each axis
        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, z_max = points.max(axis=0)

        # Create regular grid
        x = np.linspace(x_min, x_max, NUM_POINTS)
        y = np.linspace(y_min, y_max, NUM_POINTS)
        z = np.linspace(z_min, z_max, NUM_POINTS)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")

        # Flatten the grid for interpolation
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

        # Interpolate SDF values onto the grid
        grid_sdf = griddata(points, distances, grid_points, method='linear', fill_value=np.nan)
        grid_sdf = grid_sdf.reshape(grid_x.shape)

        return grid_sdf, x, y, z

    def plot_zero_crossing(self, ax, threshold=0.05):
        """
        Plots the zero-crossing points of the interpolated SDF grid.
        """
        zero_crossing_mask = np.abs(self.grid_sdf) < threshold

        # Get indices of zero-crossing points
        zero_x, zero_y, zero_z = np.where(zero_crossing_mask)

        # Map indices back to grid coordinates
        x_coords = self.grid_x[zero_x]
        y_coords = self.grid_y[zero_y]
        z_coords = self.grid_z[zero_z]

        # Get sdf values at zero-crossing points
        sdf_vals = self.grid_sdf[zero_x, zero_y, zero_z]

        # Plot zero-crossing points
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x_coords, y_coords, z_coords, c=sdf_vals, cmap='coolwarm', s=5)

        # Add colorbar and labels
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Signed Distance (SDF)", fontsize=12)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Zero-Crossing Points of SDF')
        plt.show()
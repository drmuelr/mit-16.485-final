#!/usr/bin/env python3
"""
Takes a esdf point cloud .npy file (saved from voxblox_ros
esdf_pointcloud topic) as a command line argument and plots
the zero-crossing points using Matplotlib 3D plotting. 
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


def plot_zero_crossing(data, threshold):
    """
    Plots only the zero-crossing points from the SDF point cloud data.
    """
    # Extract points and distances
    points = data['points']
    distances = data['distances']

    if points.shape[1] != 3:
        print("Error: 'points' should have 3 columns (x, y, z).")
        return
    if len(points) != len(distances):
        print("Error: 'points' and 'distances' lengths do not match.")
        return

    # Find zero-crossing points
    zero_crossing_mask = np.abs(distances) < threshold
    zero_crossing_points = points[zero_crossing_mask]
    zero_crossing_distances = distances[zero_crossing_mask]

    if zero_crossing_points.shape[0] == 0:
        print("No zero-crossing points found within the given threshold.")
        return

    # Extract coordinates
    x = zero_crossing_points[:, 0]
    y = zero_crossing_points[:, 1]
    z = zero_crossing_points[:, 2]

    # Plot the zero-crossing points
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=zero_crossing_distances, cmap='coolwarm', s=5)

    # Add colorbar and labels
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Signed Distance (SDF)", fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Zero-Crossing Points (Threshold: {threshold})")

    # Set the axis limits
    ax.set_xlim([np.min(x), np.max(x)])
    ax.set_ylim([np.min(y), np.max(y)])
    ax.set_zlim([np.min(z), np.max(z)])

    ax.set_proj_type('ortho')
    ax.set_aspect('equal')

    # Show the plot
    plt.show()


if __name__ == "__main__":

    # Read first command line arg (mesh file path)
    if len(sys.argv) < 2:
        print("Please supply a esdf file path as a command line argument.")
        sys.exit(1)
    if len(sys.argv) > 2:
        print("Please supply only one command line argument (esdf file path).")
        sys.exit(1)

    esdf_file_path = sys.argv[1]

    # Load the saved .npy file
    data = np.load(esdf_file_path, allow_pickle=True).item()

    # Plot the zero-crossing points
    plot_zero_crossing(data, 0.1)

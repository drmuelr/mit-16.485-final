#!/usr/bin/env python3
"""
Takes a mesh file path as a command line argument and 
plots the mesh using Matplotlib 3D plotting.
"""

import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


if __name__ == "__main__":

    # Read first command line arg (mesh file path)
    if len(sys.argv) < 2:
        print("Please supply a mesh file path as a command line argument.")
        sys.exit(1)
    if len(sys.argv) > 2:
        print("Please supply only one command line argument (mesh file path).")
        sys.exit(1)

    mesh_file_path = sys.argv[1]

    # Load the mesh using trimesh
    mesh = trimesh.load(mesh_file_path)

    # Extract vertex and face data
    vertices = mesh.vertices
    faces = mesh.faces

    # Create a Matplotlib 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=faces,
        cmap='viridis',
        edgecolor='k',
        linewidth=0.5,
        alpha=0.5
    )

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_proj_type('ortho')
    ax.set_aspect('equal')

    plt.show()

    

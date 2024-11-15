import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy.spatial.transform import Rotation as R

def animate_solution(optimizer, solution):
    """
    Animates the position and orientation of the robot as it follows the optimal trajectory.
    """
    # Extract the solution
    positions = [solution.value(optimizer.robot_model.body_position[k]) for k in range(optimizer.config.num_steps + 1)]
    quaternions = [solution.value(optimizer.robot_model.body_quat[k]) for k in range(optimizer.config.num_steps + 1)]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect([1, 1, 1])

    # Define the coordinate axes for visualization
    x_cords = np.array([1, 0, 0])
    y_cords = np.array([0, 1, 0])
    z_cords = np.array([0, 0, 1])

    # Plot initial frame
    (x_line,) = ax.plot([0, 1], [0, 0], [0, 0], "red")
    (y_line,) = ax.plot([0, 0], [0, 1], [0, 0], "green")
    (z_line,) = ax.plot([0, 0], [0, 0], [1], "blue")

    # Set plot limits based on data range
    pos_array = np.array(positions)
    ax.set_xlim([pos_array[:, 0].min() - 1, pos_array[:, 0].max() + 1])
    ax.set_ylim([pos_array[:, 1].min() - 1, pos_array[:, 1].max() + 1])
    ax.set_zlim([pos_array[:, 2].min() - 1, pos_array[:, 2].max() + 1])

    # Function to update coordinate frame and position
    def update_frame(i):
        pos_i = positions[i]
        rotation = R.from_quat(quaternions[i]).as_matrix()
        
        x_end = pos_i + rotation @ x_cords
        y_end = pos_i + rotation @ y_cords
        z_end = pos_i + rotation @ z_cords

        x_line.set_data([pos_i[0], x_end[0]], [pos_i[1], x_end[1]])
        x_line.set_3d_properties([pos_i[2], x_end[2]])

        y_line.set_data([pos_i[0], y_end[0]], [pos_i[1], y_end[1]])
        y_line.set_3d_properties([pos_i[2], y_end[2]])

        z_line.set_data([pos_i[0], z_end[0]], [pos_i[1], z_end[1]])
        z_line.set_3d_properties([pos_i[2], z_end[2]])

        return x_line, y_line, z_line

    # Create the animation
    fps = 1/solution.value(optimizer.h)
    ani = animation.FuncAnimation(fig, update_frame, frames=optimizer.config.num_steps + 1, repeat=False, interval=1000/fps)

    # Save as GIF or show the animation
    writergif = animation.PillowWriter(fps=fps)
    ani.save("latest_trajectory.gif", writer=writergif)
    plt.show()

    # # Plot the spring elongation over time
    # spring_elongation = [solution.value(optimizer.robot_model.spring_elongation[k]) for k in range(optimizer.config.num_steps + 1)]
    # plt.plot(spring_elongation)
    # plt.xlabel("Time")
    # plt.ylabel("Spring Elongation")
    # plt.title("Spring Elongation vs Time")
    # plt.show()

    # Plot the control forces over time
    control_forces = [solution.value(optimizer.robot_model.control_forces[k]) for k in range(optimizer.config.num_steps)]
    control_forces = np.array(control_forces)
    plt.plot(control_forces)
    plt.xlabel("Time")
    plt.ylabel("Control Forces")
    plt.title("Control Forces vs Time")
    plt.legend(["Force 1", "Force 2", "Force 3", "Force 4"])
    plt.show()

    # # Plot the contact forces over time
    # contact_forces = [solution.value(optimizer.robot_model.contact_force[k]) for k in range(optimizer.config.num_steps + 1)]
    # contact_forces = np.array(contact_forces)
    # plt.plot(contact_forces)
    # plt.xlabel("Time")
    # plt.ylabel("Contact Forces")
    # plt.title("Contact Forces vs Time")
    # plt.legend(["Force X", "Force Y", "Force Z"])
    # plt.show()

    # Plot the contact location over time
    contact_points = [solution.value(optimizer.robot_model.body_position[k]) for k in range(optimizer.config.num_steps + 1)]
    contact_points = np.array(contact_points)
    plt.plot(contact_points)
    plt.xlabel("Time")
    plt.ylabel("Contact Location")
    plt.title("Contact Location vs Time")
    plt.legend(["X", "Y", "Z"])
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy.spatial.transform import Rotation as R

def animate_solution(optimizer, solution):
    """
    Animates the position and orientation of the robot as it follows the optimal trajectory,
    and plots a plane at z=0.
    """
    # Extract the solution
    positions = [solution.value(optimizer.robot_model.position_world[k]) for k in range(optimizer.config.num_steps + 1)]
    quaternions = [solution.value(optimizer.robot_model.q_body_to_world[k]) for k in range(optimizer.config.num_steps + 1)]

    contact_point_location = [solution.value(optimizer.robot_model.contact_point_location[k]) for k in range(optimizer.config.num_steps+1)]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect([1, 1, 1])

    scaling_factor = 0.5
    x_cords = scaling_factor * np.array([1, 0, 0])
    y_cords = scaling_factor * np.array([0, 1, 0])
    z_cords = scaling_factor * np.array([0, 0, 1])

    # Plot initial frame
    (x_line,) = ax.plot([0, 1], [0, 0], [0, 0], "red")
    (y_line,) = ax.plot([0, 0], [0, 1], [0, 0], "green")
    (z_line,) = ax.plot([0, 0], [0, 0], [1], "blue")

    # Line from position to contact point
    (contact_line,) = ax.plot([0, 0], [0, 0], [0, 0], "purple")

    # Set plot limits based on data range
    pos_array = np.array(positions)
    ax.set_xlim([pos_array[:, 0].min() - 1, pos_array[:, 0].max() + 1])
    ax.set_ylim([pos_array[:, 1].min() - 1, pos_array[:, 1].max() + 1])
    ax.set_zlim([pos_array[:, 2].min() - 1, pos_array[:, 2].max() + 1])

    # Plot a plane at z=0
    x_plane = np.linspace(pos_array[:, 0].min() - 1, pos_array[:, 0].max() + 1, 10)
    y_plane = np.linspace(pos_array[:, 1].min() - 1, pos_array[:, 1].max() + 1, 10)
    X, Y = np.meshgrid(x_plane, y_plane)
    Z = optimizer.terrain_model.plot_func(X, Y)
    ax.plot_surface(X, Y, Z, color="gray", alpha=0.5, rstride=100, cstride=100)

    # Function to update coordinate frame and position
    def update_frame(i):
        pos_i = positions[i]
        rotation = R.from_quat(quaternions[i]).as_matrix()

        contact_point = contact_point_location[i]

        x_end = pos_i + rotation @ x_cords
        y_end = pos_i + rotation @ y_cords
        z_end = pos_i + rotation @ z_cords

        x_line.set_data([pos_i[0], x_end[0]], [pos_i[1], x_end[1]])
        x_line.set_3d_properties([pos_i[2], x_end[2]])

        y_line.set_data([pos_i[0], y_end[0]], [pos_i[1], y_end[1]])
        y_line.set_3d_properties([pos_i[2], y_end[2]])

        z_line.set_data([pos_i[0], z_end[0]], [pos_i[1], z_end[1]])
        z_line.set_3d_properties([pos_i[2], z_end[2]])

        # Update contact line
        contact_line.set_data([pos_i[0], contact_point[0]], [pos_i[1], contact_point[1]])
        contact_line.set_3d_properties([pos_i[2], contact_point[2]])

        return x_line, y_line, z_line, contact_line

    # Create the animation
    fps = 1 / solution.value(optimizer.h) / 2
    ani = animation.FuncAnimation(fig, update_frame, frames=optimizer.config.num_steps + 1, repeat=True, interval=1000 / fps)

    # Save as GIF or show the animation
    gif_filepath = optimizer.config.save_solution_as[:-3] + "gif"
    writergif = animation.PillowWriter(fps=fps)
    ani.save(gif_filepath, writer=writergif)
    plt.show()

    fig, axs = plt.subplots(3,2, figsize=(15,15))

    # Plot the control forces over time
    control_forces = [solution.value(optimizer.robot_model.control_thrusts[k]) for k in range(optimizer.config.num_steps+1)]
    control_forces = np.array(control_forces)
    axs[0,0].plot(control_forces)
    axs[0,0].set_xlabel("Time")
    axs[0,0].set_ylabel("Control Forces")
    axs[0,0].set_title("Control Forces vs Time")
    axs[0,0].legend(["Thrust_1", "Thrust_2", "Thrust_3", "Thrust_4"])

    # Plot the control moments over time
    control_moments = [solution.value(optimizer.robot_model.control_moment_body[k]) for k in range(optimizer.config.num_steps+1)]
    control_moments = np.array(control_moments)
    axs[0,1].plot(control_moments)
    axs[0,1].set_xlabel("Time")
    axs[0,1].set_ylabel("Control Moment")
    axs[0,1].set_title("Control Moment vs Time")
    axs[0,1].legend(["Moment"])

    # Plot position over time
    positions = np.array(positions)
    axs[1,0].plot(positions)
    axs[1,0].set_xlabel("Time")
    axs[1,0].set_ylabel("Position")
    axs[1,0].set_title("Position vs Time")
    axs[1,0].legend(["X", "Y", "Z"])
    
    # Plot the velocity over time
    velocities = [solution.value(optimizer.robot_model.velocity_world[k]) for k in range(optimizer.config.num_steps+1)]
    velocities = np.array(velocities)
    axs[1,1].plot(velocities)
    axs[1,1].set_xlabel("Time")
    axs[1,1].set_ylabel("Velocity")
    axs[1,1].set_title("Velocity vs Time")
    axs[1,1].legend(["X", "Y", "Z"])
    

    # Plot the spring force over time
    spring_force = [solution.value(optimizer.robot_model.spring_force[k]) for k in range(optimizer.config.num_steps)]
    spring_force = np.array(spring_force)
    axs[2,0].plot(spring_force)
    axs[2,0].set_xlabel("Time")
    axs[2,0].set_ylabel("Spring Force")
    axs[2,0].set_title("Spring Force vs Time")
    axs[2,0].legend(["Force"])

    # Plot the spring elongation
    spring_elongation = [solution.value(optimizer.robot_model.spring_elongation[k]) for k in range(optimizer.config.num_steps+1)]
    spring_elongation = np.array(spring_elongation)
    axs[2,1].plot(spring_elongation)
    axs[2,1].set_xlabel("Time")
    axs[2,1].set_ylabel("Spring Elongation")

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2,2, figsize=(15,15))

    # Plot the friction force over time
    friction_impulse = [solution.value(optimizer.robot_model.friction_impulse[k]) for k in range(optimizer.config.num_steps)]
    friction_impulse = np.array(friction_impulse)
    axs[0,0].plot(friction_impulse)
    axs[0,0].set_xlabel("Time")
    axs[0,0].set_ylabel("Friction Impulse")
    axs[0,0].set_title("Friction Impulse vs Time")
    axs[0,0].legend(["X", "Y", "Z"])

    # Plot the friction impulse over time
    contact_force = [solution.value(optimizer.robot_model.contact_force_world[k]) for k in range(optimizer.config.num_steps)]
    contact_force = np.array(contact_force)
    axs[0,1].plot(contact_force)
    axs[0,1].set_xlabel("Time")
    axs[0,1].set_ylabel("Contact Force")
    axs[0,1].set_title("Contact Force vs Time")
    axs[0,1].legend(["X", "Y", "Z"])

    # Plot the contact moment over time
    contact_moment = [solution.value(optimizer.robot_model.contact_moment_body[k]) for k in range(optimizer.config.num_steps)]
    contact_moment = np.array(contact_moment)
    axs[1,0].plot(contact_moment)
    axs[1,0].set_xlabel("Time")
    axs[1,0].set_ylabel("Contact Moment")
    axs[1,0].set_title("Contact Moment vs Time")
    axs[1,0].legend(["X", "Y", "Z"])

    plt.tight_layout()
    plt.show()
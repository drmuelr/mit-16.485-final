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
    Z = np.zeros_like(X)  # Plane at z=0
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
    fps = 1 / solution.value(optimizer.h)
    ani = animation.FuncAnimation(fig, update_frame, frames=optimizer.config.num_steps + 1, repeat=True, interval=1000 / fps)

    # Save as GIF or show the animation
    gif_filepath = "results/latest_trajectory.gif"
    writergif = animation.PillowWriter(fps=fps)
    ani.save(gif_filepath, writer=writergif)
    plt.show()

    # Plot the control forces over time
    control_forces = [solution.value(optimizer.robot_model.control_thrusts[k]) for k in range(optimizer.config.num_steps+1)]
    control_forces = np.array(control_forces)
    plt.plot(control_forces)
    plt.xlabel("Time")
    plt.ylabel("Control Forces")
    plt.title("Control Forces vs Time")
    plt.legend(["Thrust_1", "Thrust_2", "Thrust_3", "Thrust_4"])
    plt.show()

    # Plot the control moments over time
    control_moments = [solution.value(optimizer.robot_model.control_moment_body[k]) for k in range(optimizer.config.num_steps+1)]
    control_moments = np.array(control_moments)
    plt.plot(control_moments)
    plt.xlabel("Time")
    plt.ylabel("Control Moment")
    plt.title("Control Moment vs Time")
    plt.legend(["Moment"])
    plt.show()

    # Plot position over time
    positions = np.array(positions)
    plt.plot(positions)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Position vs Time")
    plt.legend(["X", "Y", "Z"])
    plt.show()
    
    # Plot the velocity over time
    velocities = [solution.value(optimizer.robot_model.velocity_world[k]) for k in range(optimizer.config.num_steps+1)]
    velocities = np.array(velocities)
    plt.plot(velocities)
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.title("Velocity vs Time")
    plt.legend(["X", "Y", "Z"])
    plt.show()
    

    # Plot the spring force over time
    spring_force = [solution.value(optimizer.robot_model.spring_force[k]) for k in range(optimizer.config.num_steps)]
    spring_force = np.array(spring_force)
    plt.plot(spring_force)
    plt.xlabel("Time")
    plt.ylabel("Spring Force")
    plt.title("Spring Force vs Time")
    plt.legend(["Force"])
    plt.show()

    # Plot the spring elongation
    spring_elongation = [solution.value(optimizer.robot_model.spring_elongation[k]) for k in range(optimizer.config.num_steps+1)]
    spring_elongation = np.array(spring_elongation)
    plt.plot(spring_elongation)
    plt.xlabel("Time")
    plt.ylabel("Spring Elongation")
    plt.show()

    # # Plot the SDF of the contact points over time
    # sdf = [solution.value(optimizer.robot_model.sdf_value[k]) for k in range(optimizer.config.num_steps+1)]
    # sdf = np.array(sdf)
    # plt.plot(sdf)
    # plt.xlabel("Time")
    # plt.ylabel("SDF")
    # plt.show()

    # # Plot the velocity over time
    # velocities = [solution.value(optimizer.robot_model.velocity_world[k]) for k in range(optimizer.config.num_steps+1)]
    # velocities = np.array(velocities)
    # positions = np.array(positions)
    # plt.plot(positions[:, 2])
    # plt.plot(velocities)
    # plt.xlabel("Time")
    # plt.ylabel("Velocity")
    # plt.legend(["Position", "Velocity"])
    # plt.show()

    # # Plot the contact force over time
    # contact_force = [solution.value(optimizer.robot_model.spring_force[k]) for k in range(optimizer.config.num_steps+1)]
    # contact_force = np.array(contact_force)
    # plt.plot(contact_force)
    # plt.xlabel("Time")
    # plt.ylabel("Spring Force")
    # plt.show()

    # # Plot the contact force, control force, gravity, and acceleration over time
    # # on one plot
    # gravity = np.array([-9.81]*(optimizer.config.num_steps+1))
    # acceleration = gravity + control_forces + contact_force
    # plt.plot(contact_force)
    # plt.plot(control_forces)
    # plt.plot(acceleration)
    # plt.plot(sdf)
    # plt.xlabel("Time")
    # plt.ylabel("Force")
    # plt.legend(["Contact Force", "Control Force", "Acceleration", "SDF"])
    # plt.show()

    # # # Plot the spring elongation over time
    # # spring_elongation = [solution.value(optimizer.robot_model.spring_elongation[k]) for k in range(optimizer.config.num_steps)]
    # # spring_elongation = np.array(spring_elongation)
    # # plt.plot(spring_elongation)
    # # plt.xlabel("Time")
    # # plt.ylabel("Spring Elongation")
    # # plt.show()

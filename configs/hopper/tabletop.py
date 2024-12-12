"""
The configuration for a trajectory through the mountains mesh terrain.
"""

from configs import TrajOptConfig
from traj_opt.models import (
    HoppingSoftfly,
)


config = TrajOptConfig(
    save_solution_as = "results/solution.npz",
    initial_guess_path = "results/hopper/tabletop/solution.npz",
    robot_class = HoppingSoftfly,
    terrain_source = "sdfs/tabletop.npy",
    initial_state = {
        "position": [-0.5, 0.8, 2.1],
        "velocity": [0.0, 0.0, 0.0],
        "q_body_to_world": [0.0, 0.0, 0.0, 1.0], # World2Body (x, y, z, w)
        "angular_velocity_body": [0.0, 0.0, 0.0],
    },
    final_state = {
        "position": [-0.5, 1.5, 2.1],
        "velocity": [0.0, 0.0, 0.0],
        "q_body_to_world": [0.0, 0.0, 0.0, 1.0], # World2Body (x, y, z, w)
        "angular_velocity_body": [0.0, 0.0, 0.0],
    },
    cost_weights = {
        'control_force': 100.0,
        'control_moment': 0.0,
        'contact_force': 0.0,
        'T': 0.01
    },
    state_limits = {
        "position_X": [-1.0, 0.0],
        "position_Y": [0.0, 2.0],
        "position_Z": [1.0, 4.0],
        "velocity": [-10.0, 10.0],
        "angular_velocity_body": [-30.0, 30.0],
    },
    num_steps = 100,
    max_time_s = 5.0,
    ipopt_print_level = 5
)
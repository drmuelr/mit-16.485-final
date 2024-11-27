from traj_opt.models import (
    HoppingSoftfly,
    FlatTerrain,
    InclinedPlaneTerrain,
    HillyTerrain,
    RobotBase,
    Softfly,
)
from traj_opt.models.terrain.base import TerrainBase

import math


class TrajOptConfig:

    save_solution_as: str = "results/solution.npz"
    """
    The name of the file to save the solution to.
    """
    
    initial_guess_path: str = "results/hopper/none.npz"
    """
    The name of the .npz file to load the initial guess from.

    If no file is found, no initial guess is used.
    """

    robot_class: type[RobotBase] = HoppingSoftfly
    """
    The name of the robot model being used for optimization.
    """

    terrain_source: type[TerrainBase] = "meshes/mountains.obj"
    """
    The source of the terrain data, which can be a preset class
    or a path to a '.obj' mesh file.

    Currently supported presets are:
        [
            FlatTerrain,
            InclinedPlaneTerrain,
            HillyTerrain
        ]
    """

    initial_state: dict[str, float | list[float]] = {
        "position": [-15, 0.0, 13.0],
        "velocity": [0.0, 0.0, 0.0],
        "q_body_to_world": [0.0, 0.0, 0.0, 1.0], # World2Body (x, y, z, w)
        "angular_velocity_body": [0.0, 0.0, 0.0],
    }
    """
    The initial state of the robot.
    """

    final_state: dict[str, float | list[float]] = {
        "position": [-15, 0.0, 13.0],
        "velocity": [0.0, 0.0, 0.0],
        "q_body_to_world": [0.0, 0.0, 0.0, 1.0], # World2Body (x, y, z, w)
        "angular_velocity_body": [0.0, 0.0, 0.0],
    }
    """
    The final state of the robot.
    """

    cost_weights: dict[str, float] =    {
        'control_force': 0.0,
        'control_moment': 0.0,
        'contact_force': 0.0,
        'T': 1.0
    }
    """
    Cost function weights for the optimization problem.
    Keys correspond with casadi variable names defined in the model.

    OM: I've had the best luck with:
    {
        'control_force': 100.0,
        'control_moment': 0.0,
        'contact_force': 0.0,
        'T': 0.00001
    }
    These particular weights may look weird, but they seem to work.
    You can perturb either of them by 10x or 100x to reasonably 
    affect the trajectory (i.e. should still converge but will
    have the desired effect), but any more than this will probably
    throw off the solver too much. 
    """

    state_limits: dict[str, list[float]] = {
        "position_X": [-50, 50],
        "position_Y": [-0.5, 0.5],
        "position_Z": [-50.0, 50],
        "velocity": [-20.0, 20.0],
        "angular_velocity_body": [-30.0, 30.0],
    }
    """
    The limits for the state variables in the optimization problem.

    OM: These seemed to help a lot with convergence, probably because
    they don't let the solver state escape too far from what we expect.
    Seems like we can make these pretty tight, at least with a good 
    initial guess.
    """

    num_steps: int = 100
    """
    The number of steps that the trajectory is discretely broken into.

    This is the number of collocation points in the optimization problem.

    h = T / num_steps
    """

    max_time: float = 5
    """
    The maximum time allowed to reach the final position from the initial position.
    """

    ipopt_print_level: int = 5
    """
    The verbosity level of IPOPT solver output.

    Ranges from 0 to 12, with 0 being the least verbose and 12 being the most verbose.
    """
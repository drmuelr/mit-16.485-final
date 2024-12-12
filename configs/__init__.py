import importlib.util
from pathlib import Path
from pydantic import BaseModel

from traj_opt.models import (
    RobotBase,
    TerrainBase
)

def load_config(file_name: str):
    """
    Dynamically load the `config` object from a Python file.

    Parameters
    ----------
    file_name : str
        The name or path of the Python config file to import from.

    Returns
    -------
    config : TrajOptConfig
        The `config` object from the specified file.

    """
    # Ensure the file exists
    file_path = Path(file_name)
    if not file_path.is_file():
        raise FileNotFoundError(f"The file {file_name} does not exist.")
    
    # Load the module from the specified file
    module_name = file_path.stem  # Extract module name from file (without extension)
    spec = importlib.util.spec_from_file_location(module_name, file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Retrieve the `config` object from the module
    if not hasattr(module, "config"):
        raise AttributeError(f"The module {module_name} does not contain a 'config' object.")
    
    return module.config



class TrajOptConfig(BaseModel):

    save_solution_as: str
    """
    The name of the file to save the solution to.
    """
    
    initial_guess_path: str
    """
    The name of the .npz file to load the initial guess from.

    If no file is found, no initial guess is used.
    """

    robot_class: type[RobotBase]
    """
    The name of the robot model being used for optimization.
    """

    terrain_source: type[TerrainBase] | str
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

    initial_state: dict[str, float | list[float]]
    """
    The initial state of the robot.
    """

    final_state: dict[str, float | list[float]]
    """
    The final state of the robot.
    """

    cost_weights: dict[str, float]
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

    state_limits: dict[str, list[float]]
    """
    The limits for the state variables in the optimization problem.

    OM: These seemed to help a lot with convergence, probably because
    they don't let the solver state escape too far from what we expect.
    Seems like we can make these pretty tight, at least with a good 
    initial guess.
    """

    num_steps: int
    """
    The number of steps that the trajectory is discretely broken into.

    This is the number of collocation points in the optimization problem.

    h = T / num_steps
    """

    max_time_s: float
    """
    The maximum time in seconds allowed to reach the final position from the initial position.
    """

    ipopt_print_level: int
    """
    The verbosity level of IPOPT solver output.

    Ranges from 0 to 12, with 0 being the least verbose and 12 being the most verbose.
    """
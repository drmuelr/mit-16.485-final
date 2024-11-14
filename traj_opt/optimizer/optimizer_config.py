from pydantic import BaseModel, FiniteFloat
from traj_opt.models.hopping_softfly import HoppingSoftfly


name_to_model_map: dict[str, BaseModel] = {
    "hopping_softfly": HoppingSoftfly,
}
"""
Maps a robot model name to the corresponding model configuration class.
"""


class OptimizerConfig(BaseModel):
    
    model_name: str = "hopping_softfly"
    """
    The name of the robot model being used for optimization.
    """

    initial_position: list[FiniteFloat] = [0.0, 0.0, 0.0]
    """
    The initial position of the robot.
    """

    final_position: list[FiniteFloat] = [1.0, 0.0, 0.0]
    """
    The final position of the robot.
    """

    num_steps: int = 100
    """
    The number of steps that the trajectory is discretely broken into.

    This is the number of collocation points in the optimization problem.

    h = T / num_steps
    """
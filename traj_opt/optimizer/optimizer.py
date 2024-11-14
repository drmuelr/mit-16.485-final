import casadi as ca

from traj_opt.optimizer.optimizer_config import (
    OptimizerConfig, 
    name_to_model_map
)


class Optimizer:
    def __init__(self, config: OptimizerConfig):
        
        # Free-time Optimal Control Problem
        self.solver = ca.Opti()

        # Store the optimizer configuration
        self.config = config

        # Setup the free time problem parameters
        self.setup_free_time()

        # Initialize the model
        self.model = name_to_model_map[config.model_name](config)

    def setup_free_time(self):
        # Free time variable
        self.T = self.solver.variable()
        
        # Timestep size
        self.h = self.solver.variable()

        # Constrain timestep size using h = T / num_steps
        self.solver.subject_to(self.h == self.T / self.config.num_steps)
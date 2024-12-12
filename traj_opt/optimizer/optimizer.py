from typing import cast

import casadi as ca
import numpy as np
from pathlib import Path

from traj_opt.optimizer.plot import animate_solution
from configs import TrajOptConfig
from traj_opt.models import MeshLoader, VoxbloxSdfLoader


class Optimizer:
    """
    This class reads the configuration parameters from the TrajOptConfig object,
    initializes the robot and terrain models, and solves the optimal control
    problem (OCP) using IPOPT.
    """
    def __init__(self, config: TrajOptConfig):
        """
        Initializes the optimizer with the given configuration.
        """
        # Free-time Optimal Control Problem
        self.solver = ca.Opti()
        self.solution: ca.OptiSol | None = None

        # Store the optimizer configuration
        self.config = config

        # Setup the free time problem parameters
        self.setup_free_time()

        # Setup the terrain model
        self.setup_terrain_model()

        # Initialize the robot model
        print("Initializing robot model: ", config.robot_class.__name__)
        self.robot_model = config.robot_class(self, self.terrain_model)
        print("Finished initializing robot model!")

        # Load the initial guess
        self.load_initial_guess()

    def setup_terrain_model(self):
        """
        Setup the terrain model based on the terrain source from the config:
            1. If the terrain source is a path to a .obj mesh file, use the MeshLoader
            2. If the terrain source is a path to a .npy file, use the VoxbloxSdfLoader
            3. If the terrain source is a custom class, construct that class as the model
        """
        # Check if the terrain source is a custom class
        self.terrain_type = (
            "preset" if isinstance(self.config.terrain_source, type)
            else "mesh" if Path(self.config.terrain_source).suffix == ".obj"
            else "sdf"
        )

        print(
            "Initializing terrain from source: ", 
              self.config.terrain_source if not self.terrain_type == "preset"
              else self.config.terrain_source.__name__ + " (preset)"
            )
        
        self.terrain_model = (
            MeshLoader(self.config.terrain_source) if self.terrain_type == "mesh"
            else VoxbloxSdfLoader(self.config.terrain_source) if self.terrain_type == "sdf"
            else self.config.terrain_source() 
        )
        print("Finished initializing terrain!")


    def setup_free_time(self):
        """
        Sets up the constraints related to free time in the OCP.
        """
        # Free time variable
        self.T = self.solver.variable()

        # Timestep size
        self.h = self.solver.variable()

        # Constrain timestep size using h = T / num_steps
        self.solver.subject_to(self.h == self.T / self.config.num_steps)

        self.solver.subject_to(self.T > 0.01)
        self.solver.subject_to(self.T < self.config.max_time_s)

    def solve(self):
        """
        Solves the OCP and calls animate_solution to visualize the solution.
        """
        # Solve the optimization problem

        plugin_opts = {
            "expand": self.terrain_type == "preset"
        }
        solver_options = {
            "tol": 1e-4,                # Set convergence tolerance
            "max_iter": 100000,           # Increase iteration limit
            "mu_strategy": "adaptive",  # Dynamic barrier parameter adjustment
            "nlp_scaling_method": "none",  # Enable scaling
            "derivative_test": "none",  # Disable derivative checker
            # "hessian_approximation": "limited-memory",  # Use L-BFGS for Hessian approximation
            "print_level": self.config.ipopt_print_level
        }

        self.solver.solver("ipopt", plugin_opts, solver_options)

        try:
            # Solve the problem
            self.solution = self.solver.solve()
            print("Solver successful!")

        except Exception as e:
            print("Solver failed:", e)
            return None

        self.solution = cast(ca.OptiSol, self.solution)

        # Print the trajectory time
        print("Trajectory time:", self.solution.value(self.T))
        print("Timestep size:", self.solution.value(self.h))

        # Save the solution to a file
        self.save_solution()

        # Animate the solution
        animate_solution(self, self.solution)

    def save_solution(self):
        """
        Save the solution to the file specified in the config.
        """
        self.solution = cast(ca.OptiSol, self.solution)

        position = [self.solution.value(self.robot_model.position_world[k]) for k in range(self.config.num_steps + 1)]
        velocity = [self.solution.value(self.robot_model.velocity_world[k]) for k in range(self.config.num_steps + 1)]

        q_body_to_world = [self.solution.value(self.robot_model.q_body_to_world[k]) for k in range(self.config.num_steps + 1)]
        angular_velocity_body = [self.solution.value(self.robot_model.angular_velocity_body[k]) for k in range(self.config.num_steps + 1)]

        control_thrusts = [self.solution.value(self.robot_model.control_thrusts[k]) for k in range(self.config.num_steps + 1)]
        control_moment_body = [self.solution.value(self.robot_model.control_moment_body[k]) for k in range(self.config.num_steps + 1)]

        spring_elongation = [self.solution.value(self.robot_model.spring_elongation[k]) for k in range(self.config.num_steps + 1)]

        T = self.solution.value(self.T)
        h = self.solution.value(self.h)

        np.savez(self.config.save_solution_as,
                 position=position,
                 velocity=velocity,
                 q_body_to_world=q_body_to_world,
                 angular_velocity_body=angular_velocity_body,
                 control_thrusts=control_thrusts,
                 control_moment_body=control_moment_body,
                 spring_elongation=spring_elongation,
                 T=T,
                 h=h)
        
        print("Saved solution to:", self.config.save_solution_as)

    def load_initial_guess(self):
        """
        Load an initial guess from the file specified in the config.
        """
        init_guess_path = self.config.initial_guess_path

        try:
            init_guess = np.load(init_guess_path)
            print("Loaded initial guess from:", init_guess_path)

        except Exception:
            print(f"No initial guess found at {init_guess_path}")
            return

        for k in range(self.config.num_steps + 1):
            self.solver.set_initial(self.robot_model.position_world[k], init_guess["position"][k])
            self.solver.set_initial(self.robot_model.velocity_world[k], init_guess["velocity"][k])
            self.solver.set_initial(self.robot_model.q_body_to_world[k], init_guess["q_body_to_world"][k])
            self.solver.set_initial(self.robot_model.angular_velocity_body[k], init_guess["angular_velocity_body"][k])
            self.solver.set_initial(self.robot_model.control_thrusts[k], init_guess["control_thrusts"][k])
            self.solver.set_initial(self.robot_model.control_moment_body[k], init_guess["control_moment_body"][k])
            self.solver.set_initial(self.T, init_guess["T"])
            self.solver.set_initial(self.h, init_guess["h"])

    def norm_1(self, vec):
        """
        Return the L1 norm of a casadi vector.

        Use this instead of ca.norm_2/ca.norm_1, which seem to cause numerical issues
        (INVALID NUMBER ERROR from IPOPT).
        """
        return vec[0]**2 + vec[1]**2 + vec[2]**2
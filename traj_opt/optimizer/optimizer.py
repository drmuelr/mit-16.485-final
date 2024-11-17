import casadi as ca

from traj_opt.optimizer.plot import animate_solution

from traj_opt.config import (
    TrajOptConfig, 
    robot_name_to_model_map,
    terrain_name_to_model_map
)


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
        self.solution = None

        # Store the optimizer configuration
        self.config = config

        # Setup the free time problem parameters
        self.setup_free_time()

        # Initialize the terrain model
        self.terrain_model = terrain_name_to_model_map[config.terrain_source]()

        # Initialize the robot model
        self.robot_model = robot_name_to_model_map[config.robot_name](self, self.terrain_model)

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

        self.solver.subject_to(self.T > 0.5)
        self.solver.subject_to(self.T < self.config.max_time)

        self.solver.set_initial(self.T, 1.0)  # Initial guess for T
        self.solver.set_initial(self.h, 1.0 / self.config.num_steps)  # Initial guess for h
    
    def solve(self):
        """
        Solves the OCP and calls animate_solution to visualize the solution.
        """
        # Solve the optimization problem
        
        plugin_opts = {
            "expand": True
        }
        solver_options = {
            "tol": 1e-6,
            "acceptable_tol": 1e-4,
            "max_iter": 100000,
            "mu_strategy": "adaptive",
            "nlp_scaling_method": "gradient-based",
            "nlp_scaling_max_gradient": 100,
            "hessian_approximation": "limited-memory",
            "jacobian_approximation": "finite-difference-values",
            "max_soc": 4,
            "derivative_test": "none",  # Enable for debugging
            "print_level": self.config.ipopt_print_level
        }
        self.solver.solver("ipopt", plugin_opts, solver_options)

        try:
            # Solve the problem
            self.solution = self.solver.solve()
            print("Solver successful!")

        except Exception as e:
            self.solver.debug.show_infeasibilities()
            print("Solver failed:", e)
            return None
        
        # Print the trajectory time
        print("Trajectory time:", self.solution.value(self.T))
        print("Timestep size:", self.solution.value(self.h))

        # Animate the solution
        animate_solution(self, self.solution)

    def loose_equals_constraint(self, a, b, tolerance=1e-8):
        """
        Creates a constraint that forces two values to be approximately equal.
        """
        self.solver.subject_to(
            a - b <= tolerance
        )
        self.solver.subject_to(
            b - a <= tolerance
        )

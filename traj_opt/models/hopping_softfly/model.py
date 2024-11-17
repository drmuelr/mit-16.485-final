import casadi as ca
import liecasadi as lca

from traj_opt.models.terrain.base import TerrainBase
from traj_opt.models.hopping_softfly.constants import *


class HoppingSoftfly:

    def __init__(self, optimizer, terrain: TerrainBase):

        # Store the optimizer
        self.optimizer = optimizer
        self.N = optimizer.config.num_steps
        self.dt = self.optimizer.h

        # Store the terrain
        self.terrain = terrain

        # Setup the optimization variables
        self.setup_variables()

        # Setup actuator dynamics
        self.setup_actuators()

        # Setup spring dynamics
        self.setup_spring_dynamics()

        # Setup the contact constraints
        self.setup_contact_constraints()

        # Setup the rigid body dynamics
        self.setup_rigid_body_dynamics()

        # Setup initial and final conditions
        self.setup_initial_and_final_conditions()

        # Setup the cost function
        self.setup_cost()
    
    def setup_variables(self):
        """
        Setup all variables for the optimization problem.
        """
        # State variables (q)
        self.position_world  = [self.optimizer.solver.variable(3) for _ in range(self.N + 1)]
        self.q_body_to_world = [self.optimizer.solver.variable(4) for _ in range(self.N + 1)]
        self.spring_elongation = [self.optimizer.solver.variable() for _ in range(self.N + 1)]

        # Derivatives of the state variables (qdot)
        self.velocity_world        = [self.optimizer.solver.variable(3) for _ in range(self.N + 1)]
        self.angular_velocity_body = [self.optimizer.solver.variable(3) for _ in range(self.N + 1)]

        # Control force + moment (wrench)
        self.control_thrust = [self.optimizer.solver.variable() for _ in range(self.N)]
        self.control_force_world = [self.optimizer.solver.variable(3) for _ in range(self.N)]
        self.control_moment_body = [self.optimizer.solver.variable(3) for _ in range(self.N)]

        # Contact forces and moments
        self.contact_force_z     = [self.optimizer.solver.variable()  for _ in range(self.N)]
        self.contact_force_world = [self.optimizer.solver.variable(3) for _ in range(self.N)]
        self.contact_moment_body = [self.optimizer.solver.variable(3) for _ in range(self.N)]

        # Spring force
        self.spring_force = [self.optimizer.solver.variable() for _ in range(self.N)]
        self.spring_force_world = [self.optimizer.solver.variable(3) for _ in range(self.N)]

        # Surface normal at contact point
        self.cos_theta      = [self.optimizer.solver.variable() for _ in range(self.N+1)]
        self.surface_normal = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]
        self.sdf_value      = [self.optimizer.solver.variable(2) for _ in range(self.N+1)]

        # Contact point locations in world frame
        self.contact_point_location = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]

        # Velocity of contact point location in world frame
        # Used to fake friction (TODO: Remove and replace with full friction
        # model)
        self.contact_point_velocity = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]
        
        # Gravity force = m*g
        self.gravity = ca.vertcat(0, 0, -BODY_MASS_KG*GRAVITY_M_S2)

        # Total force acting on body w.r.t. world frame
        self.total_force_world = [self.optimizer.solver.variable(3) for _ in range(self.N)]

        # Total moment acting on body w.r.t. body frame
        self.total_moment_body = [self.optimizer.solver.variable(3) for _ in range(self.N)]
    
    def setup_spring_dynamics(self):
        """
        Setup the spring dynamics for the optimization problem.

        Solves for the spring force in terms of the elongation.
        """
        for k in range(self.N+1):
            R_body_to_world = lca.SO3(self.q_body_to_world[k]).as_matrix()

            # Spring is compressed when contacting and otherwise has no elongation
            self.optimizer.solver.subject_to(
                self.cos_theta[k] ==
                ca.dot(self.surface_normal[k], R_body_to_world @ ca.vertcat(0, 0, 1)),
            )
            self.optimizer.loose_equals_constraint(
                self.spring_elongation[k],
                ca.fmin(
                    0, 
                    (self.position_world[k][2] / (self.cos_theta[k] + 1e-6)) - ORIGINAL_SPRING_LENGTH_M
                    # Add small value to prevent division by zero
                )
            )

        for k in range(self.N):
            R_body_to_world = lca.SO3(self.q_body_to_world[k]).as_matrix()

            # Compute the spring force
            self.spring_force[k] = - SPRING_CONSTANT_N_M * self.spring_elongation[k]
            self.optimizer.loose_equals_constraint(
                self.spring_force_world[k],
                R_body_to_world @ ca.vertcat(0, 0, self.spring_force[k])
            )
                
    
    def setup_actuators(self):
        """
        Set up the actuator dynamics and constraints for the optimization problem.

        Constraints are defined in constants.py.
        """
        for k in range(self.N):
            # Control force w.r.t. world frame
            R_body_to_world = lca.SO3(self.q_body_to_world[k]).as_matrix()
            self.optimizer.loose_equals_constraint(
                self.control_force_world[k],
                R_body_to_world @ ca.vertcat(0, 0, self.control_thrust[k])
            )

            # Thrust limits
            self.optimizer.solver.subject_to(
                self.control_thrust[k] <= MAX_THRUST_N
            )
            self.optimizer.solver.subject_to(
                self.control_thrust[k] >= 0
            )
        
            # Torque limits
            self.optimizer.solver.subject_to(
                self.control_moment_body[k][0] <= MAX_TORQUE_XY_N_M
            )
            self.optimizer.solver.subject_to(
                self.control_moment_body[k][0] >= -MAX_TORQUE_XY_N_M
            )
            self.optimizer.solver.subject_to(
                self.control_moment_body[k][1] <= MAX_TORQUE_XY_N_M
            )
            self.optimizer.solver.subject_to(
                self.control_moment_body[k][1] >= -MAX_TORQUE_XY_N_M
            )
            self.optimizer.loose_equals_constraint(
                self.control_moment_body[k][2], 0.0
            )

    def setup_contact_constraints(self):
        """
        Sets up the contact constraints for the optimization problem:
            1. Non-penetration constraint
            2. Contact force always positive
            3. Complimentary constraint
        """
        for k in range(self.N+1):
            # Contact point location
            # p_contact = p_drone - R_b^w @ [0, 0, l + d]
            R_body_to_world = lca.SO3(self.q_body_to_world[k]).as_matrix()

            self.optimizer.solver.subject_to(
                self.contact_point_location[k] == 
                self.position_world[k] + 
                R_body_to_world @ ca.vertcat (
                    0, 0, -ORIGINAL_SPRING_LENGTH_M - self.spring_elongation[k]
                )
            )

            # Surface normal at contact point
            self.optimizer.solver.subject_to(
                self.surface_normal[k] ==
                self.terrain.normal_vector(self.contact_point_location[k])
            )

            # Signed distance function value
            self.optimizer.solver.subject_to(
                self.sdf_value[k] == 
                self.terrain.sdf(self.contact_point_location[k])
            )

            # Non-penetration constraint
            self.optimizer.solver.subject_to(
                self.sdf_value[k] >= 0
            )

        
        for k in range(self.N):
            # Integrate contact point velocity
            # TODO: Remove this and replace with full friction model
            self.optimizer.loose_equals_constraint(
                (self.contact_point_location[k+1] - self.contact_point_location[k]),
                self.dt * self.contact_point_velocity[k]
            )

            # Contact force in world frame
            self.optimizer.loose_equals_constraint(
                self.contact_force_world[k],
                self.surface_normal[k] * self.contact_force_z[k]
            )

            # Contact moment in body frame
            R_world_to_body = lca.SO3(self.q_body_to_world[k]).inverse().as_matrix()
            moment_arm = (self.contact_point_location[k] - self.position_world[k])
            self.optimizer.loose_equals_constraint(
                self.contact_moment_body[k],
                R_world_to_body @ ca.cross(moment_arm, self.contact_force_world[k])
            )

            # Complimentary constraint on contact force
            # Contact force is only nonzero during contact
            self.optimizer.loose_equals_constraint(
                self.contact_force_z[k] * self.terrain.sdf(self.contact_point_location[k]),
                0
            )

            # Contact force is positive
            self.optimizer.solver.subject_to(
                self.contact_force_z[k] >= 0
            )

            # Complimentary constraint on zero velocity of contact point
            self.optimizer.loose_equals_constraint(
                self.contact_point_velocity[k] * self.contact_force_z[k],
                0
            )
    
    def setup_rigid_body_dynamics(self):
        """
        Set up the rigid body dynamics for the optimization problem.

        Integrates:
            - velocity -> position
            
            - angular velocity -> orientation

            - total force -> velocity
                - total force consists of control force, gravity, and contact force

            - total moment -> angular velocity
                - total moment consists of control moment and contact moment
        """
        for k in range(self.N):
            # Integrate velocity to get position
            self.optimizer.solver.subject_to(
                (self.position_world[k+1] - self.position_world[k]) ==
                self.dt * self.velocity_world[k]
            )
        
            # Integrate angular velocity to get orientation
            # Leverage Lie Group properties to optimize on SO(3) manifold
            vector_SO3 = lca.SO3Tangent(self.angular_velocity_body[k] * self.dt)
            rotation_SO3 = lca.SO3(self.q_body_to_world[k])
            self.optimizer.solver.subject_to(
                self.q_body_to_world[k + 1] ==
                (vector_SO3 + rotation_SO3).as_quat()
            )

            # Compute total force acting on the body w.r.t. the world frame
            total_force = self.control_force_world[k] + self.gravity + self.contact_force_world[k] + self.spring_force_world[k]

            # Integrate force to get velocity
            self.optimizer.loose_equals_constraint(
                self.dt*total_force,
                BODY_MASS_KG * (self.velocity_world[k+1] - self.velocity_world[k])
            )

            # Compute total moment acting on the body w.r.t. the body frame
            # NOTE: Do not transform to world frame b/c angular velocity is in body frame
            total_moment = self.control_moment_body[k] + self.contact_moment_body[k]

            # Integrate moment to get angular velocity
            self.optimizer.loose_equals_constraint(
                self.dt * total_moment,
                I @ (self.angular_velocity_body[k+1] - self.angular_velocity_body[k])
            )

    def setup_initial_and_final_conditions(self):
        """
        Sets up the initial and final state constraints for the optimization problem with epsilon regularization.
        """
        epsilon = 1e-3  # Small tolerance value

        # Initial state constraints
        initial_state = self.optimizer.config.initial_state

        self.optimizer.loose_equals_constraint(
            self.position_world[0],
            ca.vertcat(*initial_state["position"]),
            epsilon
        )
        self.optimizer.loose_equals_constraint(
            self.q_body_to_world[0],
            ca.vertcat(*initial_state["q_body_to_world"]),
            epsilon
        )
        self.optimizer.loose_equals_constraint(
            self.velocity_world[0],
            ca.vertcat(*initial_state["velocity"]),
            epsilon
        )
        self.optimizer.loose_equals_constraint(
            self.angular_velocity_body[0],
            ca.vertcat(*initial_state["angular_velocity_body"]),
            epsilon
        )
        self.optimizer.loose_equals_constraint(
            self.control_thrust[0],
            initial_state["control_force"],
            epsilon
        )
        self.optimizer.loose_equals_constraint(
            self.control_moment_body[0],
            ca.vertcat(*initial_state["control_moment"]),
            epsilon
        )
        self.optimizer.loose_equals_constraint(
            self.spring_elongation[0],
            0.0,
            epsilon
        )
        
        # Final state constraints
        final_state = self.optimizer.config.final_state
        
        self.optimizer.loose_equals_constraint(
            self.position_world[self.N],
            ca.vertcat(*final_state["position"]),
            epsilon
        )
        self.optimizer.loose_equals_constraint(
            self.q_body_to_world[self.N],
            ca.vertcat(*final_state["q_body_to_world"]),
            epsilon
        )
        self.optimizer.loose_equals_constraint(
            self.velocity_world[self.N],
            ca.vertcat(*final_state["velocity"]),
            epsilon
        )
        self.optimizer.loose_equals_constraint(
            self.angular_velocity_body[self.N],
            ca.vertcat(*final_state["angular_velocity_body"]),
            epsilon
        )
        self.optimizer.loose_equals_constraint(
            self.control_thrust[self.N-1],
            final_state["control_force"],
            epsilon
        )
        self.optimizer.loose_equals_constraint(
            self.control_moment_body[self.N-1],
            ca.vertcat(*final_state["control_moment"]),
            epsilon
        )

    def setup_cost(self):
        """
        Define the cost function for the optimization problem.
        """
        cost_weights = self.optimizer.config.cost_weights

        cost = 0

        for k in range(self.optimizer.config.num_steps):
            # Define the cost function using weights given in constants.py
            cost += (
                cost_weights["T"]*self.optimizer.T
                + cost_weights["position_world"]*ca.sumsqr(
                    self.position_world[k] - ca.vertcat(*self.optimizer.config.final_state["position"])
                )
                + cost_weights["velocity_world"]*ca.sumsqr(self.velocity_world[k])
                + cost_weights["q_body_to_world"]*ca.sumsqr(
                    2*ca.acos(
                        ca.dot(self.q_body_to_world[k], ca.vertcat(0, 0, 0, 1))
                    ) # Geodesic Distance for attitude error
                )
                + cost_weights["contact_force"]*self.contact_force_z[k]
                + cost_weights["angular_velocity_body"]*ca.sumsqr(self.angular_velocity_body[k])
                + cost_weights["control_force"]*ca.sumsqr(
                    self.control_thrust[k] + self.gravity[2] + self.contact_force_z[k] # Penalize deviation from equilibrium
                )
                + cost_weights["control_moment"]*ca.sumsqr(self.control_moment_body[k])
            )

        self.optimizer.solver.minimize(cost)
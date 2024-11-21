# Bandaid for circular import. Real solution is to not have circular depdenence.
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from traj_opt.optimizer.optimizer import Optimizer

import casadi as ca
import liecasadi as lca

from traj_opt.models.robot_base.model import RobotBase
from traj_opt.models.terrain.base import TerrainBase
from traj_opt.models.hopping_softfly.constants import *


class HoppingSoftfly(RobotBase):

    def __init__(self, optimizer: Optimizer, terrain: TerrainBase):

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
        self.setup_contact_dynamics()

        # Setup the rigid body dynamics
        self.setup_rigid_body_dynamics()

        # Setup initial and final conditions
        self.setup_initial_and_final_conditions()

        # Setup state limits
        self.setup_state_limits()

        # Setup the cost function
        self.setup_cost()
    
    def setup_variables(self):
        """
        Setup all variables for the optimization problem.
        """
        # State variables (q)
        self.position_world    = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]
        self.q_body_to_world   = [self.optimizer.solver.variable(4) for _ in range(self.N+1)]
        self.spring_elongation = [self.optimizer.solver.variable() for _ in range(self.N+1)]

        # Derivatives of the state variables (qdot)
        self.velocity_world        = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]
        self.angular_velocity_body = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]

        # Control forces (1 per actuator) + moments 
        self.control_thrusts     = [self.optimizer.solver.variable(4) for _ in range(self.N+1)]
        self.control_force_world = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]
        self.control_moment_body = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]

        # Spring force
        self.spring_force       = [self.optimizer.solver.variable() for _ in range(self.N+1)]

        # Contact forces + moment
        self.friction_impulse    = [self.optimizer.solver.variable(3)  for _ in range(self.N+1)]
        self.contact_force_world = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]
        self.contact_moment_body = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]

        # Signed-distance-field value at contact point
        self.sdf_value      = [self.optimizer.solver.variable() for _ in range(self.N+1)]

        # Surface normal at contact point
        self.surface_normal = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]

        # Contact point location + velocity in world frame
        self.contact_point_location = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]
        self.contact_point_velocity = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]

        # Gravity force = m*g (constant)
        self.gravity = ca.vertcat(0, 0, -BODY_MASS_KG*GRAVITY_M_S2)

        # Total force acting on body w.r.t. world frame
        self.total_force_world = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]

        # Total moment acting on body w.r.t. body frame
        self.total_moment_body = [self.optimizer.solver.variable(3) for _ in range(self.N+1)]
    
    def setup_actuators(self):
        """
        Set up the actuator dynamics and constraints for the optimization problem.

        Constraints are defined in constants.py.
        """
        for k in range(self.N+1):
            # Control force w.r.t. world frame
            R_body_to_world = lca.SO3(self.q_body_to_world[k]).as_matrix()
            self.optimizer.solver.subject_to(
                self.control_force_world[k] == 
                R_body_to_world @ ca.vertcat(0, 0, ca.sum1(self.control_thrusts[k]))
            )

            # Thrust limits
            for i in range(4):
                self.optimizer.solver.subject_to(
                    self.control_thrusts[k][i] <= MAX_THRUST_SINGLE_ACTUATOR_N
                )
                self.optimizer.solver.subject_to(
                    self.control_thrusts[k][i] >= 0
                )

            # Control moment w.r.t. body frame in terms of individual thrusts
            self.optimizer.solver.subject_to(
                self.control_moment_body[k][0] ==
                ARM_LENGTH_M * (self.control_thrusts[k][1] - self.control_thrusts[k][3])
            )
            self.optimizer.solver.subject_to(
                self.control_moment_body[k][1] ==
                ARM_LENGTH_M * (self.control_thrusts[k][2] - self.control_thrusts[k][0])
            )
            self.optimizer.solver.subject_to(
                self.control_moment_body[k][2] == 0
            )

    def setup_spring_dynamics(self):
        """
        Setup the spring dynamics for the optimization problem.

        Constrains the spring elongation and defines the spring force 
        in terms of the elongation.
        """
        for k in range(self.N+1):
            # The spring cannot be compressed beyond its original length
            self.optimizer.solver.subject_to(
                self.spring_elongation[k] >= -ORIGINAL_SPRING_LENGTH_M
            )
        
            # Compute the spring force
            self.optimizer.solver.subject_to(
                self.spring_force[k] == - SPRING_CONSTANT_N_M * self.spring_elongation[k]
            )

            # Spring force can only be nonzero if in contact with the ground
            self.optimizer.solver.subject_to(
                self.spring_force[k]*self.sdf_value[k] == 0
            )
        
    def setup_contact_dynamics(self):
        """
        Sets up the contact dynamics for the optimization problem

        These include:
            - Contact point location in terms of drone position and spring elongation
            - SDF value at contact point
            - Surface normal at contact point
            - Non-penetration constraint (SDF > 0)
            - Contact Force:
                - Always positive
                - Equal to spring force plus friction impulse
                - Friction immpulse has no normal component and lies within the friction cone
                - Friction impulse is zero if not in contact
            - Contact moment:
                - Equal to contact position x contact force (in body frame)
        """
        for k in range(self.N+1):
            # Contact point location
            # p_contact = p_drone - R_b^w @ [0, 0, l + d]
            R_body_to_world = lca.SO3(self.q_body_to_world[k]).as_matrix()

            contact_point_body = ca.vertcat(
                0, 0, -ORIGINAL_SPRING_LENGTH_M - self.spring_elongation[k]
            )
            self.optimizer.solver.subject_to(
                self.contact_point_location[k] == (
                    self.position_world[k] + 
                    R_body_to_world @ contact_point_body
                )
            )

            # Signed distance function value
            self.optimizer.solver.subject_to(
                self.sdf_value[k] == self.terrain.sdf(self.contact_point_location[k])
            )

            # Surface normal at contact point
            self.optimizer.solver.subject_to(
                self.surface_normal[k] == self.terrain.normal_vector(self.contact_point_location[k])
            )

            # Non-penetration constraint
            self.optimizer.solver.subject_to(
                self.sdf_value[k] >= 0
            )

            # Contact force normal to surface is always positive
            self.optimizer.solver.subject_to(
                ca.dot(
                    self.contact_force_world[k], self.surface_normal[k]
                ) >= 0
            )

            # Friction impulse lies within the friction cone
            friction_impulse_normal = ca.dot(self.friction_impulse[k], self.surface_normal[k])
            friction_impulse_tangent = (
                self.friction_impulse[k] - (friction_impulse_normal*self.surface_normal[k])
            )

            self.optimizer.solver.subject_to(
                friction_impulse_normal == 0
            )
            self.optimizer.solver.subject_to(
                self.optimizer.norm_1(friction_impulse_tangent)
                <= (self.contact_force_world[k][2]*FRICTION_COEFFICIENT)**2
            )

            # Complimentary constraint on friction impulse
            self.optimizer.solver.subject_to(
                self.friction_impulse[k]*self.sdf_value[k] == 0
            )

            # Complimentary constraint for no slip condition
            self.optimizer.solver.subject_to(
                self.optimizer.norm_1(self.contact_point_velocity[k])
                * ca.dot(
                    self.contact_force_world[k], self.surface_normal[k]
                ) == 0
            )

            # Contact force is spring force plus friction impulse
            R_body_to_world = lca.SO3(self.q_body_to_world[k]).as_matrix()
            spring_force_world = R_body_to_world @ ca.vertcat(0, 0, self.spring_force[k])

            self.optimizer.solver.subject_to(
                self.contact_force_world[k] == spring_force_world + self.friction_impulse[k]
            )

            # Contact moment is r x contact force
            R_world_to_body = lca.SO3(self.q_body_to_world[k]).inverse().as_matrix()
            contact_force_body = R_world_to_body @ self.contact_force_world[k]
            self.optimizer.solver.subject_to(
                self.contact_moment_body[k] == ca.cross(
                    contact_point_body, contact_force_body
                )
            )

    def setup_rigid_body_dynamics(self):
        """
        Set up the rigid body dynamics for the optimization problem.

        Integrates:
            - velocity -> position
            
            - angular velocity -> orientation

            - total force -> velocity
                - total force consists of control force, gravity, and total contact force

            - total moment -> angular velocity
                - total moment consists of control moment and total contact moment
        """
        for k in range(self.N):
            # Integrate contact point velocity
            self.optimizer.solver.subject_to(
               self.dt*self.contact_point_velocity[k+1] == 
               (self.contact_point_location[k+1] - self.contact_point_location[k])
            ) 

            # Integrate velocity to get position
            self.optimizer.solver.subject_to(
                (self.position_world[k+1] - self.position_world[k])
                == self.dt * self.velocity_world[k+1]
            )
        
            # Integrate angular velocity to get orientation
            # Leverage Lie Group properties to optimize on SO(3) manifold
            vector_SO3 = lca.SO3Tangent(self.angular_velocity_body[k+1] * self.dt)
            rotation_SO3 = lca.SO3(self.q_body_to_world[k])
            self.optimizer.solver.subject_to(
                self.q_body_to_world[k + 1] ==
                (vector_SO3 + rotation_SO3).as_quat()
            )

            # Compute total force acting on the body w.r.t. the world frame
            total_force = (
                self.control_force_world[k+1] + self.gravity + self.contact_force_world[k+1]
            )

            # Integrate force to get velocity
            self.optimizer.solver.subject_to(
                self.dt * total_force ==
                BODY_MASS_KG * (self.velocity_world[k+1] - self.velocity_world[k])
            )

            # Compute total moment acting on the body w.r.t. the body frame
            # NOTE: Do not transform to world frame b/c angular velocity is in body frame
            total_moment = self.control_moment_body[k+1] + self.contact_moment_body[k+1]

            # Integrate moment to get angular velocity
            self.optimizer.solver.subject_to(
                self.dt * total_moment ==
                I @ (self.angular_velocity_body[k+1] - self.angular_velocity_body[k])
            )

    def setup_initial_and_final_conditions(self):
        """
        Sets up the initial and final state constraints for the optimization problem.
        """
        # Initial state constraints
        initial_state = self.optimizer.config.initial_state

        self.optimizer.solver.subject_to(
            self.position_world[0] == ca.vertcat(*initial_state["position"])
        )
        self.optimizer.solver.subject_to(
            self.q_body_to_world[0] == ca.vertcat(*initial_state["q_body_to_world"])
        )
        self.optimizer.solver.subject_to(
            self.velocity_world[0] == ca.vertcat(*initial_state["velocity"])
        )
        self.optimizer.solver.subject_to(
            self.angular_velocity_body[0] == ca.vertcat(*initial_state["angular_velocity_body"])
        )

        # Final state constraints
        final_state = self.optimizer.config.final_state

        self.optimizer.solver.subject_to(
            self.position_world[-1] == ca.vertcat(*final_state["position"])
        )
        self.optimizer.solver.subject_to(
            self.q_body_to_world[-1] == ca.vertcat(*final_state["q_body_to_world"])
        )
        self.optimizer.solver.subject_to(
            self.velocity_world[-1] == ca.vertcat(*final_state["velocity"])
        )
        self.optimizer.solver.subject_to(
            self.angular_velocity_body[-1] == ca.vertcat(*initial_state["angular_velocity_body"])
        )

    def setup_state_limits(self):
        """
        Set up the state limits for the optimization problem.
        """
        state_limits = self.optimizer.config.state_limits

        for k in range(self.N+1):
            # setup the limits
            self.optimizer.solver.subject_to(
                self.position_world[k][0] >= state_limits["position_X"][0]
            )
            self.optimizer.solver.subject_to(
                self.position_world[k][0] <= state_limits["position_X"][1]
            )
            self.optimizer.solver.subject_to(
                self.position_world[k][1] >= state_limits["position_Y"][0]
            )
            self.optimizer.solver.subject_to(
                self.position_world[k][1] <= state_limits["position_Y"][1]
            )
            self.optimizer.solver.subject_to(
                self.position_world[k][2] >= state_limits["position_Z"][0]
            )
            self.optimizer.solver.subject_to(
                self.position_world[k][2] <= state_limits["position_Z"][1]
            )

            for i in range(3):
                self.optimizer.solver.subject_to(
                    self.velocity_world[k][i] >= state_limits["velocity"][0]
                )
                self.optimizer.solver.subject_to(
                    self.velocity_world[k][i] <= state_limits["velocity"][1]
                )
                self.optimizer.solver.subject_to(
                    self.angular_velocity_body[k][i] >= state_limits["angular_velocity_body"][0]
                )
                self.optimizer.solver.subject_to(
                    self.angular_velocity_body[k][i] <= state_limits["angular_velocity_body"][1]
                )

    def setup_cost(self):
        """
        Define the cost function for the optimization problem.

        Variables are attempted to be normalized to be on the same scale.
        """
        cost_weights = self.optimizer.config.cost_weights        

        cost = self.optimizer.T/self.optimizer.config.max_time * cost_weights["T"]

        for k in range(self.N+1):
            cost += (
                cost_weights["control_force"]/(self.N+1) * ca.sumsqr(self.control_force_world[k]/(4*MAX_THRUST_SINGLE_ACTUATOR_N))
                + cost_weights["control_moment"]/(self.N+1) * ca.sumsqr(self.control_moment_body[k]/(4*MAX_THRUST_SINGLE_ACTUATOR_N*ARM_LENGTH_M))
                + cost_weights["contact_force"]/(self.N+1) * ca.sumsqr(self.contact_force_world[k]/(4*MAX_THRUST_SINGLE_ACTUATOR_N))
            )

        self.optimizer.solver.minimize(cost)
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

        # Setup the dynamics
        self.setup_dynamics()

        # Setup the contact constraints
        self.setup_contact_constraints()

        # Define the cost function
        self.setup_cost()

        # Define the initial and final conditions
        self.setup_initial_and_final_conditions()

    def setup_cost(self):
        cost = 0
        for k in range(self.optimizer.config.num_steps):
            # Minimize the sum of the squares of the control input
            cost += (
                ca.sumsqr(self.optimizer.T)
                + 0.01*ca.sumsqr(self.control_forces[k])
            )

        self.optimizer.solver.minimize(cost)
    
    def setup_initial_and_final_conditions(self):
        # Initial state constraints
        self.optimizer.solver.subject_to(
            self.body_position[0] == ca.vertcat(*self.optimizer.config.initial_position)
        )
        self.optimizer.solver.subject_to(
                self.body_quat[0] == lca.SO3.Identity().as_quat()
        )
        self.optimizer.solver.subject_to(
            self.body_velocity[0] == ca.vertcat(*[0.0, 0.0, 0.0])
        )
        self.optimizer.solver.subject_to(
            self.body_angular_velocity[0] == ca.vertcat(*[0.0, 0.0, 0.0])
        )
        self.optimizer.solver.subject_to(
            self.spring_elongation[0] == 0
        )
        self.optimizer.solver.subject_to(
            self.spring_velocity[0] == 0
        )

        # Final state constraints
        self.optimizer.solver.subject_to(
            self.body_position[self.N] == ca.vertcat(*self.optimizer.config.final_position)
        )
        self.optimizer.solver.subject_to(
            self.body_quat[self.N] == lca.SO3.Identity().as_quat()
        )
        # self.optimizer.solver.subject_to(
        #     self.body_velocity[self.N] == ca.vertcat(*[0.0, 0.0, 0.0])
        # )
        self.optimizer.solver.subject_to(
            self.body_angular_velocity[self.N] == ca.vertcat(*[0.0, 0.0, 0.0])
        )

    def setup_contact_constraints(self):
        for k in range(self.N):
            # Non-penetration constraint
            self.optimizer.solver.subject_to(
                self.terrain.sdf(self.contact_point_location[k]) >= 0
            )

            # Contact force is positive
            self.optimizer.solver.subject_to(
                self.contact_force[k] >= 0
            )

            # Complimentary constraint
            self.optimizer.solver.subject_to(
                self.terrain.normal_vector(self.contact_point_location[k]) @ self.contact_force[k] == 0
            )


    def setup_dynamics(self):
        # Define the state variables (q)
        self.body_position     = [self.optimizer.solver.variable(3, 1) for _ in range(self.N + 1)]
        self.body_quat         = [self.optimizer.solver.variable(4, 1) for _ in range(self.N + 1)]
        self.spring_elongation = [self.optimizer.solver.variable(1)    for _ in range(self.N + 1)]

        for k in range(self.N):
            # Spring cannot deform more than its length
            self.optimizer.solver.subject_to(
                self.spring_elongation[k] >= -SPRING_LENGTH_M
            )
            self.optimizer.solver.subject_to(
                self.spring_elongation[k] <= SPRING_LENGTH_M
            )

        # Define derivatives of the state variables (qdot)
        self.body_velocity         = [self.optimizer.solver.variable(3, 1) for _ in range(self.N + 1)]
        self.body_angular_velocity = [self.optimizer.solver.variable(3, 1) for _ in range(self.N + 1)]
        self.spring_velocity       = [self.optimizer.solver.variable(1)    for _ in range(self.N + 1)]

        # qk - qk+1 + h*qdotk+1 = 0 (eq 7a from Posa, Cantu, Tedrake 2013)
        for k in range(self.N):
            # Relate position and velocity
            self.optimizer.solver.subject_to(
                (self.body_position[k+1] - self.body_position[k]) 
                + self.dt * self.body_velocity[k] 
                == 0
            )

            # Relate 3D rotation and angular velocity
            vector_SO3 = lca.SO3Tangent(self.body_angular_velocity[k] * self.dt)
            rotation_SO3 = lca.SO3(self.body_quat[k])
            self.optimizer.solver.subject_to(self.body_quat[k + 1] == (vector_SO3 + rotation_SO3).as_quat())

            # Relate spring elongation and spring velocity
            self.optimizer.solver.subject_to(
                (self.spring_elongation[k+1] - self.spring_elongation[k]) 
                + self.dt * self.spring_velocity[k] 
                == 0
            )

        # Contact point is (SPRING_LENGTH + spring_elongation) * -z_body
        self.contact_point_location = [
            self.body_position[k] +
            lca.SO3(self.body_quat[k]).inverse().as_matrix() @
            ca.vertcat(0, 0, -(SPRING_LENGTH_M + self.spring_elongation[k]))
            for k in range(self.N+1)
        ]

        # 3D contact force applied to at end of spring
        self.contact_force = [self.optimizer.solver.variable(1) for _ in range(self.N+1)]

        # Forces from 4 flapping wings and constraints on them
        self.control_forces = [self.optimizer.solver.variable(4, 1) for _ in range(self.N)]
        self.optimizer.solver.subject_to(
            ca.vertcat(*self.control_forces) >= 0
        )
        self.optimizer.solver.subject_to(
            ca.vertcat(*self.control_forces) <= 5
        )
        
        # Gravity force = m*g
        self.gravity = BODY_MASS_KG * ca.vertcat(0, 0, -GRAVITY_M_S2)
        
        # Model acceleration dynamics
        for k in range(self.N):
            
            # World to body rotation matrix
            R = lca.SO3(self.body_quat[k]).inverse().as_matrix()

            # Control forces + moments
            control_force_1 = ca.vertcat(0, 0, self.control_forces[k][0])
            control_force_2 = ca.vertcat(0, 0, self.control_forces[k][1])
            control_force_3 = ca.vertcat(0, 0, self.control_forces[k][2])
            control_force_4 = ca.vertcat(0, 0, self.control_forces[k][3])

            control_moment_1 = ca.cross(ca.vertcat(ARM_LENGTH_M, 0, 0),  control_force_1)
            control_moment_2 = ca.cross(ca.vertcat(0, ARM_LENGTH_M, 0),  control_force_2)
            control_moment_3 = ca.cross(ca.vertcat(-ARM_LENGTH_M, 0, 0), control_force_3)
            control_moment_4 = ca.cross(ca.vertcat(0, -ARM_LENGTH_M, 0), control_force_4)

            total_control_force = R @ (
                control_force_1 + control_force_2 + control_force_3 + control_force_4
            )
            total_control_moment =  R @ (
                control_moment_1 + control_moment_2 + control_moment_3 + control_moment_4 
            )

            # Contact forces and moments w.r.t. world frame
            total_contact_force = (
                self.terrain.normal_vector(self.contact_point_location[k]) * self.contact_force[k]
            )

            total_contact_moment = (ca.cross(
                self.contact_point_location[k] - self.body_position[k],
                total_contact_force
            ))

            # Spring force
            spring_force = -SPRING_CONSTANT_N_M * self.spring_elongation[k]

            self.optimizer.solver.subject_to(
                (self.spring_velocity[k+1] - self.spring_velocity[k]) == self.dt * spring_force
            )

            # Total force acting on the body w.r.t. the world frame
            total_force = total_control_force + self.gravity + total_contact_force - R@ca.vertcat(0, 0, spring_force)
            
            self.optimizer.solver.subject_to(
                self.dt*total_force == BODY_MASS_KG * (self.body_velocity[k+1] - self.body_velocity[k])
            )

            #Total moment acting on the body w.r.t. the world frame
            total_moment = total_control_moment + total_contact_moment
            
            self.optimizer.solver.subject_to(
                total_moment == I @ (self.body_angular_velocity[k+1] - self.body_angular_velocity[k]) / self.dt
            )
            
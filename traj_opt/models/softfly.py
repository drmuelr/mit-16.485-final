import casadi as ca
import liecasadi as lca

from traj_opt.models.terrain import TerrainBase
from traj_opt.models.hopping_softfly_constants import *


class Softfly:

    def __init__(self, optimizer: "Optimizer", terrain: TerrainBase):

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

        # Final state constraints
        self.optimizer.solver.subject_to(
            self.body_position[self.N] == ca.vertcat(*self.optimizer.config.final_position)
        )
        self.optimizer.solver.subject_to(
            self.body_quat[self.N] == lca.SO3.Identity().as_quat()
        )
        self.optimizer.solver.subject_to(
            self.body_velocity[self.N] == ca.vertcat(*[0.0, 0.0, 0.0])
        )
        self.optimizer.solver.subject_to(
            self.body_angular_velocity[self.N] == ca.vertcat(*[0.0, 0.0, 0.0])
        )

    def setup_dynamics(self):
        # Number and size of timesteps in optimization

        # Define the state variables (q)
        self.body_position = [self.optimizer.solver.variable(3, 1) for _ in range(self.N + 1)]
        self.body_quat     = [self.optimizer.solver.variable(4, 1) for _ in range(self.N + 1)]

        # Define derivatives of the state variables (qdot)
        self.body_velocity = [self.optimizer.solver.variable(3, 1) for _ in range(self.N + 1)]
        self.body_angular_velocity = [self.optimizer.solver.variable(3, 1) for _ in range(self.N + 1)]
        
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


        # Define forces
        # 4 flapping wings
        self.control_forces = [self.optimizer.solver.variable(4, 1) for _ in range(self.N)]
        self.optimizer.solver.subject_to(
            ca.vertcat(*self.control_forces) >= 0
        )
        self.optimizer.solver.subject_to(
            ca.vertcat(*self.control_forces) <= 10
        )
        
        # Gravity force = m*g
        self.gravity = BODY_MASS_KG * ca.vertcat(0, 0, -GRAVITY_M_S2)

        # Contact force at bottom of drone
        self.contact_force = [self.optimizer.solver.variable(1) for _ in range(self.N+1)]
        self.contact_point_location = [
            self.body_position[k] +
            lca.SO3(self.body_quat[k]).inverse().as_matrix() @
            ca.vertcat(0, 0, 0) # Change this to test out contact points in diff locations
            for k in range(self.N+1)
        ]
        
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

            # Total force acting on the body w.r.t. the world frame
            total_force = total_control_force + self.gravity + total_contact_force
            
            self.optimizer.solver.subject_to(
                self.dt*total_force == BODY_MASS_KG * (self.body_velocity[k+1] - self.body_velocity[k])
            )

            #Total moment acting on the body w.r.t. the world frame
            total_moment = total_control_moment
            
            self.optimizer.solver.subject_to(
                total_moment == I @ (self.body_angular_velocity[k+1] - self.body_angular_velocity[k]) / self.dt
            )
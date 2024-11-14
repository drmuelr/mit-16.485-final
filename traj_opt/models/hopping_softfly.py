import casadi as ca
import liecasadi as lca
from traj_opt.optimizer.optimizer import Optimizer

NUM_CONTACT_POINTS = 1
SPRING_CONSTANT = 1.0
SPRING_LENGTH_M = 1.0
GRAVITY_M_S2 = 9.81
BODY_MASS_KG = 1.0

class HoppingSoftfly:

    def __init__(self, optimizer: Optimizer):

        # Store the optimizer
        self.optimizer = optimizer

        # Setup the dynamics
        self.setup_dynamics()


    def setup_dynamics(self):
        # Number and size of timesteps in optimization
        N = self.optimizer.config.num_steps
        dt = self.optimizer.h

        # Define the state variables (q)
        self.body_position = self.optimizer.solver.variable(3, N+1)  
        self.body_rotation_quat = self.optimizer.solver.variable(4, N+1)
        self.spring_elongation = self.optimizer.solver.variable(1, N+1)

        # Define derivatives of the state variables (qdot)
        self.body_velocity = self.optimizer.solver.variable(3, N+1)
        self.body_angular_velocity = self.optimizer.solver.variable(3, N+1)
        self.spring_velocity = self.optimizer.solver.variable(1, N+1)
        
        # qk - qk+1 + h*qdotk+1 = 0 (eq 7a from Posa, Cantu, Tedrake 2013)
        for k in range(N):
            # Relate position and velocity
            self.optimizer.solver.subject_to(
                (self.body_position[:, k+1] - self.body_position[:, k]) 
                + dt * self.body_velocity[:, k] 
                == 0
            )

            # Relate 3D rotation and angular velocity using liecasadi
            vector_SO3 = lca.SO3Tangent(self.body_angular_velocity[:, k] * dt)
            rotation_SO3 = lca.SO3(self.body_rotation_quat[:, k])
            self.optimizer.solver.subject_to(
                self.body_rotation_quat[k + 1] == (vector_SO3 + rotation_SO3).as_quat()
            )

            # Relate spring elongation and spring velocity
            self.optimizer.solver.subject_to(
                (self.spring_elongation[:, k+1] - self.spring_elongation[:, k]) 
                + dt * self.spring_velocity[:, k] 
                == 0
            )

        # Contact point is (SPRING_LENGTH + spring_elongation) * -z_body
        self.contact_point_location = (
            self.body_position + 
            (
                (SPRING_LENGTH_M + self.spring_elongation) 
                * (lca.SO3(self.body_rotation_quat).as_matrix() @ ca.vertcat(0, 0, -1))
            )
        )

        # Define forces
        # 4 flapping wings
        self.thrusts = self.optimizer.solver.variable(4, N+1)
        
        # Gravity force = m*g
        self.gravity = BODY_MASS_KG * ca.vertcat(0, 0, -GRAVITY_M_S2)
        
        # 3D contact force for each contact point
        # TODO: Constrain these here?
        self.contact_forces = [self.optimizer.solver.variable(3, N+1) for i in range(NUM_CONTACT_POINTS)]

        # Spring force = -k*x
        self.spring_forces = -SPRING_CONSTANT * self.spring_elongation

        # TODO: Define moments 

        for k in range(N):
            # TODO: Sum of forces = m*a
            # TODO: Sum of moments = I*alpha
            raise NotImplementedError
        
        # TODO: Complimentary constraints
        
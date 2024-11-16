import casadi as ca

NUM_CONTACT_POINTS = 1
"""
Number of contact points on the robot (just the bottom of the spring).
"""

SPRING_CONSTANT_N_M = 1.0
"""
Spring constant for the spring in the robot [N/m].
"""

SPRING_LENGTH_M = 1.0
"""
Length of the spring when undeformed [m].
"""

GRAVITY_M_S2 = 9.81
"""
Acceleration due to Earth's gravity [m/s^2].
"""

BODY_MASS_KG = 1.0
"""
Total mass of body [kg].
"""

ARM_LENGTH_M = 1.0
"""
Length of arm connecting the body to each wing [m].
"""


IX = 0.1  # Example inertia around x-axis, adjust based on your model
IY = 0.2  # Example inertia around y-axis, adjust based on your model
IZ = 0.3  # Example inertia around z-axis, adjust based on your model

I = ca.diag(ca.vertcat(IX, IY, IZ))
"""
Inertial tensor for the body [kg*m^2].
"""
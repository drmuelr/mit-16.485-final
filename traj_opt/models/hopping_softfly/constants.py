import casadi as ca

ORIGINAL_SPRING_LENGTH_M = 0.5
"""
The length of the spring when it is at rest [m].
"""

SPRING_CONSTANT_N_M = 100.0
"""
The spring constant of the spring [N/m].
"""

GRAVITY_M_S2 = 9.81
"""
Acceleration due to Earth's gravity [m/s^2].
"""

BODY_MASS_KG = 1.0
"""
Total mass of body [kg].
"""

ARM_LENGTH_M = 0.5
"""
Length of arm connecting the body to each wing [m].
"""

MAX_THRUST_N = 10.0
"""
Maximum vertical thrust that can be applied by the wings [N].
"""

MAX_TORQUE_XY_N_M = 0.0 #ARM_LENGTH_M * MAX_THRUST_N
"""
Maximum X/Y torque that can be applied by the wings [N*m].
"""

I = ca.diag(ca.vertcat(0.1, 0.1, 0.1))
"""
Inertial tensor for the body [kg*m^2].
"""

import casadi as ca

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

MAX_THRUST_SINGLE_ACTUATOR_N = 20.0
"""
Maximum vertical thrust that can be applied by the wings [N].
"""

I = ca.diag(ca.vertcat(0.1, 0.1, 0.1))
"""
Inertial tensor for the body [kg*m^2].
"""
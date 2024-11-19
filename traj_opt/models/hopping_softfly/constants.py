import casadi as ca

ORIGINAL_SPRING_LENGTH_M = 0.45
"""
The length of the spring when it is at rest [m].
"""

SPRING_CONSTANT_N_M = 200.0
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

MAX_THRUST_SINGLE_ACTUATOR_N = 9.81/3.8
"""
Maximum vertical thrust that can be applied by the wings [N].

9.81/3.8 gives approx 1.05 thrust/weight ratio.
"""

I = ca.diag(ca.vertcat(0.1, 0.1, 0.1))
"""
Inertial tensor for the body [kg*m^2].
"""

SIGMOID_STEEPNESS = 1000.0
"""
The steepness of the sigmoid function used for the spring deformation constraints.

1000.0 seems to give good results without giving the solver a hard time.
"""
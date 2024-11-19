import casadi as ca

# Define sigmoid function for use in spring deformation constraints
x = ca.MX.sym("x")
k = ca.MX.sym("k") # Sigmoid steepness
sigmoid_value = 1 / (1 + ca.exp(-k*x))
sigmoid = ca.Function("sigmoid", [x, k], [sigmoid_value])
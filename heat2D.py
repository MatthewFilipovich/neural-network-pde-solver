"""Module solves heat equation in 2D using nengo_pde."""
from nengo_pde import Solver2D


def feedback_connection(u):
    return - (K/dx**2) * 2*u


def lateral_connection(u):
    return K/dx**2 * u


# Nengo simulation
t_steps = 50  # Number of time steps
x_steps = 20  # Number of x steps
neurons = 600  # Number of neurons
radius = 80  # Radius of neurons
boundaries = 50  # Constant boundary conditions
solver = Solver2D(feedback_connection, lateral_connection)

# Grid properties
K = 4.2
x_len = 20  # mm
dx = x_len/x_steps
dt = dx**2/(2*K**2)  # dt chosen for stability

# Run nengo_pde simulation
solver.run_nengo_order1(dt, t_steps, x_steps, boundaries, neurons, radius)
solver.plot_grid(0)  # show plot at t=0
solver.animate()

"""Module solves wave equation in 1D using nengo_pde."""
from nengo_pde import Solver1D
import matplotlib.pyplot as plt
import numpy as np


def feedback_connection(u):
    return np.array((u[1], -2*c**2/dx**2 * u[0]))


def lateral_connection(u):
    return np.array((0, c**2/dx**2 * u[0]))


def boundaries(t):
    return[10 * np.exp(-(t / dt - 10) ** 2 / 20),
           10 * np.exp(-(t / dt - 10) ** 2 / 20) * -2 * (t / dt - 10) / 20 / dt]


# Nengo simulation
t_steps = 80  # Number of time steps
x_steps = 80  # Number of x steps
neurons = 2000
radius = 14
solver = Solver1D(feedback_connection, lateral_connection)

# Grid properties
c = 1
x_len = 80  # m
dx = x_len/x_steps
dt = dx/(2*c)  # dt chosen for stability

# Run nengo_pde simulation
solver.run_nengo_order2(dt, t_steps, x_steps, boundaries, neurons, radius)
fig, ax = solver.plot_population(0.001, False)
ax.set_xlabel('Time (s)')
ax.set_ylabel('$u$')
plt.show()
fig, ax = solver.plot_grid(t_steps, False)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature ($^{\circ}$C)')
plt.show()
solver.animate()

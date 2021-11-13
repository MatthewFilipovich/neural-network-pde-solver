"""Module solves heat equation in 1D using finite difference method and nengo_pde."""
from nengo_pde import Solver1D
import matplotlib.pyplot as plt


def feedback_connection(u):
    return - (K/dx**2) * 2*u


def lateral_connection(u):
    return K/dx**2 * u


# Nengo simulation
t_steps = 80  # Number of time steps
x_steps = 8  # Number of x steps
neurons = 500  # Number of neurons
radius = 100  # Radius of neurons
boundaries = [-50, 50]  # Constant boundary conditions
solver = Solver1D(feedback_connection, lateral_connection)

# Grid properties
K = 4.2
x_len = 20  # mm
dx = x_len/x_steps
dt = dx**2/(2*K**2)  # dt chosen for stability

# Run finite difference method simulation
solver.run_FDM_order1(dt, t_steps, x_steps, boundaries)
fig, ax = solver.plot_population(dt, False)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature ($^{\circ}$C)')
plt.show()
fig, ax = solver.plot_grid(t_steps, False)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature ($^{\circ}$C)')
plt.show()
solver.animate(nframes=t_steps)

# Run nengo_pde simulation
solver.run_nengo_order1(dt, t_steps, x_steps, boundaries, neurons, radius)
fig, ax = solver.plot_population(0.001, False)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature ($^{\circ}$C)')
plt.show()
fig, ax = solver.plot_grid(t_steps, False)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature ($^{\circ}$C)')
plt.show()
solver.animate(nframes=t_steps)

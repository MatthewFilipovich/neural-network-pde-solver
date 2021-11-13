"""Method contains Solver1D class for simulating PDEs on 1D grids."""
import nengo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Solver2D:

    def __init__(self, feedback_connection, lateral_connection):
        """Simulates and plots 1D PDEs using Nengo and standard numerical approach.

        Args:
            feedback_connection (function): Recurrent terms of discretized PDE.
            lateral_connection (function): Lateral terms of discretized PDE.
        """
        self.feedback_connection = feedback_connection
        self.lateral_connection = lateral_connection
        self.grid_values = None  # [time step, grid index]
        self.t_range = None
        self.dt = None

    @property
    def population_values(self):  # [grid index, time step]
        return np.swapaxes(self.grid_values, 0, 1)

    def _feedback_update(self, u):
        """Update equation for the feedback connection."""
        return u + self.dt*self.feedback_connection(u)

    def _lateral_update(self, u):
        """Update equation for the lateral connection"""
        return self.dt*self.lateral_connection(u)

    def run_nengo_order1(self, dt, t_steps, x_steps, boundaries, neurons, radius):
        """Run simulation using Nengo framework.

        Args:
            dt (float): Time step of PDE.
            t_steps (int): Number of time steps in simulation.
            x_steps (int): Number of spatial steps in simulation.
            boundaries (float): Boundary conditions.
            neurons (int): Number of neurons used per population.
            radius (float): The radius for training the neuron populations.
        """
        self.dt = dt
        model = nengo.Network()
        with model:
            states = []
            state_probes = []
            for i in range(x_steps):
                states.append([nengo.Ensemble(neurons, 1, radius) for _ in range(x_steps)])
                state_probes.append([nengo.Probe(state, synapse=dt) for state in states[i]])

            for i in range(x_steps):
                for j in range(x_steps):
                    if i == 0:
                        nengo.Connection(nengo.Node(boundaries),
                                         states[0][j], dt, self._lateral_update)
                    else:
                        nengo.Connection(states[i][j], states[i-1][j], dt, self._lateral_update)
                    if i == x_steps-1:
                        nengo.Connection(nengo.Node(boundaries),
                                         states[-1][j], dt, self._lateral_update)
                    else:
                        nengo.Connection(states[i][j], states[i+1][j], dt, self._lateral_update)

                    if j == 0:
                        nengo.Connection(nengo.Node(boundaries),
                                         states[i][0], dt, self._lateral_update)
                    else:
                        nengo.Connection(states[i][j], states[i][j-1], dt, self._lateral_update)
                    if j == x_steps-1:
                        nengo.Connection(nengo.Node(boundaries),
                                         states[i][-1], dt, self._lateral_update)
                    else:
                        nengo.Connection(states[i][j], states[i][j+1], dt, self._lateral_update)

                    nengo.Connection(states[i][j], states[i][j], dt, self._feedback_update)

        with nengo.Simulator(model, dt=0.001) as sim:
            sim.run(t_steps*dt)

        self.t_range = sim.trange()
        population_values = np.zeros((len(state_probes)+2, len(state_probes)+2, len(self.t_range)))
        population_values[0] = boundaries
        population_values[-1] = boundaries
        population_values[:, 0] = boundaries
        population_values[:, -1] = boundaries

        for i in range(len(state_probes)):
            for j in range(len(state_probes)):
                population_values[i+1][j+1] = np.asarray(sim.data[state_probes[i][j]]).flatten()
        self.grid_values = np.swapaxes(population_values, 0, 2)

    def plot_grid(self, t_step, show=True):
        """Plot the grid values at a certain time step."""
        t = t_step*self.dt
        t_index = (np.abs(self.t_range - t)).argmin()  # Finds closest t value
        plot_vals = self.grid_values[t_index]
        fig, ax = plt.subplots()
        im = ax.imshow(plot_vals)
        plt.colorbar(im)
        if show:
            plt.show()
        return fig, ax

    def animate(self, ylim=None, show=True, save=False):
        """Animate the grid values over time."""
        fig, ax = plt.subplots()
        im = ax.imshow(self.grid_values[0])
        cbar = plt.colorbar(im)
        cbar.set_label('Temperature (K)')

        def init():
            pass

        def update(tstep):
            im.set_data(self.grid_values[tstep])
            ax.set_title('Time: ' + str(self.t_range[tstep]))

        ani = animation.FuncAnimation(fig, update, frames=range(0, len(self.t_range), 10),
                                      init_func=init, blit=False, interval=1)
        if save is not False:
            ani.save(save+'.mp4', writer=animation.FFMpegWriter(fps=24))
        if show:
            plt.show()

        return ani

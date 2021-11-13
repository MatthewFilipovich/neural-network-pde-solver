"""Method contains Solver1D class for simulating PDEs on 1D grids."""
import nengo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Solver1D:

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
            boundaries (List[float, float]): First and end boundary conditions.
            neurons (int): Number of neurons used per population.
            radius (float): The radius for training the neuron populations.
        """
        self.dt = dt
        model = nengo.Network()
        with model:
            states = []
            for _ in range(x_steps):
                states.append(nengo.Ensemble(neurons, 1, radius))
            state_probes = [nengo.Probe(state, synapse=dt) for state in states]

            for i in range(x_steps):
                if i == 0:
                    nengo.Connection(nengo.Node(boundaries[0]),
                                     states[0], dt, self._lateral_update)
                else:
                    nengo.Connection(states[i], states[i-1], dt, self._lateral_update)
                if i == x_steps-1:
                    nengo.Connection(nengo.Node(boundaries[1]),
                                     states[-1], dt, self._lateral_update)
                else:
                    nengo.Connection(states[i], states[i+1], dt, self._lateral_update)

                nengo.Connection(states[i], states[i], dt, self._feedback_update)

        with nengo.Simulator(model, dt=0.001) as sim:
            sim.run(t_steps*dt)

        self.t_range = sim.trange()
        population_values = np.zeros((len(state_probes)+2, len(self.t_range)))
        population_values[0] = boundaries[0]
        population_values[-1] = boundaries[1]
        for i, state_probe in enumerate(state_probes):
            population_values[i+1] = np.asarray(sim.data[state_probe]).flatten()
        self.grid_values = np.swapaxes(population_values, 0, 1)

    def run_nengo_order2(self, dt, t_steps, x_steps, boundaries, neurons, radius):
        """Run simulation using Nengo framework.

        Args:
            dt (float): Time step of PDE.
            t_steps (int): Number of time steps in simulation.
            x_steps (int): Number of spatial steps in simulation.
            boundaries (List[float, float]): Val and derivative boundary conditions.
            neurons (int): Number of neurons used per population.
            radius (float): The radius for training the neuron populations.
        """
        self.dt = dt
        model = nengo.Network()
        with model:
            states = []
            for _ in range(x_steps):
                states.append(nengo.Ensemble(neurons, 2, radius))
            state_probes = [nengo.Probe(state, synapse=dt) for state in states]

            for i in range(x_steps):
                if i == 0:
                    nengo.Connection(nengo.Node(boundaries),
                                     states[0], dt, self._lateral_update)
                else:
                    nengo.Connection(states[i], states[i-1], dt, self._lateral_update)
                if i == x_steps-1:
                    nengo.Connection(nengo.Node((0, 0)),
                                     states[-1], dt, self._lateral_update)
                else:
                    nengo.Connection(states[i], states[i+1], dt, self._lateral_update)

                nengo.Connection(states[i], states[i], dt, self._feedback_update)

        with nengo.Simulator(model, dt=0.001) as sim:
            sim.run(t_steps*dt)
        self.t_range = sim.trange()
        population_values = np.zeros((len(state_probes)+2, len(self.t_range)))
        for i, state_probe in enumerate(state_probes):
            population_values[i+1] = np.asarray(sim.data[state_probe])[:, 0].flatten()
        self.grid_values = np.swapaxes(population_values, 0, 1)

    def run_FDM_order1(self, dt, t_steps, x_steps, boundaries):
        """Run simulation using finite difference method (standard numerical method).

        Args:
            dt (float): Time step of PDE.
            t_steps (int): Number of time steps in simulation.
            x_steps (int): Number of spatial steps in simulation.
            boundaries (List[float, float]): First and end boundary conditions.
        """
        self.dt = dt
        self.t_range = dt*np.arange(t_steps)
        u = np.zeros(x_steps+2)  # + 2 for boundaries
        u[0] = boundaries[0]
        u[-1] = boundaries[1]
        self.grid_values = np.zeros((t_steps, x_steps+2))
        self.grid_values[0] = u

        for t_step in range(1, t_steps):
            u[1:-1] = (self._feedback_update(u[1:-1]) + self._lateral_update(u[:-2])
                       + self._lateral_update(u[2:]))
            self.grid_values[t_step] = u

    def plot_grid(self, t_step, show=True):
        """Plot the grid values at a certain time step."""
        t = t_step*self.dt
        t_index = (np.abs(self.t_range - t)).argmin()  # Finds closest t value
        plot_vals = self.grid_values[t_index]
        fig, ax = plt.subplots()
        ax.plot(plot_vals)
        ax.set_xlim(0, len(plot_vals)-1)
        ax.set_title('Time: ' + str(t))
        if show:
            plt.show()
        return fig, ax

    def plot_population(self, dt, show=True):
        """Plot the populations as functions of time."""
        fig, ax = plt.subplots()
        for pop in self.population_values:
            ax.plot(dt*np.arange(len(pop)), pop)
        ax.set_xlim(0, dt*len(pop)-dt)
        if show:
            plt.show()
        return fig, ax

    def animate(self, ylim=None, show=True, nframes=200, save=False):
        """Animate the grid values over time."""
        fig, ax = plt.subplots()
        ln, = ax.plot(self.grid_values[0])
        ax.set_xlim(0, len(self.grid_values[0])-1)
        if ylim == None:
            ylim = (np.amin(self.grid_values), np.amax(self.grid_values))
        ax.set_ylim(*ylim)

        def init():
            pass

        def update(tstep):
            ln.set_data((range(len(self.grid_values[tstep])), self.grid_values[tstep]))
            ax.set_title('Time: ' + str(self.t_range[tstep]))
            return ln,

        ani = animation.FuncAnimation(fig, update,
                                      frames=range(
                                          0, len(self.t_range),
                                          int(len(self.t_range) / nframes)),
                                      init_func=init, blit=False, interval=1)
        if save is not False:
            ani.save(save+'.mp4', writer=animation.FFMpegWriter(fps=24))
        if show:
            plt.show()
        return ani

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy import stats

class LJVisualizerExtended:
    """Visualization tools for Lennard-Jones simulation"""

    def __init__(self, simulator):
        """
        Initialize visualizer

        Parameters:
        -----------
        simulator : LJSimulator
            The simulator instance to visualize
        """
        self.simulator = simulator

    def plot_energy(self, start_step=0):
        """
        Plot energy components over time

        Parameters:
        -----------
        start_step : int
            Index to start plotting from (to skip transient behavior)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        time = np.array(self.simulator.time_history[start_step:])
        pe = np.array(self.simulator.potential_energy_history[start_step:])
        ke = np.array(self.simulator.kinetic_energy_history[start_step:])
        te = np.array(self.simulator.total_energy_history[start_step:])

        ax.plot(time, pe, 'r-', label='Potential Energy')
        ax.plot(time, ke, 'b-', label='Kinetic Energy')
        ax.plot(time, te, 'g-', label='Total Energy')

        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.set_title('Energy vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def plot_temperature(self, start_step=0):
        """
        Plot temperature over time

        Parameters:
        -----------
        start_step : int
            Index to start plotting from (to skip transient behavior)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        time = np.array(self.simulator.time_history[start_step:])
        temp = np.array(self.simulator.temperature_history[start_step:])

        ax.plot(time, temp, 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature')
        ax.set_title('Temperature vs Time')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def plot_pressure(self, start_step=0):
        """
        Plot pressure over time

        Parameters:
        -----------
        start_step : int
            Index to start plotting from (to skip transient behavior)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        time = np.array(self.simulator.time_history[start_step:])
        pressure = np.array(self.simulator.pressure_history[start_step:])

        ax.plot(time, pressure, 'b-')
        ax.set_xlabel('Time')
        ax.set_ylabel('Pressure')
        ax.set_title('Pressure vs Time')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def plot_rdf(self, num_bins=100, max_r=None):
        """
        Plot the radial distribution function

        Parameters:
        -----------
        num_bins : int
            Number of bins for the histogram
        max_r : float or None
            Maximum distance to consider
        """
        r_values, g_r = self.simulator.compute_rdf(num_bins, max_r)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(r_values, g_r, 'b-')
        ax.set_xlabel('r')
        ax.set_ylabel('g(r)')
        ax.set_title('Radial Distribution Function')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def plot_configuration(self, snapshot_index=-1, particle_size=100, alpha=0.7):
        """
        Plot a specific configuration from snapshots

        Parameters:
        -----------
        snapshot_index : int
            Index of the snapshot to plot (-1 for latest)
        particle_size : float
            Size of particles in scatter plot
        alpha : float
            Transparency of particles
        """
        if not self.simulator.snapshots:
            print("No snapshots available")
            return None, None

        snapshot = self.simulator.snapshots[snapshot_index]
        positions = snapshot['positions']

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot particles
        ax.scatter(positions[:, 0], positions[:, 1], s=particle_size,
                   alpha=alpha, c='b', edgecolors='k')

        # Set axis limits with a small margin
        ax.set_xlim(-0.05 * self.simulator.Lx, 1.05 * self.simulator.Lx)
        ax.set_ylim(-0.05 * self.simulator.Ly, 1.05 * self.simulator.Ly)

        # Draw box boundaries
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax.axhline(y=self.simulator.Ly, color='k', linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        ax.axvline(x=self.simulator.Lx, color='k', linestyle='-', alpha=0.5)

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Particle Configuration at t = {snapshot["time"]:.2f}')

        plt.tight_layout()
        return fig, ax

    def animate(self, interval=50, particle_size=100, alpha=0.7, step=1):
        """
        Create an animation of the simulation

        Parameters:
        -----------
        interval : int
            Interval between frames in milliseconds
        particle_size : float
            Size of particles in scatter plot
        alpha : float
            Transparency of particles
        step : int
            Step size between snapshots (to skip frames)

        Returns:
        --------
        HTML animation that can be displayed in Jupyter
        """
        if not self.simulator.snapshots:
            print("No snapshots available")
            return None

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-0.05 * self.simulator.Lx, 1.05 * self.simulator.Lx)
        ax.set_ylim(-0.05 * self.simulator.Ly, 1.05 * self.simulator.Ly)

        # Draw box boundaries
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax.axhline(y=self.simulator.Ly, color='k', linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        ax.axvline(x=self.simulator.Lx, color='k', linestyle='-', alpha=0.5)

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Initialize scatter plot
        particles = ax.scatter([], [], s=particle_size, alpha=alpha, c='b', edgecolors='k')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            particles.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            return particles, time_text

        def update(frame):
            i = frame * step
            if i >= len(self.simulator.snapshots):
                i = len(self.simulator.snapshots) - 1

            snapshot = self.simulator.snapshots[i]
            positions = snapshot['positions']

            particles.set_offsets(positions)
            time_text.set_text(f't = {snapshot["time"]:.2f}')
            return particles, time_text

        frames = len(self.simulator.snapshots) // step
        anim = FuncAnimation(fig, update, frames=frames,
                             init_func=init, blit=True, interval=interval)

        plt.close(fig)  # Prevent display of the figure
        return HTML(anim.to_jshtml())

    def plot_velocity_distribution(self, bins=30, snapshot_index=-1):
        """
        Plot the velocity distribution compared to Maxwell-Boltzmann (Problem 8.6)

        Parameters:
        -----------
        bins : int
            Number of bins for histogram
        snapshot_index : int
            Which snapshot to use (-1 for latest)
        """
        if not self.simulator.snapshots:
            print("No snapshots available")
            return None, None

        # Use the specified snapshot
        snapshot = self.simulator.snapshots[snapshot_index]
        velocities = snapshot['velocities']
        temperature = snapshot['temperature']

        # Extract velocity components
        vx = velocities[:, 0]
        vy = velocities[:, 1]
        v_components = np.concatenate([vx, vy])

        # Calculate speeds (magnitude of velocity vectors)
        speeds = np.sqrt(vx**2 + vy**2)

        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot velocity components distribution (vx and vy)
        sns.histplot(v_components, bins=bins, kde=True, stat='density', color='blue',
                     alpha=0.6, label='Simulation', ax=ax1)

        # Generate theoretical Maxwell-Boltzmann for velocity components (1D Gaussian)
        v_range = np.linspace(min(v_components), max(v_components), 1000)
        mb_velocity = stats.norm.pdf(v_range, loc=0, scale=np.sqrt(temperature))
        ax1.plot(v_range, mb_velocity, 'r-', label='Maxwell-Boltzmann')

        ax1.set_xlabel('Velocity Component')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Velocity Components Distribution')
        ax1.legend()

        # Plot speed distribution
        sns.histplot(speeds, bins=bins, kde=True, stat='density', color='green',
                     alpha=0.6, label='Simulation', ax=ax2)

        # Generate theoretical Maxwell-Boltzmann for speeds (Rayleigh distribution in 2D)
        s_range = np.linspace(0, max(speeds)*1.2, 1000)
        # In 2D, P(v)dv = (m/2πkT) * exp(-mv²/2kT) * 2πv dv = (mv/kT) * exp(-mv²/2kT) dv
        mb_speed = (s_range/temperature) * np.exp(-(s_range**2)/(2*temperature))
        ax2.plot(s_range, mb_speed, 'r-', label='Maxwell-Boltzmann')

        ax2.set_xlabel('Speed')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Speed Distribution')
        ax2.legend()

        plt.tight_layout()
        return fig, (ax1, ax2)

    def plot_combined_velocity_distribution(self, bins=30, snapshot_index=-1):
        """
        Plot the average of x and y velocity component distributions (Problem 8.6a)

        Parameters:
        -----------
        bins : int
            Number of bins for histogram
        snapshot_index : int
            Which snapshot to use (-1 for latest)
        """
        if not self.simulator.snapshots:
            print("No snapshots available")
            return None, None

        snapshot = self.simulator.snapshots[snapshot_index]
        velocities = snapshot['velocities']
        temperature = snapshot['temperature']

        vx = velocities[:, 0]
        vy = velocities[:, 1]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create histograms for vx and vy with the same bins
        v_min = min(min(vx), min(vy))
        v_max = max(max(vx), max(vy))
        bin_edges = np.linspace(v_min, v_max, bins+1)

        # Compute histograms
        hist_x, _ = np.histogram(vx, bins=bin_edges, density=True)
        hist_y, _ = np.histogram(vy, bins=bin_edges, density=True)

        # Compute bin centers
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Compute average of P(vx) and P(vy)
        hist_avg = 0.5 * (hist_x + hist_y)

        # Plot the average
        ax.plot(bin_centers, hist_avg, 'g-', linewidth=2,
                label='Average of P(vx) and P(vy)')

        # Plot the individual histograms for comparison
        sns.histplot(vx, bins=bin_edges, kde=True, stat='density', color='blue',
                     alpha=0.4, label='P(vx)', ax=ax)
        sns.histplot(vy, bins=bin_edges, kde=True, stat='density', color='red',
                     alpha=0.4, label='P(vy)', ax=ax)

        # Plot the theoretical Maxwell-Boltzmann distribution
        v_range = np.linspace(v_min, v_max, 1000)
        mb_pdf = (1/np.sqrt(2*np.pi*temperature)) * np.exp(-v_range**2/(2*temperature))
        ax.plot(v_range, mb_pdf, 'k--', linewidth=2, label='Maxwell-Boltzmann')

        ax.set_xlabel('Velocity Component (u)')
        ax.set_ylabel('Probability Density P(u)')
        ax.set_title('Average of P(vx) and P(vy) vs u')
        ax.legend()

        plt.tight_layout()
        return fig, ax

    def plot_ideal_gas_comparison(self, start_step=0, end_step=None):
        """
        Compare the ratio PV/NkT with the ideal gas value (Problem 8.7)

        Parameters:
        -----------
        start_step : int
            Starting index for average calculation
        end_step : int or None
            Ending index for average calculation
        """
        if end_step is None:
            end_step = len(self.simulator.pressure_history)

        pressure = np.array(self.simulator.pressure_history[start_step:end_step])
        temperature = np.array(self.simulator.temperature_history[start_step:end_step])

        # For an ideal gas, PV/NkT = 1
        # Calculate ratio for our system (k=1 in reduced units)
        ratio = pressure * (self.simulator.Lx * self.simulator.Ly) / (self.simulator.N * temperature)

        fig, ax = plt.subplots(figsize=(10, 6))

        time = np.array(self.simulator.time_history[start_step:end_step])
        ax.plot(time, ratio, 'b-', label='Simulation')
        ax.axhline(y=1.0, color='r', linestyle='--', label='Ideal Gas')

        ax.set_xlabel('Time')
        ax.set_ylabel('PV/NkT')
        ax.set_title('Comparison with Ideal Gas Equation of State')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Calculate and display mean value
        mean_ratio = np.mean(ratio)
        ax.text(0.05, 0.95, f'Mean PV/NkT = {mean_ratio:.4f}',
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig, ax

    def plot_heat_capacity(self, start_step=0, end_step=None, method='fluctuations'):
        """
        Calculate and plot heat capacity (Problem 8.8)

        Parameters:
        -----------
        start_step : int
            Starting index for analysis
        end_step : int or None
            Ending index for analysis
        method : str
            Method for calculating heat capacity:
            - 'fluctuations': Using temperature fluctuations (Ray and Graben)
            - 'energy_temp': Using E(T) relationship
        """
        if end_step is None:
            end_step = len(self.simulator.temperature_history)

        temperature = np.array(self.simulator.temperature_history[start_step:end_step])

        fig, ax = plt.subplots(figsize=(10, 6))

        if method == 'fluctuations':
            # Calculate Cv using temperature fluctuations (equation 8.12)
            # Cv = (dNk/2)[1 - (2N/d)((T²) - (T)²)/(kT)²]⁻¹

            # For 2D system, d = 2
            d = 2
            T_mean = np.mean(temperature)
            T_squared_mean = np.mean(temperature**2)

            # Calculate heat capacity
            sigma_squared = T_squared_mean - T_mean**2
            denominator = sigma_squared / (self.simulator.N * T_mean**2) - 1.0
            Cv = self.simulator.N / denominator

            ax.text(0.05, 0.95, f'Heat Capacity (Cv) = {Cv:.4f}',
                    transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            ax.set_title('Heat Capacity from Temperature Fluctuations')

        elif method == 'energy_temp':
            # Calculate Cv using E(T) relationship
            # Cv = dE/dT

            energy = np.array(self.simulator.total_energy_history[start_step:end_step])

            # Sort by temperature to ensure monotonicity
            sorted_indices = np.argsort(temperature)
            sorted_temp = temperature[sorted_indices]
            sorted_energy = energy[sorted_indices]

            # Smooth the data
            window_size = min(15, len(sorted_temp) // 10)
            if window_size > 1:
                from scipy.signal import savgol_filter
                smoothed_energy = savgol_filter(sorted_energy, window_size, 3)
            else:
                smoothed_energy = sorted_energy

            # Calculate derivative dE/dT
            dE = np.diff(smoothed_energy)
            dT = np.diff(sorted_temp)
            Cv = dE / dT
            T_points = 0.5 * (sorted_temp[1:] + sorted_temp[:-1])

            # Plot E vs T
            ax.plot(sorted_temp, sorted_energy, 'o', markersize=3, alpha=0.5, label='E(T) data')
            ax.plot(sorted_temp, smoothed_energy, 'r-', label='Smoothed E(T)')

            # Create a second y-axis for Cv
            ax2 = ax.twinx()
            ax2.plot(T_points, Cv, 'g-', label='Cv = dE/dT')
            ax2.set_ylabel('Heat Capacity (Cv)')

            # Calculate mean Cv
            mean_Cv = np.mean(Cv)
            ax2.text(0.05, 0.90, f'Mean Cv = {mean_Cv:.4f}',
                     transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

            ax.set_xlabel('Temperature')
            ax.set_ylabel('Energy')
            ax.set_title('Heat Capacity from E(T) Relationship')

            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        return fig, ax

    def plot_lattice_energy_comparison(self, square_energy, triangular_energy):
        """
        Compare potential energy of square and triangular lattices (Problem 8.9)

        Parameters:
        -----------
        square_energy : float
            Potential energy per particle for square lattice
        triangular_energy : float
            Potential energy per particle for triangular lattice
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        lattice_types = ['Square', 'Triangular']
        energies = [square_energy, triangular_energy]

        ax.bar(lattice_types, energies, color=['blue', 'green'])

        ax.set_ylabel('Potential Energy per Particle')
        ax.set_title('Comparison of Lattice Energies')

        # Add energy values on top of bars
        for i, v in enumerate(energies):
            ax.text(i, v + 0.05, f"{v:.4f}", ha='center')

        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def plot_melting_analysis(self, temperatures, energies, pressures):
        """
        Plot E(T) and P(T) for analyzing melting behavior (Problem 8.11)

        Parameters:
        -----------
        temperatures : list or array
            List of temperatures
        energies : list or array
            Corresponding energies
        pressures : list or array
            Corresponding pressures
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Convert to numpy arrays if not already
        temperatures = np.array(temperatures)
        energies = np.array(energies)
        pressures = np.array(pressures)

        # Calculate E(T) - E(0)
        if len(temperatures) > 0:
            e_minus_e0 = energies - energies[0]
        else:
            e_minus_e0 = energies

        # Plot E(T) - E(0) vs T
        ax1.plot(temperatures, e_minus_e0, 'bo-')
        ax1.set_xlabel('Temperature (T)')
        ax1.set_ylabel('E(T) - E(0)')
        ax1.set_title('Energy vs Temperature')
        ax1.grid(True, alpha=0.3)

        # Plot P(T) vs T
        ax2.plot(temperatures, pressures, 'ro-')
        ax2.set_xlabel('Temperature (T)')
        ax2.set_ylabel('Pressure P(T)')
        ax2.set_title('Pressure vs Temperature')
        ax2.grid(True, alpha=0.3)

        # Check if E(T) - E(0) is proportional to T
        if len(temperatures) > 2:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(temperatures, e_minus_e0)

            # Plot the linear fit
            fit_line = slope * temperatures + intercept
            ax1.plot(temperatures, fit_line, 'g--',
                    label=f'Fit: y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}')
            ax1.legend()

            # Add annotation about heat capacity
            ax1.text(0.05, 0.95, f'Heat Capacity (slope) = {slope:.4f}',
                    transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig, (ax1, ax2)

    def plot_trajectories(self, duration=10.0, num_particles=5, skip_steps=1):
        """
        Plot trajectories of selected particles over time
        Useful for Problems 8.10 and 8.11 to analyze particle motion

        Parameters:
        -----------
        duration : float
            Duration to plot trajectories for
        num_particles : int
            Number of randomly selected particles to track
        skip_steps : int
            Number of steps to skip between trajectory points
        """
        if not self.simulator.snapshots:
            print("No snapshots available")
            return None, None

        # Select random particles to track
        particles_to_track = np.random.choice(self.simulator.N,
                                             size=min(num_particles, self.simulator.N),
                                             replace=False)

        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw simulation box
        ax.set_xlim(-0.05 * self.simulator.Lx, 1.05 * self.simulator.Lx)
        ax.set_ylim(-0.05 * self.simulator.Ly, 1.05 * self.simulator.Ly)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axhline(y=self.simulator.Ly, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=self.simulator.Lx, color='k', linestyle='-', alpha=0.3)

        # Find snapshots within the time range
        max_time = self.simulator.snapshots[-1]['time']
        start_time = max(0.0, max_time - duration)

        relevant_snapshots = [s for s in self.simulator.snapshots
                             if s['time'] >= start_time]
        relevant_snapshots = relevant_snapshots[::skip_steps]

        # Different colors for different particles
        colors = plt.cm.jet(np.linspace(0, 1, len(particles_to_track)))

        # Plot trajectories
        for i, particle_idx in enumerate(particles_to_track):
            x_traj = []
            y_traj = []

            for snapshot in relevant_snapshots:
                x_traj.append(snapshot['positions'][particle_idx, 0])
                y_traj.append(snapshot['positions'][particle_idx, 1])

            ax.plot(x_traj, y_traj, '-', color=colors[i], alpha=0.7,
                   label=f'Particle {particle_idx}')

            # Mark the final position
            ax.scatter(x_traj[-1], y_traj[-1], color=colors[i],
                      edgecolors='k', s=100, zorder=10)

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Particle Trajectories (t = {start_time:.2f} to {max_time:.2f})')
        ax.legend(loc='upper right')

        plt.tight_layout()
        return fig, ax

    ############################################
    ##### Time Based Visualization Helpers #####
    ############################################
    def find_snapshots_by_time(self, start_time=None, end_time=None):
        """
        Find snapshots within a specified time range

        Parameters:
        -----------
        start_time : float or None
            Start time (None means beginning of simulation)
        end_time : float or None
            End time (None means end of simulation)

        Returns:
        --------
        list of snapshots within the time range
        """
        if not self.simulator.snapshots:
            return []

        # Set default values if None
        if start_time is None:
            start_time = self.simulator.snapshots[0]['time']
        if end_time is None:
            end_time = self.simulator.snapshots[-1]['time']

        # Filter snapshots by time range
        return [s for s in self.simulator.snapshots
                if start_time <= s['time'] <= end_time]

    def plot_configuration_by_time(self, time, particle_size=100, alpha=0.7):
        """
        Plot particle configuration at the snapshot closest to the specified time

        Parameters:
        -----------
        time : float
            The time to find the nearest snapshot for
        particle_size : float
            Size of particles in scatter plot
        alpha : float
            Transparency of particles

        Returns:
        --------
        Matplotlib figure and axes
        """
        if not self.simulator.snapshots:
            print("No snapshots available")
            return None, None

        # Find the snapshot closest to the specified time
        closest_snapshot = min(self.simulator.snapshots,
                            key=lambda s: abs(s['time'] - time))

        positions = closest_snapshot['positions']
        actual_time = closest_snapshot['time']

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot particles
        ax.scatter(positions[:, 0], positions[:, 1], s=particle_size,
                alpha=alpha, c='b', edgecolors='k')

        # Set axis limits with a small margin
        ax.set_xlim(-0.05 * self.simulator.Lx, 1.05 * self.simulator.Lx)
        ax.set_ylim(-0.05 * self.simulator.Ly, 1.05 * self.simulator.Ly)

        # Draw box boundaries
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax.axhline(y=self.simulator.Ly, color='k', linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        ax.axvline(x=self.simulator.Lx, color='k', linestyle='-', alpha=0.5)

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Particle Configuration at t = {actual_time:.2f} (requested: {time:.2f})')

        plt.tight_layout()
        return fig, ax

    def plot_energy_by_time(self, start_time=None, end_time=None):
        """
        Plot energy components for a specific time range

        Parameters:
        -----------
        start_time : float or None
            Start time (None means beginning of simulation)
        end_time : float or None
            End time (None means end of simulation)

        Returns:
        --------
        Matplotlib figure and axes
        """
        # Convert time history to numpy array for faster comparisons
        time_array = np.array(self.simulator.time_history)

        # Set default values if None
        if start_time is None:
            start_time = time_array[0]
        if end_time is None:
            end_time = time_array[-1]

        # Find indices within the time range
        time_mask = (time_array >= start_time) & (time_array <= end_time)
        time_indices = np.where(time_mask)[0]

        if len(time_indices) == 0:
            print(f"No data found in the time range {start_time} to {end_time}")
            return None, None

        # Extract data for the time range
        times = time_array[time_indices]
        pe = np.array(self.simulator.potential_energy_history)[time_indices]
        ke = np.array(self.simulator.kinetic_energy_history)[time_indices]
        te = np.array(self.simulator.total_energy_history)[time_indices]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(times, pe, 'r-', label='Potential Energy')
        ax.plot(times, ke, 'b-', label='Kinetic Energy')
        ax.plot(times, te, 'g-', label='Total Energy')

        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.set_title(f'Energy vs Time (t = {start_time:.2f} to {end_time:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def plot_temperature_by_time(self, start_time=None, end_time=None):
        """
        Plot temperature for a specific time range

        Parameters:
        -----------
        start_time : float or None
            Start time (None means beginning of simulation)
        end_time : float or None
            End time (None means end of simulation)

        Returns:
        --------
        Matplotlib figure and axes
        """
        # Convert time history to numpy array for faster comparisons
        time_array = np.array(self.simulator.time_history)

        # Set default values if None
        if start_time is None:
            start_time = time_array[0]
        if end_time is None:
            end_time = time_array[-1]

        # Find indices within the time range
        time_mask = (time_array >= start_time) & (time_array <= end_time)
        time_indices = np.where(time_mask)[0]

        if len(time_indices) == 0:
            print(f"No data found in the time range {start_time} to {end_time}")
            return None, None

        # Extract data for the time range
        times = time_array[time_indices]
        temp = np.array(self.simulator.temperature_history)[time_indices]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(times, temp, 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature')
        ax.set_title(f'Temperature vs Time (t = {start_time:.2f} to {end_time:.2f})')
        ax.grid(True, alpha=0.3)

        # Calculate and display mean temperature
        mean_temp = np.mean(temp)
        ax.axhline(y=mean_temp, color='k', linestyle='--', alpha=0.7)
        ax.text(0.05, 0.95, f'Mean Temperature = {mean_temp:.4f}',
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig, ax

    def plot_pressure_by_time(self, start_time=None, end_time=None):
        """
        Plot pressure for a specific time range

        Parameters:
        -----------
        start_time : float or None
            Start time (None means beginning of simulation)
        end_time : float or None
            End time (None means end of simulation)

        Returns:
        --------
        Matplotlib figure and axes
        """
        # Convert time history to numpy array for faster comparisons
        time_array = np.array(self.simulator.time_history)

        # Set default values if None
        if start_time is None:
            start_time = time_array[0]
        if end_time is None:
            end_time = time_array[-1]

        # Find indices within the time range
        time_mask = (time_array >= start_time) & (time_array <= end_time)
        time_indices = np.where(time_mask)[0]

        if len(time_indices) == 0:
            print(f"No data found in the time range {start_time} to {end_time}")
            return None, None

        # Extract data for the time range
        times = time_array[time_indices]
        pressure = np.array(self.simulator.pressure_history)[time_indices]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(times, pressure, 'b-')
        ax.set_xlabel('Time')
        ax.set_ylabel('Pressure')
        ax.set_title(f'Pressure vs Time (t = {start_time:.2f} to {end_time:.2f})')
        ax.grid(True, alpha=0.3)

        # Calculate and display mean pressure
        mean_pressure = np.mean(pressure)
        ax.axhline(y=mean_pressure, color='k', linestyle='--', alpha=0.7)
        ax.text(0.05, 0.95, f'Mean Pressure = {mean_pressure:.4f}',
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig, ax

    def animate_time_range(self, start_time=None, end_time=None, interval=50,
                        particle_size=100, alpha=0.7, max_frames=200):
        """
        Create an animation for a specific time range

        Parameters:
        -----------
        start_time : float or None
            Start time (None means beginning of simulation)
        end_time : float or None
            End time (None means end of simulation)
        interval : int
            Interval between frames in milliseconds
        particle_size : float
            Size of particles in scatter plot
        alpha : float
            Transparency of particles
        max_frames : int
            Maximum number of frames to include (to prevent very large animations)

        Returns:
        --------
        HTML animation that can be displayed in Jupyter
        """
        # Get snapshots within the time range
        snapshots = self.find_snapshots_by_time(start_time, end_time)

        if not snapshots:
            print("No snapshots available in the specified time range")
            return None

        # Downsample if necessary
        if len(snapshots) > max_frames:
            step = len(snapshots) // max_frames
            snapshots = snapshots[::step]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-0.05 * self.simulator.Lx, 1.05 * self.simulator.Lx)
        ax.set_ylim(-0.05 * self.simulator.Ly, 1.05 * self.simulator.Ly)

        # Draw box boundaries
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax.axhline(y=self.simulator.Ly, color='k', linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        ax.axvline(x=self.simulator.Lx, color='k', linestyle='-', alpha=0.5)

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Initialize scatter plot
        particles = ax.scatter([], [], s=particle_size, alpha=alpha, c='b', edgecolors='k')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            particles.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            return particles, time_text

        def update(frame):
            snapshot = snapshots[frame]
            positions = snapshot['positions']

            particles.set_offsets(positions)
            time_text.set_text(f't = {snapshot["time"]:.2f}')
            return particles, time_text

        anim = FuncAnimation(fig, update, frames=len(snapshots),
                            init_func=init, blit=True, interval=interval)

        plt.close(fig)  # Prevent display of the figure
        return HTML(anim.to_jshtml())
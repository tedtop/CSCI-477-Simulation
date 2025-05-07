# HourglassVisualizer.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
from IPython.display import HTML

class HourglassVisualizer:
    """
    Visualization tools for the HourglassSimulator
    """
    def __init__(self, simulator):
        """
        Initialize the visualizer with a simulator instance

        Parameters:
        -----------
        simulator : HourglassSimulator
            The simulator to visualize
        """
        self.sim = simulator

    def create_animation(self, frames=500, interval=20, limited_duration=None, steps_per_frame=5):
        """
        Create an animation of the simulation

        Parameters:
        -----------
        frames : int
            Number of animation frames
        interval : int
            Delay between frames in milliseconds
        limited_duration : float
            If provided, run the simulation for this amount of time instead of frames
        steps_per_frame : int
            Simulation steps per animation frame

        Returns:
        --------
        Animation object
        """
        # Set up the figure and axes
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3)

        # Main simulation display
        ax_sim = fig.add_subplot(gs[:, 0:2])
        ax_sim.set_xlim(0, self.sim.Lx)
        ax_sim.set_ylim(0, self.sim.Ly)
        ax_sim.set_aspect('equal')
        ax_sim.grid(True)
        ax_sim.set_title("Hourglass Simulator")
        ax_sim.set_xlabel("X position")
        ax_sim.set_ylabel("Y position")

        # Energy plot
        ax_energy = fig.add_subplot(gs[0, 2])
        ax_energy.set_title("Energy vs Time")
        # ax_energy.set_xlabel("Time")
        ax_energy.set_ylabel("Energy")
        ax_energy.grid(True)

        # Temperature plot
        ax_temp = fig.add_subplot(gs[1, 2])
        ax_temp.set_title("Kinetic Temperature vs Time")
        # ax_temp.set_xlabel("Time")
        ax_temp.set_ylabel("Temperature")
        ax_temp.grid(True)

        # Flow plot
        ax_flow = fig.add_subplot(gs[2, 2])
        ax_flow.set_title("Particles Through Neck vs Time")
        ax_flow.set_xlabel("Time")
        ax_flow.set_ylabel("Particle Count")
        ax_flow.grid(True)

        # Set up the limits for the plots based on simulation time
        # Calculate expected maximum time
        if limited_duration:
            max_time = limited_duration
        else:
            # Estimate max time based on frames and steps per frame
            max_time = self.sim.time + (frames * steps_per_frame * self.sim.dt)

        ax_energy.set_xlim(0, max_time)
        ax_energy.set_ylim(0, 1)  # Will auto-adjust
        ax_temp.set_xlim(0, max_time)
        ax_temp.set_ylim(0, 1)  # Will auto-adjust
        ax_flow.set_xlim(0, max_time)
        ax_flow.set_ylim(0, self.sim.N)

        # Create the particles as circles
        circles = []
        for i in range(self.sim.N):
            # Color based on vertical position (lighter at top)
            color_val = 0.7 + 0.3 * (self.sim.y[i] / self.sim.Ly)
            circle = Circle((self.sim.x[i], self.sim.y[i]),
                           radius=self.sim.r[i],
                           facecolor=(0.9, color_val * 0.8, 0.4),  # Sandy color
                           edgecolor='#996633',  # Brown edge
                           alpha=0.9)
            ax_sim.add_patch(circle)
            circles.append(circle)

        # Add box boundaries
        box = plt.Rectangle((0, 0), self.sim.Lx, self.sim.Ly, fill=False,
                           edgecolor='black', linewidth=2)
        ax_sim.add_patch(box)

        # Add hourglass walls
        self.draw_walls(ax_sim)

        # Add text for information
        time_text = ax_sim.text(0.02, 0.96, '', transform=ax_sim.transAxes, fontsize=10)
        flow_text = ax_sim.text(0.02, 0.92, '', transform=ax_sim.transAxes, fontsize=10)
        respawn_text = ax_sim.text(0.02, 0.88, '', transform=ax_sim.transAxes, fontsize=10)

        # Initialize data arrays for plots
        time_data = []
        energy_data = []
        kinetic_data = []
        potential_data = []
        temperature_data = []
        particles_through_neck_data = []

        # Initialize set to track particles that have passed through the neck
        particles_passed_through = set()
        neck_y = self.sim.Ly / 2
        neck_width = self.sim.neck_width
        neck_x_min = (self.sim.Lx - neck_width) / 2
        neck_x_max = neck_x_min + neck_width

        # Create plot lines with empty initial data
        energy_line, = ax_energy.plot([], [], 'k-', label='Total')
        kinetic_line, = ax_energy.plot([], [], 'r-', label='Kinetic')
        potential_line, = ax_energy.plot([], [], 'b-', label='Potential')
        ax_energy.legend(loc='upper right')

        temperature_line, = ax_temp.plot([], [], 'g-')
        flow_line, = ax_flow.plot([], [], 'm-', linewidth=2)

        # Animation update function
        def update(frame):
            nonlocal time_data, energy_data, kinetic_data, potential_data
            nonlocal temperature_data, particles_through_neck_data, particles_passed_through

            # For limited duration, check if we've reached the time limit
            if limited_duration and self.sim.time >= limited_duration:
                return circles + [time_text, flow_text, respawn_text,
                                 energy_line, kinetic_line, potential_line,
                                 temperature_line, flow_line]

            # Run simulation steps
            for _ in range(steps_per_frame):
                self.sim.step()

            # Collect data using the actual simulation time
            current_time = self.sim.time
            time_data.append(current_time)
            energy_data.append(self.sim.total_energy)
            kinetic_data.append(self.sim.kinetic_energy)
            potential_data.append(self.sim.potential_energy)
            temperature_data.append(self.sim.temperature)

            # Track particles passing through the hourglass neck
            for i in range(self.sim.N):
                # If particle has crossed the neck line from above to below
                if (self.sim.y[i] < neck_y and
                    i not in particles_passed_through and
                    neck_x_min < self.sim.x[i] < neck_x_max):
                    particles_passed_through.add(i)

            # Store particle count
            particles_through_neck_data.append(len(particles_passed_through))

            # Update particle positions
            for i, circle in enumerate(circles):
                circle.center = (self.sim.x[i], self.sim.y[i])

                # Update color based on position (optional)
                color_val = 0.7 + 0.3 * (self.sim.y[i] / self.sim.Ly)
                circle.set_facecolor((0.9, color_val * 0.8, 0.4))

            # Update info text with actual simulation time
            time_text.set_text(f'Time: {current_time:.2f}')
            flow_text.set_text(f'Particles passed: {len(particles_passed_through)}')

            if self.sim.respawn_particles:
                stats = self.sim.get_respawn_stats()
                respawn_text.set_text(f'Total respawns: {stats["total_respawns"]}')
            else:
                respawn_text.set_text('')

            # Auto-adjust y-limits for energy and temperature if needed
            if energy_data and max(energy_data) > ax_energy.get_ylim()[1]:
                ax_energy.set_ylim(0, max(energy_data) * 1.2)

            if temperature_data and max(temperature_data) > ax_temp.get_ylim()[1]:
                ax_temp.set_ylim(0, max(temperature_data) * 1.2)

            # Update the plot lines with new data
            energy_line.set_data(time_data, energy_data)
            kinetic_line.set_data(time_data, kinetic_data)
            potential_line.set_data(time_data, potential_data)
            temperature_line.set_data(time_data, temperature_data)
            flow_line.set_data(time_data, particles_through_neck_data)

            # Return all artists that were updated
            return circles + [time_text, flow_text, respawn_text,
                             energy_line, kinetic_line, potential_line,
                             temperature_line, flow_line]

        # Create animation
        ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

        # Return the animation object and figures for display
        return ani, fig

    def draw_walls(self, ax):
        """Draw walls on the provided axis"""
        # Draw left wall segments
        if self.sim.left_wall_segments:
            for segment in self.sim.left_wall_segments:
                # Create a polygon for the wall segment
                wall_polygon = Polygon([
                    [segment['top_x'], segment['top_y']],
                    [segment['bottom_x'], segment['bottom_y']],
                    [segment['bottom_x'] + segment['width'], segment['bottom_y']],
                    [segment['top_x'] + segment['width'], segment['top_y']]
                ], closed=True, fill=True, facecolor='gray', alpha=0.5, edgecolor='black', linewidth=1.5)

                ax.add_patch(wall_polygon)

        # Draw right wall segments
        if self.sim.right_wall_segments:
            for segment in self.sim.right_wall_segments:
                # Create a polygon for the wall segment
                wall_polygon = Polygon([
                    [segment['top_x'], segment['top_y']],
                    [segment['bottom_x'], segment['bottom_y']],
                    [segment['bottom_x'] - segment['width'], segment['bottom_y']],
                    [segment['top_x'] - segment['width'], segment['top_y']]
                ], closed=True, fill=True, facecolor='gray', alpha=0.5, edgecolor='black', linewidth=1.5)

                ax.add_patch(wall_polygon)

    def plot_snapshots(self, times=None, rows=1, cols=None, figsize=(15, 10)):
        """
        Plot snapshots of the simulation at specific times

        Parameters:
        -----------
        times : list of float
            List of times to plot. If None, evenly spaced times are chosen.
        rows, cols : int
            Number of rows and columns in the plot grid
        figsize : tuple
            Figure size in inches

        Returns:
        --------
        Figure object
        """
        if not self.sim.snapshots:
            print("No snapshots available to plot.")
            return None

        all_times = [s['time'] for s in self.sim.snapshots]

        # If no specific times are provided, choose evenly spaced times
        if times is None:
            if not cols:
                cols = 4
            num_plots = rows * cols
            indices = np.linspace(0, len(all_times)-1, num_plots, dtype=int)
            times = [all_times[i] for i in indices]
        else:
            # Find closest snapshots to requested times
            times = [all_times[np.argmin(np.abs(np.array(all_times) - t))] for t in times]
            if not cols:
                cols = len(times)

        # Create figure and axes
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Plot each snapshot
        for i, time in enumerate(times):
            if i >= len(axes):
                break

            # Find the snapshot closest to this time
            idx = np.argmin(np.abs(np.array(all_times) - time))
            snapshot = self.sim.snapshots[idx]

            ax = axes[i]
            ax.set_xlim(0, snapshot['Lx'])
            ax.set_ylim(0, snapshot['Ly'])
            ax.set_aspect('equal')

            # Draw particles
            positions = snapshot['positions']
            radii = snapshot.get('radii', np.ones(len(positions)) * self.sim.particle_radius)

            for j in range(len(positions)):
                # Color based on position
                color_val = 0.7 + 0.3 * (positions[j, 1] / snapshot['Ly'])
                circle = plt.Circle((positions[j, 0], positions[j, 1]),
                                  radius=radii[j],
                                  facecolor=(0.9, color_val * 0.8, 0.4),
                                  edgecolor='#996633',
                                  alpha=0.9)
                ax.add_patch(circle)

            # Draw box and walls
            rect = plt.Rectangle((0, 0), snapshot['Lx'], snapshot['Ly'],
                                fill=False, edgecolor='black', linewidth=1)
            ax.add_patch(rect)

            # Draw walls
            self.draw_walls(ax)

            # Add title with actual snapshot time
            ax.set_title(f"Time: {snapshot['time']:.2f}")

            # Minimal axes
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide any unused axes
        for i in range(len(times), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        return fig

    def plot_flow_rate(self, time_window=1.0, y_threshold=None, figsize=(10, 6)):
        """
        Plot the flow rate of particles through a horizontal line

        Parameters:
        -----------
        time_window : float
            Time window over which to calculate the flow rate
        y_threshold : float
            Y-coordinate of the horizontal line (defaults to the neck position)
        figsize : tuple
            Figure size in inches

        Returns:
        --------
        Figure object
        """
        # Calculate flow rate data
        flow_data = self.sim.calculate_flow_rate(time_window, y_threshold)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot cumulative count
        ax1.plot(flow_data["cumulative_times"], flow_data["cumulative_counts"], 'b-')
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Cumulative Particle Count")
        ax1.set_title("Particles Passed Through Neck")
        ax1.grid(True)

        # Plot flow rate
        if flow_data["times"]:
            ax2.plot(flow_data["times"], flow_data["flow_rates"], 'r-')
            ax2.set_xlabel("Time")
            ax2.set_ylabel(f"Flow Rate (particles/time)")
            ax2.set_title(f"Flow Rate (window = {time_window})")
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "Insufficient data for flow rate calculation",
                    ha='center', va='center', transform=ax2.transAxes)

        plt.tight_layout()
        return fig

    def plot_force_statistics(self, figsize=(10, 6)):
        """
        Plot statistics about inter-particle forces

        Parameters:
        -----------
        figsize : tuple
            Figure size in inches

        Returns:
        --------
        Figure object
        """
        # Calculate force statistics
        force_stats = self.sim.calculate_force_statistics()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot force data if available
        if force_stats["times"]:
            ax.plot(force_stats["times"], force_stats["max_forces"], 'r-',
                   label="Maximum Force")
            ax.plot(force_stats["times"], force_stats["mean_forces"], 'b-',
                   label="Mean Force")
            ax.set_xlabel("Time")
            ax.set_ylabel("Force Magnitude")
            ax.set_title("Inter-particle Forces vs Time")
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No force data available",
                  ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        return fig
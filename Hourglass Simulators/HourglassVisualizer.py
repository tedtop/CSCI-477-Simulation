import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class HourglassVisualizer:
    def __init__(self, simulator):
        self.sim = simulator

    def plot_positions(self, time=None):
        """Plot the particle positions with their radii at a given simulation time."""
        if not self.sim.snapshots:
            raise ValueError("No snapshots available to plot positions.")

        if time is None:
            # Use the latest snapshot
            snapshot = self.sim.snapshots[-1]
        else:
            times = np.array([s['time'] for s in self.sim.snapshots])
            idx = np.argmin(np.abs(times - time))
            snapshot = self.sim.snapshots[idx]

        positions = snapshot['positions']
        radii = snapshot['radii'] if 'radii' in snapshot else np.ones(len(positions)) * 0.5
        Lx = snapshot['Lx']
        Ly = snapshot['Ly']
        actual_time = snapshot['time']

        fig, ax = plt.subplots(figsize=(8, 8 * Ly/Lx))

        # Draw the walls first (so particles appear on top)
        self.draw_walls(ax, snapshot)

        # Draw particles as circles with their proper radii
        for i in range(len(positions)):
            circle = plt.Circle((positions[i, 0], positions[i, 1]),
                            radii[i], fill=True, alpha=0.6)
            ax.add_patch(circle)

        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"Particle Positions (Time {actual_time:.2f})")
        ax.grid()
        ax.set_aspect('equal')
        plt.show()

    def plot_metrics(self, start_time=0.0, end_time=None):
        """
        Plot Energy, Pressure, and Temperature vs Time between start_time and end_time.
        """
        if not self.sim.snapshots:
            raise ValueError("No snapshots available to plot metrics.")

        full_times = np.array([s['time'] for s in self.sim.snapshots])
        total_energies = np.array([s['total_energy'] for s in self.sim.snapshots])
        pressures = np.array([s['pressure'] for s in self.sim.snapshots])
        temperatures = np.array([s['temperature'] for s in self.sim.snapshots])

        if end_time is None:
            end_time = np.max(full_times)

        # Apply time mask
        mask = (full_times >= start_time) & (full_times <= end_time)

        times = full_times[mask]
        energies = total_energies[mask]
        pressures = pressures[mask]
        temperatures = temperatures[mask]

        # Plot
        plt.figure(figsize=(15,5))

        plt.subplot(1, 3, 1)
        plt.plot(times, energies, label='Energy')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy vs Time')
        plt.grid()
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(times, pressures, label='Pressure', color='orange')
        plt.xlabel('Time')
        plt.ylabel('Pressure')
        plt.title('Pressure vs Time')
        plt.grid()
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(times, temperatures, label='Temperature', color='green')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.title('Temperature vs Time')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()


    def animate_positions(self, interval=50):
        """
        Animate the particle positions over time.
        interval: Delay between frames in milliseconds.
        """
        if not self.sim.snapshots:
            raise ValueError("No snapshots available to animate.")

        # Extract all position histories and times
        positions_history = [s['positions'] for s in self.sim.snapshots]
        time_points = [s['time'] for s in self.sim.snapshots]
        box_sizes = [(s['Lx'], s['Ly']) for s in self.sim.snapshots]

        # Create the figure
        fig, ax = plt.subplots(figsize=(6,6))

        # Draw the walls (won't change during animation)
        self.draw_walls(ax)

        # Initialize particle scatter
        particles = ax.scatter([], [], s=20)

        # Initialize text
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        count_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

        # Set axis limits based on first frame
        initial_Lx, initial_Ly = box_sizes[0]
        ax.set_xlim(0, initial_Lx)
        ax.set_ylim(0, initial_Ly)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Particle Animation Over Time')
        ax.grid()
        ax.set_aspect('equal')

        # --- Animation functions ---

        def init():
            particles.set_offsets(np.empty((0,2)))
            time_text.set_text('')
            count_text.set_text('')
            return [particles, time_text, count_text]

        def animate(i):
            pos = positions_history[i]
            Lx, Ly = box_sizes[i]

            particles.set_offsets(pos)

            # Update limits if the box size changed
            ax.set_xlim(0, Lx)
            ax.set_ylim(0, Ly)

            # Update the time counter
            time_text.set_text(f'Time: {time_points[i]:.2f}')

            return [particles, time_text, count_text]

        # Create the animation
        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=len(positions_history), interval=interval, blit=True
        )

        plt.close(fig)  # Avoid double showing
        return HTML(anim.to_jshtml())

    def draw_walls(self, ax, snapshot=None):
        """
        Draw the simulation walls including the sloped left and right walls.

        Parameters:
        -----------
        ax : matplotlib axis
            The axis on which to draw the walls
        snapshot : dict, optional
            Simulation snapshot data; if None, uses the simulator's current state
        """
        # Get wall parameters either from snapshot or directly from simulator
        if snapshot:
            Lx = snapshot['Lx']
            Ly = snapshot['Ly']
            # If the snapshot contains wall info (might need to be added to snapshots)
            left_wall_top_x = self.sim.left_wall_top_x if hasattr(self.sim, 'left_wall_top_x') else None
            left_wall_bottom_x = self.sim.left_wall_bottom_x if hasattr(self.sim, 'left_wall_bottom_x') else None
            left_wall_width = self.sim.left_wall_width if hasattr(self.sim, 'left_wall_width') else None

            right_wall_top_x = self.sim.right_wall_top_x if hasattr(self.sim, 'right_wall_top_x') else None
            right_wall_bottom_x = self.sim.right_wall_bottom_x if hasattr(self.sim, 'right_wall_bottom_x') else None
            right_wall_width = self.sim.right_wall_width if hasattr(self.sim, 'right_wall_width') else None
        else:
            Lx = self.sim.Lx
            Ly = self.sim.Ly
            left_wall_top_x = self.sim.left_wall_top_x if hasattr(self.sim, 'left_wall_top_x') else None
            left_wall_bottom_x = self.sim.left_wall_bottom_x if hasattr(self.sim, 'left_wall_bottom_x') else None
            left_wall_width = self.sim.left_wall_width if hasattr(self.sim, 'left_wall_width') else None

            right_wall_top_x = self.sim.right_wall_top_x if hasattr(self.sim, 'right_wall_top_x') else None
            right_wall_bottom_x = self.sim.right_wall_bottom_x if hasattr(self.sim, 'right_wall_bottom_x') else None
            right_wall_width = self.sim.right_wall_width if hasattr(self.sim, 'right_wall_width') else None

        # Draw the bottom wall (standard box edge)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=2)  # Bottom wall

        # Draw the diagonal left wall if parameters are available
        if left_wall_top_x is not None and left_wall_bottom_x is not None:
            # Draw the outer edge of the wall
            ax.plot([left_wall_top_x, left_wall_bottom_x],
                   [Ly, 0], 'k-', linewidth=2)

            # Draw the inner edge of the wall if width is defined
            if left_wall_width is not None:
                ax.plot([left_wall_top_x + left_wall_width, left_wall_bottom_x + left_wall_width],
                       [Ly, 0], 'k-', linewidth=2)

                # Fill the wall area with a semi-transparent color
                import matplotlib.patches as patches
                wall_polygon = patches.Polygon([
                    [left_wall_top_x, Ly],
                    [left_wall_bottom_x, 0],
                    [left_wall_bottom_x + left_wall_width, 0],
                    [left_wall_top_x + left_wall_width, Ly]
                ], closed=True, fill=True, color='gray', alpha=0.5)
                ax.add_patch(wall_polygon)

        # Draw the diagonal right wall if parameters are available
        if right_wall_top_x is not None and right_wall_bottom_x is not None:
            # Draw the outer edge of the wall
            ax.plot([right_wall_top_x, right_wall_bottom_x],
                   [Ly, 0], 'k-', linewidth=2)

            # Draw the inner edge of the wall if width is defined
            if right_wall_width is not None:
                ax.plot([right_wall_top_x - right_wall_width, right_wall_bottom_x - right_wall_width],
                       [Ly, 0], 'k-', linewidth=2)

                # Fill the wall area with a semi-transparent color
                import matplotlib.patches as patches
                wall_polygon = patches.Polygon([
                    [right_wall_top_x, Ly],
                    [right_wall_bottom_x, 0],
                    [right_wall_bottom_x - right_wall_width, 0],
                    [right_wall_top_x - right_wall_width, Ly]
                ], closed=True, fill=True, color='gray', alpha=0.5)
                ax.add_patch(wall_polygon)
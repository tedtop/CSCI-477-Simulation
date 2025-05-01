import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class LJVisualizer:
    def __init__(self, simulator):
        self.sim = simulator

    def plot_positions(self, time=None):
        """
        Plot the particle positions at a given simulation time.
        If time is None, plot the latest available positions.
        """
        if not self.sim.snapshots:
            raise ValueError("No snapshots available to plot positions.")

        if time is None:
            # Use the latest snapshot
            snapshot = self.sim.snapshots[-1]
        else:
            times = np.array([s['time'] for s in self.sim.snapshots])
            idx = np.argmin(np.abs(times - time))
            snapshot = self.sim.snapshots[idx]

        # Debug snapshot timestamps
        # print(times)

        positions = snapshot['positions']
        Lx = snapshot['Lx']
        Ly = snapshot['Ly']
        actual_time = snapshot['time']

        plt.figure(figsize=(6,6))
        plt.scatter(positions[:,0], positions[:,1], s=20)
        plt.xlim(0, Lx)
        plt.ylim(0, Ly)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"Particle Positions (Time {actual_time:.2f}, Box Size {Lx:.2f}x{Ly:.2f})")
        plt.grid()
        plt.gca().set_aspect('equal')
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
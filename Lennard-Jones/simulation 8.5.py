import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Import the LJSimulator class
from LJSimulator import LJSimulator

# Set simulation parameters
N = 64              # Number of particles
Lx = Ly = 10.0      # Box dimensions
temperature = 1.0   # Initial temperature
dt = 0.01           # Time step
num_steps = 500     # Number of simulation steps
anim_interval = 5   # Save position every N steps for animation

# Create the simulation
sim = LJSimulator(N=N, Lx=Lx, Ly=Ly, temperature=temperature, dt=dt,
                  lattice_type='square', snapshot_interval=anim_interval)

# Run the simulation
for _ in range(num_steps):
    sim.step()

# Create animation from the snapshots
def create_animation():
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Lennard-Jones Particles Simulation (N={N}, L={Lx})')
    ax.grid(True)

    # Create scatter plot for particles
    particles = ax.scatter([], [], s=30)

    # Initialize function
    def init():
        particles.set_offsets(np.c_[[], []])
        return [particles]

    # Animation function
    def animate(i):
        snapshot = sim.snapshots[i]
        positions = snapshot['positions']
        particles.set_offsets(positions)
        return [particles]

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(sim.snapshots), interval=50, blit=True
    )

    return anim, fig

# Create and show the animation
anim, fig = create_animation()

# Plot the energy, temperature, and pressure evolution
def plot_results():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot energies
    ax1.plot(sim.time_history, sim.kinetic_energy_history, label='Kinetic Energy')
    ax1.plot(sim.time_history, sim.potential_energy_history, label='Potential Energy')
    ax1.plot(sim.time_history, sim.total_energy_history, label='Total Energy')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Energy')
    ax1.legend()
    ax1.grid(True)

    # Plot temperature
    ax2.plot(sim.time_history, sim.temperature_history)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Temperature')
    ax2.grid(True)

    # Plot pressure
    ax3.plot(sim.time_history, sim.pressure_history)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Pressure')
    ax3.grid(True)

    plt.tight_layout()
    return fig

# Create and show the plots
energy_fig = plot_results()

# Uncomment the following if running in a Jupyter notebook
# display(HTML(anim.to_jshtml()))
# display(energy_fig)

# Uncomment if running in a standard Python script (not Jupyter)
plt.show()
"""
Perpetual Hourglass Simulation - Showing Particle Respawning Feature

This example demonstrates how particles respawn when they hit the bottom
of the hourglass, creating a continuous flow of particles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
from IPython.display import HTML

# Import the simulator
from RespawningHourglassSimulator import RespawningHourglassSimulator

# Configure the simulation parameters for clear respawning behavior
simulator = RespawningHourglassSimulator(
    N=150,                  # Fewer particles to better see individual behavior
    Lx=20.0,                # Box width
    Ly=20.0,                # Box height
    dt=0.005,               # Small time step for stability
    gravity=5.0,            # Standard gravity
    particle_radius=0.5,    # Visible particle size
    k=25.0,                 # Medium-stiff spring constant
    gamma=2.5,              # Higher damping for less bouncing
    friction_coef=0.6,      # Medium-high friction for sand-like behavior
    restitution_coef=0.2,   # Low restitution (less bouncy)
    neck_width=3.0,         # Moderately narrow neck
    respawn_particles=True  # Enable respawning!
)

# Create the hourglass shape
simulator.draw_hourglass()

# Initialize particles at the top
simulator.initialize_random_falling_particles()

# Set up figure and axes
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3)

# Main simulation display
ax = fig.add_subplot(gs[:, 0:2])
ax.set_xlim(0, simulator.Lx)
ax.set_ylim(0, simulator.Ly)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Perpetual Hourglass with Particle Respawning")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

# Energy plot
ax_energy = fig.add_subplot(gs[0, 2])
ax_energy.set_title("Energy vs Time")
ax_energy.set_xlabel("Time")
ax_energy.set_ylabel("Energy")
ax_energy.grid(True)

# Temperature plot
ax_temp = fig.add_subplot(gs[1, 2])
ax_temp.set_title("Kinetic Temperature vs Time")
ax_temp.set_xlabel("Time")
ax_temp.set_ylabel("Temperature")
ax_temp.grid(True)

# Flow plot
ax_flow = fig.add_subplot(gs[2, 2])
ax_flow.set_title("Particles Through Neck vs Time")
ax_flow.set_xlabel("Time")
ax_flow.set_ylabel("Particle Count")
ax_flow.grid(True)

# Create particles as circles
circles = []
for i in range(simulator.N):
    circle = Circle((simulator.x[i], simulator.y[i]),
                   radius=simulator.r[i],
                   facecolor='orange',
                   edgecolor='brown',
                   alpha=0.9)
    ax.add_patch(circle)
    circles.append(circle)

# Add box boundaries
box = plt.Rectangle((0, 0), simulator.Lx, simulator.Ly, fill=False,
                   edgecolor='black', linewidth=2)
ax.add_patch(box)

# Add hourglass walls
for segment in simulator.left_wall_segments:
    wall = Polygon([
        [segment['top_x'], segment['top_y']],
        [segment['bottom_x'], segment['bottom_y']],
        [segment['bottom_x'] + segment['width'], segment['bottom_y']],
        [segment['top_x'] + segment['width'], segment['top_y']]
    ], closed=True, fill=True, facecolor='gray', alpha=0.5, edgecolor='black', linewidth=1.5)
    ax.add_patch(wall)

for segment in simulator.right_wall_segments:
    wall = Polygon([
        [segment['top_x'], segment['top_y']],
        [segment['bottom_x'], segment['bottom_y']],
        [segment['bottom_x'] - segment['width'], segment['bottom_y']],
        [segment['top_x'] - segment['width'], segment['top_y']]
    ], closed=True, fill=True, facecolor='gray', alpha=0.5, edgecolor='black', linewidth=1.5)
    ax.add_patch(wall)

# Add text for information
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
respawn_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12)
flow_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=12)

# Tracker for particles passing through neck
particles_passed = set()
neck_y = simulator.Ly / 2
neck_width = simulator.neck_width
neck_x_min = (simulator.Lx - neck_width) / 2
neck_x_max = neck_x_min + neck_width

# Highlight recently respawned particles
highlight_duration = 10  # frames to highlight particles after respawning
particle_highlights = np.zeros(simulator.N, dtype=int)  # countdown for highlighting

# Initialize data arrays for plots
time_data = []
energy_data = []
kinetic_data = []
potential_data = []
temperature_data = []
particles_through_neck_data = []

# Set up the limits for the plots
ax_energy.set_xlim(0, 500 * 5 * simulator.dt)  # 500 frames * 5 steps per frame * dt
ax_energy.set_ylim(0, 1.0)  # Start with a reasonable range, will auto-adjust
ax_temp.set_xlim(0, 500 * 5 * simulator.dt)
ax_temp.set_ylim(0, 1.0)  # Start with a reasonable range, will auto-adjust
ax_flow.set_xlim(0, 500 * 5 * simulator.dt)
ax_flow.set_ylim(0, simulator.N)

# Create plot lines with empty initial data
energy_line, = ax_energy.plot([], [], 'k-', label='Total')
kinetic_line, = ax_energy.plot([], [], 'r-', label='Kinetic')
potential_line, = ax_energy.plot([], [], 'b-', label='Potential')
ax_energy.legend(loc='upper right')

temperature_line, = ax_temp.plot([], [], 'g-')
flow_line, = ax_flow.plot([], [], 'm-', linewidth=2)

# Animation update function
def update(frame):
    # Steps per frame
    steps_per_frame = 5

    # Run multiple simulation steps per frame for speed
    for _ in range(steps_per_frame):
        # Store previous positions to detect respawns
        prev_y = simulator.y.copy()

        # Step the simulation
        simulator.step()

        # Detect respawns (particles that move from bottom to top)
        for i in range(simulator.N):
            if prev_y[i] < simulator.r[i] and simulator.y[i] > 0.8 * simulator.Ly:
                particle_highlights[i] = highlight_duration

    # Collect energy data
    time_data.append(simulator.time)

    # Calculate energy values - assuming unit mass (m=1) for all particles
    # The RespawningHourglassSimulator uses unit mass internally
    kinetic_energy = 0.5 * np.sum(simulator.vx**2 + simulator.vy**2)
    potential_energy = np.sum(simulator.gravity * simulator.y)
    total_energy = kinetic_energy + potential_energy

    energy_data.append(total_energy)
    kinetic_data.append(kinetic_energy)
    potential_data.append(potential_energy)

    # Calculate temperature (proportional to average kinetic energy)
    # Temperature is 2x average KE per particle in 2D
    temperature = np.sum(simulator.vx**2 + simulator.vy**2) / simulator.N
    temperature_data.append(temperature)

    # Update particle positions
    for i, circle in enumerate(circles):
        circle.center = (simulator.x[i], simulator.y[i])

        # Color particles based on their state
        if particle_highlights[i] > 0:
            # Recently respawned - highlight in bright yellow
            circle.set_facecolor('yellow')
            particle_highlights[i] -= 1
        else:
            # Normal particles - color gradient based on height
            color_val = 0.5 + 0.5 * (simulator.y[i] / simulator.Ly)
            circle.set_facecolor((1.0, color_val * 0.8, 0.0))  # Orange gradient

    # Track particles passing through neck
    for i in range(simulator.N):
        if (simulator.y[i] < neck_y and
            i not in particles_passed and
            neck_x_min < simulator.x[i] < neck_x_max):
            particles_passed.add(i)

    # Add particle count to data
    particles_through_neck_data.append(len(particles_passed))

    # Update info text
    time_text.set_text(f'Time: {simulator.time:.2f}')

    stats = simulator.get_respawn_stats()
    respawn_text.set_text(f'Total respawns: {stats["total_respawns"]}')
    flow_text.set_text(f'Particles passed through neck: {len(particles_passed)}')

    # Auto-adjust y-limits for temperature if needed
    if temperature_data and max(temperature_data) > ax_temp.get_ylim()[1]:
        ax_temp.set_ylim(0, max(temperature_data) * 1.2)

    # Auto-adjust y-limits for energy if needed
    if energy_data and max(energy_data) > ax_energy.get_ylim()[1]:
        ax_energy.set_ylim(0, max(energy_data) * 1.2)

    # Update the plot lines with new data
    energy_line.set_data(time_data, energy_data)
    kinetic_line.set_data(time_data, kinetic_data)
    potential_line.set_data(time_data, potential_data)
    temperature_line.set_data(time_data, temperature_data)
    flow_line.set_data(time_data, particles_through_neck_data)

    # Return all updated artists
    return circles + [time_text, respawn_text, flow_text, energy_line,
                      kinetic_line, potential_line, temperature_line, flow_line]

# Create animation
ani = FuncAnimation(fig, update, frames=500, interval=30, blit=True)

# Display the animation
plt.tight_layout()
plt.show()

# Uncomment to save animation
# ani.save('perpetual_hourglass.mp4', writer='ffmpeg', fps=30)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from IPython.display import HTML
import time

from HourglassSimulator import HourglassSimulator

# Create simulation with parameters optimized for visible falling
simulator = HourglassSimulator(
    N=64,                   # Number of particles
    Lx=20.0,                # Box width
    Ly=20.0,                # Box height
    temperature=0.0,        # No initial random motion
    dt=0.01,               # Time step
    gravity=5.0,            # Strong gravity to overcome particle interactions
    particle_radius=0.5,    # Smaller radius to reduce overlapping
    k=0.1,                  # Weaker spring constant
    gamma=0.05,             # Lower damping
    snapshot_interval=1     # Every step
)

# Animation timing parameters
reference_dt = 0.01         # Reference timestep for animation timing
time_per_frame = 0.05       # Simulation time to advance per animation frame (5 * reference_dt)
steps_per_frame = max(1, int(round(time_per_frame / simulator.dt)))  # Adjusted dynamically based on dt

print(f"Using timestep dt={simulator.dt}, running {steps_per_frame} simulation steps per animation frame")

# Add a diagonal sloped left wall to create an hourglass shape
simulator.add_left_wall(top_x=1.0, bottom_x=6.0, wall_width=0.5)

# Add a diagonal sloped right wall to complete the hourglass shape
simulator.add_right_wall(top_x=19.0, bottom_x=14.0, wall_width=0.5)

# Initialize particles to flow in from the top
simulator.initialize_particles()

# Debug information
print(f"Number of particles: {simulator.N}")
print(f"Particle radius: {simulator.r[0]}")
print(f"Box dimensions: {simulator.Lx} x {simulator.Ly}")
print(f"Particle positions:")
print(f"  x range: {np.min(simulator.x):.2f} to {np.max(simulator.x):.2f}")
print(f"  y range: {np.min(simulator.y):.2f} to {np.max(simulator.y):.2f}")

# Create visualization with side-by-side plots
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 3)

# Main simulation display
ax_sim = fig.add_subplot(gs[:, 0:2])
ax_sim.set_xlim(0, simulator.Lx)
ax_sim.set_ylim(0, simulator.Ly)
ax_sim.set_aspect('equal')  # Maintain equal scaling
ax_sim.grid(True)
ax_sim.set_title("Granular Particles Falling Under Gravity")
ax_sim.set_xlabel("X position")
ax_sim.set_ylabel("Y position")

# Energy plot
ax_energy = fig.add_subplot(gs[0, 2])
ax_energy.set_title("Energy vs Time")
ax_energy.set_xlabel("Time")
ax_energy.set_ylabel("Energy")
ax_energy.grid(True)

# Velocity plot
ax_velocity = fig.add_subplot(gs[1, 2])
ax_velocity.set_title("Velocity vs Time")
ax_velocity.set_xlabel("Time")
ax_velocity.set_ylabel("Average Velocity")
ax_velocity.grid(True)

# When creating circles
circles = []
for i in range(simulator.N):
    # Use a sand-like color gradient based on position
    color_val = 0.7 + 0.3 * (simulator.y[i] / simulator.Ly)  # Lighter at top, darker at bottom
    circle = Circle((simulator.x[i], simulator.y[i]),
                   radius=simulator.r[i],
                   facecolor=(0.9, color_val * 0.8, 0.4),  # Sandy color
                   edgecolor='#996633',  # Brown edge
                   alpha=0.9)
    ax_sim.add_patch(circle)
    circles.append(circle)

# Add box boundaries
box = plt.Rectangle((0, 0), simulator.Lx, simulator.Ly, fill=False,
                    edgecolor='black', linewidth=2)
ax_sim.add_patch(box)

# Add the sloped left wall visualization
if hasattr(simulator, 'left_wall_top_x') and hasattr(simulator, 'left_wall_bottom_x'):
    from matplotlib.patches import Polygon

    # Create a polygon for the sloped wall
    wall_polygon = Polygon([
        [simulator.left_wall_top_x, simulator.Ly],
        [simulator.left_wall_bottom_x, 0],
        [simulator.left_wall_bottom_x + simulator.left_wall_width, 0],
        [simulator.left_wall_top_x + simulator.left_wall_width, simulator.Ly]
    ], closed=True, fill=True, color='gray', alpha=0.5, edgecolor='black', linewidth=1.5)

    ax_sim.add_patch(wall_polygon)

# Add the sloped right wall visualization
if hasattr(simulator, 'right_wall_top_x') and hasattr(simulator, 'right_wall_bottom_x'):
    from matplotlib.patches import Polygon

    # Create a polygon for the sloped wall
    wall_polygon = Polygon([
        [simulator.right_wall_top_x, simulator.Ly],
        [simulator.right_wall_bottom_x, 0],
        [simulator.right_wall_bottom_x + simulator.right_wall_width, 0],
        [simulator.right_wall_top_x + simulator.right_wall_width, simulator.Ly]
    ], closed=True, fill=True, color='gray', alpha=0.5, edgecolor='black', linewidth=1.5)

    ax_sim.add_patch(wall_polygon)

# Add text for time and debug info
time_text = ax_sim.text(0.02, 0.98, '', transform=ax_sim.transAxes, fontsize=10)
debug_text = ax_sim.text(0.02, 0.94, '', transform=ax_sim.transAxes, fontsize=10)

# Initialize data arrays for plots
time_data = []
energy_data = []
kinetic_data = []
potential_data = []
velocity_data = []

# Create plot lines
energy_line, = ax_energy.plot([], [], 'k-', label='Total')
kinetic_line, = ax_energy.plot([], [], 'r-', label='Kinetic')
potential_line, = ax_energy.plot([], [], 'b-', label='Potential')
ax_energy.legend(loc='upper right')

velocity_line, = ax_velocity.plot([], [], 'g-')

# Animation update function
def update(frame):
    global time_data, energy_data, kinetic_data, potential_data, velocity_data

    # Run multiple steps per frame to speed up animation
    for _ in range(steps_per_frame):  # Adjusted simulation steps per frame
        simulator.step()

    # Collect data for plots
    time_data.append(simulator.time)
    energy_data.append(simulator.total_energy)
    kinetic_data.append(simulator.kinetic_energy)
    potential_data.append(simulator.potential_energy)

    # Calculate average particle velocity
    avg_velocity = np.sqrt(np.mean(simulator.vx**2 + simulator.vy**2))
    velocity_data.append(avg_velocity)


    # Update particle positions
    for i, circle in enumerate(circles):
        circle.center = (simulator.x[i], simulator.y[i])

    # Update info text
    time_text.set_text(f'Time: {simulator.time:.2f}')

    # Display debugging info
    max_ay = np.max(np.abs(simulator.ay))
    avg_vy = np.mean(np.abs(simulator.vy))
    debug_text.set_text(f'Max Accel: {max_ay:.2f}, Avg |Vy|: {avg_vy:.2f}')

    # Update plot data
    energy_line.set_data(time_data, energy_data)
    kinetic_line.set_data(time_data, kinetic_data)
    potential_line.set_data(time_data, potential_data)
    velocity_line.set_data(time_data, velocity_data)

    # Adjust plot ranges
    if len(time_data) > 1:
        for ax in [ax_energy, ax_velocity]:
            ax.set_xlim(time_data[0], time_data[-1])
            ax.relim()
            ax.autoscale_view(scaley=True)

    # Return all artists that were updated
    return circles + [time_text, debug_text,
                     energy_line, kinetic_line, potential_line,
                     velocity_line]

# Create animation
ani = FuncAnimation(fig, update, frames=500, interval=20, blit=True)

# Display animation
plt.tight_layout()
plt.show()
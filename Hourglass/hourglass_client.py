import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from IPython.display import HTML
import time

from HourglassSimulator import HourglassSimulator

# Create simulation with parameters optimized for visible falling
simulator = HourglassSimulator(
    N=200,                  # Number of particles
    Lx=20.0,                # Box width
    Ly=20.0,                # Box height
    temperature=0.0,        # No initial random motion
    dt=0.01,                # Time step
    gravity=5.0,            # Strong gravity to overcome particle interactions
    particle_radius=0.5,    # Smaller radius to reduce overlapping
    k=0.1,                  # Weaker spring constant
    gamma=0.05,             # Lower damping
    neck_width=2.5,         # Width of the neck in the hourglass
    wall_width=0.5,         # Width of the walls
    )

# Animation timing parameters
reference_dt = 0.01         # Reference timestep for animation timing
time_per_frame = 0.05       # Simulation time to advance per animation frame (5 * reference_dt)
steps_per_frame = max(1, int(round(time_per_frame / simulator.dt)))  # Adjusted dynamically based on dt

print(f"Using timestep dt={simulator.dt}, running {steps_per_frame} simulation steps per animation frame")

# Create an hourglass shape with a narrow neck in the middle
simulator.draw_hourglass()

# Initialize particles to randomly fall in from the top
simulator.initialize_random_falling_particles()

# Debug information
print(f"Number of particles: {simulator.N}")
print(f"Particle radius: {simulator.r[0]}")
print(f"Box dimensions: {simulator.Lx} x {simulator.Ly}")
print(f"Particle positions:")
print(f"  x range: {np.min(simulator.x):.2f} to {np.max(simulator.x):.2f}")
print(f"  y range: {np.min(simulator.y):.2f} to {np.max(simulator.y):.2f}")

# Change the figure layout and add a new plot for particle flow
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3)

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
ax_velocity.set_title("Kinetic Temperature vs Time")
ax_velocity.set_xlabel("Time")
ax_velocity.set_ylabel("Mean Kinetic Energy")
ax_velocity.grid(True)

# New plot for particle flow
ax_flow = fig.add_subplot(gs[2, 2])
ax_flow.set_title("Particles Through Neck vs Time")
ax_flow.set_xlabel("Time")
ax_flow.set_ylabel("Particle Count")
ax_flow.grid(True)

# Fix axes with appropriate static limits based on actual data
ax_energy.set_xlim(0, 15)        # Fixed time range: 0-15 seconds
ax_energy.set_ylim(0, 3500)      # Energy range: 0-3500 units (based on max ~3140)
ax_velocity.set_xlim(0, 15)      # Same time range for both plots
ax_velocity.set_ylim(0, 35)      # Temperature range: 0-35 units (based on max ~31.39)
ax_flow.set_xlim(0, 15)          # Same time range
ax_flow.set_ylim(0, simulator.N) # Max possible number of particles

# Turn off debug printing now that we have the proper scales
DEBUG_ENERGY_RANGES = False

# Regular grid lines
ax_energy.grid(True)
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
    ], closed=True, fill=True, facecolor='gray', alpha=0.5, edgecolor='black', linewidth=1.5)

    ax_sim.add_patch(wall_polygon)

# Add the sloped right wall visualization
if hasattr(simulator, 'right_wall_top_x') and hasattr(simulator, 'right_wall_bottom_x'):
    from matplotlib.patches import Polygon

    # Create a polygon for the sloped wall
    wall_polygon = Polygon([
        [simulator.right_wall_top_x, simulator.Ly],
        [simulator.right_wall_bottom_x, 0],
        [simulator.right_wall_bottom_x - simulator.right_wall_width, 0],
        [simulator.right_wall_top_x - simulator.right_wall_width, simulator.Ly]
    ], closed=True, fill=True, facecolor='gray', alpha=0.5, edgecolor='black', linewidth=1.5)

    ax_sim.add_patch(wall_polygon)

# Add the wall visualizations
if hasattr(simulator, 'left_wall_segments') and simulator.left_wall_segments:
    from matplotlib.patches import Polygon

    # Draw each left wall segment
    for segment in simulator.left_wall_segments:
        # Create a polygon for the wall segment
        wall_polygon = Polygon([
            [segment['top_x'], segment['top_y']],
            [segment['bottom_x'], segment['bottom_y']],
            [segment['bottom_x'] + segment['width'], segment['bottom_y']],
            [segment['top_x'] + segment['width'], segment['top_y']]
        ], closed=True, fill=True, facecolor='gray', alpha=0.5, edgecolor='black', linewidth=1.5)

        ax_sim.add_patch(wall_polygon)

if hasattr(simulator, 'right_wall_segments') and simulator.right_wall_segments:
    from matplotlib.patches import Polygon

    # Draw each right wall segment
    for segment in simulator.right_wall_segments:
        # Create a polygon for the wall segment
        wall_polygon = Polygon([
            [segment['top_x'], segment['top_y']],
            [segment['bottom_x'], segment['bottom_y']],
            [segment['bottom_x'] - segment['width'], segment['bottom_y']],
            [segment['top_x'] - segment['width'], segment['top_y']]
        ], closed=True, fill=True, facecolor='gray', alpha=0.5, edgecolor='black', linewidth=1.5)

        ax_sim.add_patch(wall_polygon)

# Add text for time and debug info
time_text = ax_sim.text(0.02, 0.96, '', transform=ax_sim.transAxes, fontsize=10)
debug_text = ax_sim.text(0.02, 0.93, '', transform=ax_sim.transAxes, fontsize=10)

# Initialize data arrays for plots
time_data = []
energy_data = []
kinetic_data = []
potential_data = []
kinetic_temp_data = []
particles_through_neck_data = []  # Track particles that have crossed the midpoint

# Initialize a set to track which particles have passed through the neck
particles_passed_through = set()

# Create plot lines with empty initial data
energy_line, = ax_energy.plot([], [], 'k-', label='Total')
kinetic_line, = ax_energy.plot([], [], 'r-', label='Kinetic')
potential_line, = ax_energy.plot([], [], 'b-', label='Potential')
ax_energy.legend(loc='upper right')

kinetic_temp_line, = ax_velocity.plot([], [], 'g-')

# Add plot line for particle flow
flow_line, = ax_flow.plot([], [], 'm-', linewidth=2)

# Add a horizontal line at the hourglass neck position to visualize the counting boundary
neck_y = simulator.Ly / 2  # The neck is at the middle height

# Add text display for particles passed count
flow_text = ax_sim.text(0.02, 0.90, '', transform=ax_sim.transAxes, fontsize=10)

# Simple animation update function with no dynamic scaling
def update(frame):
    global time_data, energy_data, kinetic_data, potential_data, kinetic_temp_data, particles_through_neck_data, particles_passed_through

    # Run simulation steps
    for _ in range(steps_per_frame):
        simulator.step()

    # Collect data
    current_time = simulator.time
    time_data.append(current_time)
    energy_data.append(simulator.total_energy)
    kinetic_data.append(simulator.kinetic_energy)
    potential_data.append(simulator.potential_energy)
    kinetic_temp_data.append(simulator.kinetic_energy / simulator.N)

    # Track particles passing through the hourglass neck (middle height)
    neck_y = simulator.Ly / 2
    neck_width = simulator.neck_width  # Use the simulator's neck_width parameter
    neck_x_min = (simulator.Lx - neck_width) / 2
    neck_x_max = neck_x_min + neck_width

    # Check each particle
    for i in range(simulator.N):
        # If particle has crossed the neck line from above to below and is within the neck width
        if (simulator.y[i] < neck_y and
            i not in particles_passed_through and
            neck_x_min < simulator.x[i] < neck_x_max):
            particles_passed_through.add(i)

    # Store the count of particles that have passed through
    particles_through_neck_data.append(len(particles_passed_through))

    # Update the flow text
    flow_text.set_text(f'Particles passed: {len(particles_passed_through)}')

    # Print max energy values every 20 frames to help determine appropriate scale ranges
    if DEBUG_ENERGY_RANGES and frame > 0 and frame % 20 == 0:
        print(f"Time: {current_time:.2f}")
        print(f"Max Total Energy: {max(energy_data):.2f}")
        print(f"Max Kinetic Energy: {max(kinetic_data):.2f}")
        print(f"Max Potential Energy: {max(potential_data):.2f}")
        print(f"Max Kinetic Temperature: {max(kinetic_temp_data):.2f}")
        print("-" * 30)

    # Update particle positions
    for i, circle in enumerate(circles):
        circle.center = (simulator.x[i], simulator.y[i])

    # Update info text
    time_text.set_text(f'Time: {current_time:.2f}')

    # Update the plot lines with new data
    energy_line.set_data(time_data, energy_data)
    kinetic_line.set_data(time_data, kinetic_data)
    potential_line.set_data(time_data, potential_data)
    kinetic_temp_line.set_data(time_data, kinetic_temp_data)
    flow_line.set_data(time_data, particles_through_neck_data)

    # Return all artists that were updated
    return circles + [time_text, flow_text, energy_line, kinetic_line, potential_line, kinetic_temp_line, flow_line]

# Create animation
ani = FuncAnimation(fig, update, frames=500, interval=20, blit=True)

# Display animation
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Import the simulator
from GranularHourglassSimulator import GranularHourglassSimulator

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run granular hourglass simulation')

    # Simulation parameters
    parser.add_argument('--particles', type=int, default=200, help='Number of particles')
    parser.add_argument('--width', type=float, default=20.0, help='Box width')
    parser.add_argument('--height', type=float, default=20.0, help='Box height')
    parser.add_argument('--dt', type=float, default=0.005, help='Time step')
    parser.add_argument('--gravity', type=float, default=5.0, help='Gravitational acceleration')
    parser.add_argument('--radius', type=float, default=0.5, help='Particle radius')
    parser.add_argument('--k', type=float, default=20.0, help='Spring constant')
    parser.add_argument('--gamma', type=float, default=2.0, help='Damping coefficient')
    parser.add_argument('--friction', type=float, default=0.5, help='Coefficient of friction')
    parser.add_argument('--restitution', type=float, default=0.3, help='Coefficient of restitution')
    parser.add_argument('--neck-width', type=float, default=2.5, help='Width of hourglass neck')

    # Simulation control
    parser.add_argument('--frames', type=int, default=500, help='Number of animation frames')
    parser.add_argument('--steps-per-frame', type=int, default=10, help='Simulation steps per frame')
    parser.add_argument('--no-animation', action='store_true', help='Run without animation')
    parser.add_argument('--save', action='store_true', help='Save animation to file')
    parser.add_argument('--output', type=str, default='hourglass_simulation.mp4', help='Output file name')

    return parser.parse_args()

def run_simulation(args):
    """Run the simulation with the specified parameters"""
    # Create the simulator with the given parameters
    simulator = GranularHourglassSimulator(
        N=args.particles,
        Lx=args.width,
        Ly=args.height,
        dt=args.dt,
        gravity=args.gravity,
        particle_radius=args.radius,
        k=args.k,
        gamma=args.gamma,
        friction_coef=args.friction,
        restitution_coef=args.restitution,
        neck_width=args.neck_width
    )

    # Create an hourglass shape
    simulator.draw_hourglass()

    # Initialize particles
    simulator.initialize_random_falling_particles()

    # Print simulation info
    print(f"Simulation parameters:")
    print(f"  Particles: {args.particles}")
    print(f"  Box: {args.width} x {args.height}")
    print(f"  Timestep: {args.dt}")
    print(f"  Gravity: {args.gravity}")
    print(f"  Spring constant: {args.k}")
    print(f"  Damping: {args.gamma}")
    print(f"  Friction: {args.friction}")
    print(f"  Restitution: {args.restitution}")
    print(f"  Neck width: {args.neck_width}")
    print(f"  Steps per frame: {args.steps_per_frame}")

    if args.no_animation:
        # Run without animation (for performance testing)
        run_simulation_without_animation(simulator, args)
    else:
        # Run with animation
        ani = create_animation(simulator, args)

        # Save animation if requested
        if args.save:
            print(f"Saving animation to {args.output}...")
            ani.save(args.output, writer='ffmpeg', fps=30)

        # Show animation
        plt.tight_layout()
        plt.show()

def run_simulation_without_animation(simulator, args):
    """Run the simulation without visualization for performance testing"""
    particles_passed_through = set()
    neck_y = simulator.Ly / 2
    neck_width = simulator.neck_width
    neck_x_min = (simulator.Lx - neck_width) / 2
    neck_x_max = neck_x_min + neck_width

    # Calculate total steps
    total_steps = args.frames * args.steps_per_frame
    progress_interval = max(1, total_steps // 10)

    print(f"Running {total_steps} simulation steps without animation...")
    start_time = time.time()

    for step in range(total_steps):
        simulator.step()

        # Track particles passing through neck
        for i in range(simulator.N):
            if (simulator.y[i] < neck_y and
                i not in particles_passed_through and
                neck_x_min < simulator.x[i] < neck_x_max):
                particles_passed_through.add(i)

        # Print progress
        if step % progress_interval == 0 or step == total_steps - 1:
            elapsed = time.time() - start_time
            progress = (step + 1) / total_steps * 100
            print(f"Progress: {progress:.1f}% - Step {step+1}/{total_steps}, time={simulator.time:.2f}s, " +
                 f"particles passed: {len(particles_passed_through)}, " +
                 f"elapsed: {elapsed:.1f}s")

    end_time = time.time()
    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds")
    print(f"Final simulation time: {simulator.time:.2f}")
    print(f"Total particles passed through: {len(particles_passed_through)}")

def create_animation(simulator, args):
    """Create an animation of the hourglass simulation"""
    # Set up the figure and axes
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3)

    # Main simulation display
    ax_sim = fig.add_subplot(gs[:, 0:2])
    ax_sim.set_xlim(0, simulator.Lx)
    ax_sim.set_ylim(0, simulator.Ly)
    ax_sim.set_aspect('equal')
    ax_sim.grid(True)
    ax_sim.set_title("Granular Hourglass Simulation")
    ax_sim.set_xlabel("X position")
    ax_sim.set_ylabel("Y position")

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

    # Set up the limits for the plots
    ax_energy.set_xlim(0, args.frames * args.steps_per_frame * simulator.dt)
    ax_energy.set_ylim(0, 1.0)  # Start with a reasonable range, will auto-adjust
    ax_temp.set_xlim(0, args.frames * args.steps_per_frame * simulator.dt)
    ax_temp.set_ylim(0, 1.0)  # Start with a reasonable range, will auto-adjust
    ax_flow.set_xlim(0, args.frames * args.steps_per_frame * simulator.dt)
    ax_flow.set_ylim(0, simulator.N)

    # Create the particles as circles
    from matplotlib.patches import Circle, Polygon
    circles = []
    for i in range(simulator.N):
        # Color based on vertical position (lighter at top)
        color_val = 0.7 + 0.3 * (simulator.y[i] / simulator.Ly)
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

    # Add hourglass walls
    for segment in simulator.left_wall_segments:
        wall = Polygon([
            [segment['top_x'], segment['top_y']],
            [segment['bottom_x'], segment['bottom_y']],
            [segment['bottom_x'] + segment['width'], segment['bottom_y']],
            [segment['top_x'] + segment['width'], segment['top_y']]
        ], closed=True, fill=True, facecolor='gray', alpha=0.5, edgecolor='black', linewidth=1.5)
        ax_sim.add_patch(wall)

    for segment in simulator.right_wall_segments:
        wall = Polygon([
            [segment['top_x'], segment['top_y']],
            [segment['bottom_x'], segment['bottom_y']],
            [segment['bottom_x'] - segment['width'], segment['bottom_y']],
            [segment['top_x'] - segment['width'], segment['top_y']]
        ], closed=True, fill=True, facecolor='gray', alpha=0.5, edgecolor='black', linewidth=1.5)
        ax_sim.add_patch(wall)

    # Add text for information
    time_text = ax_sim.text(0.02, 0.96, '', transform=ax_sim.transAxes, fontsize=10)
    flow_text = ax_sim.text(0.02, 0.93, '', transform=ax_sim.transAxes, fontsize=10)

    # Initialize data arrays for plots
    time_data = []
    energy_data = []
    kinetic_data = []
    potential_data = []
    temperature_data = []
    particles_through_neck_data = []

    # Initialize set to track particles that have passed through the neck
    particles_passed_through = set()

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

        # Run simulation steps
        for _ in range(args.steps_per_frame):
            simulator.step()

        # Collect data
        current_time = simulator.time
        time_data.append(current_time)
        energy_data.append(simulator.total_energy)
        kinetic_data.append(simulator.kinetic_energy)
        potential_data.append(simulator.potential_energy)
        temperature_data.append(simulator.temperature)

        # Track particles passing through the hourglass neck
        neck_y = simulator.Ly / 2
        neck_width = simulator.neck_width
        neck_x_min = (simulator.Lx - neck_width) / 2
        neck_x_max = neck_x_min + neck_width

        # Check each particle
        for i in range(simulator.N):
            # If particle has crossed the neck line from above to below
            if (simulator.y[i] < neck_y and
                i not in particles_passed_through and
                neck_x_min < simulator.x[i] < neck_x_max):
                particles_passed_through.add(i)

        # Store particle count
        particles_through_neck_data.append(len(particles_passed_through))

        # Update the flow text
        flow_text.set_text(f'Particles passed: {len(particles_passed_through)}')

        # Update particle positions
        for i, circle in enumerate(circles):
            circle.center = (simulator.x[i], simulator.y[i])

        # Update info text
        time_text.set_text(f'Time: {current_time:.2f}')

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

        # Return all artists that were updated
        return circles + [time_text, flow_text, energy_line, kinetic_line,
                        potential_line, temperature_line, flow_line]

    # Create animation
    ani = FuncAnimation(fig, update, frames=args.frames, interval=20, blit=True)

    # Return the animation object
    return ani

if __name__ == "__main__":
    args = parse_arguments()
    run_simulation(args)
import numpy as np
import matplotlib.pyplot as plt
from LJSimulator import LJSimulator
from IPython.display import HTML
import matplotlib.animation as animation

# Function to create an animation of particle trajectories
def animate_simulation(simulator, num_frames=200, interval=50):
    """Create an animation of particle trajectories"""
    fig, ax = plt.subplots(figsize=(8, 8))

    particles, = ax.plot([], [], 'ro', ms=10)

    # Set axis limits
    ax.set_xlim(0, simulator.Lx)
    ax.set_ylim(0, simulator.Ly)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Lennard-Jones Particle Simulation')

    # Initialize function for animation
    def init():
        particles.set_data([], [])
        return particles,

    # Update function for animation
    def update(frame):
        if frame < len(simulator.snapshots):
            snapshot = simulator.snapshots[frame]
            particles.set_data(snapshot['positions'][:, 0], snapshot['positions'][:, 1])
            ax.set_title(f'Time: {snapshot["time"]:.2f}, T: {snapshot["temperature"]:.2f}, P: {snapshot["pressure"]:.2f}')
        return particles,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=min(num_frames, len(simulator.snapshots)),
                                  init_func=init, blit=True, interval=interval)

    # Return animation HTML
    return HTML(ani.to_jshtml())

# Function to calculate mean squared displacement from equilibrium positions
def calculate_msd(simulator, reference_positions):
    """Calculate mean squared displacement from reference positions"""
    msd_values = []

    for snapshot in simulator.snapshots:
        positions = snapshot['positions']
        # Calculate squared displacements with periodic boundary conditions
        dx = np.array([simulator.pbc_separation(pos[0] - ref[0], simulator.Lx) for pos, ref in zip(positions, reference_positions)])
        dy = np.array([simulator.pbc_separation(pos[1] - ref[1], simulator.Ly) for pos, ref in zip(positions, reference_positions)])

        # Mean squared displacement
        msd = np.mean(dx**2 + dy**2)
        msd_values.append(msd)

    return np.array(msd_values)

# Function to plot system metrics over time
def plot_metrics(simulator, title):
    """Plot temperature, pressure, energy over time"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot temperature
    ax1.plot(simulator.time_history, simulator.temperature_history)
    ax1.set_ylabel('Temperature')
    ax1.set_title(f'{title} - System Metrics')
    ax1.grid(True)

    # Plot pressure
    ax2.plot(simulator.time_history, simulator.pressure_history)
    ax2.set_ylabel('Pressure')
    ax2.grid(True)

    # Plot energies
    ax3.plot(simulator.time_history, simulator.potential_energy_history, label='Potential')
    ax3.plot(simulator.time_history, simulator.kinetic_energy_history, label='Kinetic')
    ax3.plot(simulator.time_history, simulator.total_energy_history, label='Total')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Energy')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

# Main script for Problem 8.11
if __name__ == "__main__":
    # Part (a) - Triangular lattice with zero initial velocity
    print("\n===== Part (a) - Triangular lattice with zero initial velocity =====")
    N = 64
    Lx = 8.0
    Ly = np.sqrt(3) * Lx / 2

    # Create simulator with triangular lattice
    sim_a = LJSimulator(N=N, Lx=Lx, Ly=Ly, temperature=0.0, lattice_type='triangular')

    # Set all velocities to zero
    sim_a.vx = np.zeros(N)
    sim_a.vy = np.zeros(N)

    # Recalculate metrics
    sim_a.compute_metrics()

    # Print initial state
    print(f"Initial configuration:")
    print(f"N = {N}, Lx = {Lx}, Ly = {Ly}")
    print(f"Initial potential energy = {sim_a.potential_energy:.6f}")
    print(f"Initial kinetic energy = {sim_a.kinetic_energy:.6f}")
    print(f"Initial total energy = {sim_a.total_energy:.6f}")
    print(f"Initial temperature = {sim_a.temperature:.6f}")
    print(f"Initial pressure = {sim_a.pressure:.6f}")

    # Run simulation
    for _ in range(1000):
        sim_a.step()

    # Print final state
    print(f"\nAfter 1000 steps:")
    print(f"Final potential energy = {sim_a.potential_energy:.6f}")
    print(f"Final kinetic energy = {sim_a.kinetic_energy:.6f}")
    print(f"Final total energy = {sim_a.total_energy:.6f}")
    print(f"Final temperature = {sim_a.temperature:.6f}")
    print(f"Final pressure = {sim_a.pressure:.6f}")

    # Plot results
    plot_metrics(sim_a, "Part (a)")

    # Store initial positions for MSD calculation
    initial_positions = sim_a.snapshots[0]['positions']

    # Calculate MSD
    msd_a = calculate_msd(sim_a, initial_positions)

    plt.figure(figsize=(8, 6))
    plt.plot(np.array(range(len(msd_a))) * sim_a.snapshot_interval * sim_a.dt, msd_a)
    plt.xlabel('Time')
    plt.ylabel('Mean Squared Displacement')
    plt.title('Part (a) - MSD from Initial Positions')
    plt.grid(True)
    plt.show()

    # Part (b) - Triangular lattice with random initial velocities
    print("\n===== Part (b) - Triangular lattice with random initial velocities =====")

    # Create simulator with triangular lattice
    sim_b = LJSimulator(N=N, Lx=Lx, Ly=Ly, temperature=0.0, lattice_type='triangular')

    # Set random velocities in range [-0.5, 0.5]
    sim_b.vx = np.random.uniform(-0.5, 0.5, N)
    sim_b.vy = np.random.uniform(-0.5, 0.5, N)

    # Remove center of mass motion
    sim_b.vx -= np.mean(sim_b.vx)
    sim_b.vy -= np.mean(sim_b.vy)

    # Recalculate metrics
    sim_b.compute_metrics()

    # Print initial state
    print(f"Initial configuration:")
    print(f"Initial potential energy = {sim_b.potential_energy:.6f}")
    print(f"Initial kinetic energy = {sim_b.kinetic_energy:.6f}")
    print(f"Initial total energy = {sim_b.total_energy:.6f}")
    print(f"Initial temperature = {sim_b.temperature:.6f}")
    print(f"Initial pressure = {sim_b.pressure:.6f}")

    # Run simulation to equilibrate
    for _ in range(2000):
        sim_b.step()

    # Print equilibrated state
    print(f"\nAfter equilibration (2000 steps):")
    print(f"Equilibrium potential energy = {sim_b.potential_energy:.6f}")
    print(f"Equilibrium kinetic energy = {sim_b.kinetic_energy:.6f}")
    print(f"Equilibrium total energy = {sim_b.total_energy:.6f}")
    print(f"Equilibrium temperature = {sim_b.temperature:.6f}")
    print(f"Equilibrium pressure = {sim_b.pressure:.6f}")

    # Calculate mean values after equilibration (second half of simulation)
    halfway = len(sim_b.temperature_history) // 2
    mean_temp_b = np.mean(sim_b.temperature_history[halfway:])
    mean_press_b = np.mean(sim_b.pressure_history[halfway:])
    mean_pe_b = np.mean(sim_b.potential_energy_history[halfway:])
    mean_ke_b = np.mean(sim_b.kinetic_energy_history[halfway:])
    mean_te_b = np.mean(sim_b.total_energy_history[halfway:])

    print(f"\nMean values after equilibration:")
    print(f"Mean temperature = {mean_temp_b:.6f}")
    print(f"Mean pressure = {mean_press_b:.6f}")
    print(f"Mean potential energy = {mean_pe_b:.6f}")
    print(f"Mean kinetic energy = {mean_ke_b:.6f}")
    print(f"Mean total energy = {mean_te_b:.6f}")

    # Plot results
    plot_metrics(sim_b, "Part (b)")

    # Store equilibrium positions for MSD calculation
    equil_positions = sim_b.snapshots[-1]['positions']

    # Store equilibrium state for part (c)
    equil_state_b = {
        'positions': equil_positions.copy(),
        'velocities': np.column_stack((sim_b.vx, sim_b.vy)).copy(),
        'temperature': sim_b.temperature,
        'pressure': sim_b.pressure,
        'total_energy': sim_b.total_energy
    }

    # Part (c) - Increasing temperature from equilibrium state
    print("\n===== Part (c) - Increasing temperature from equilibrium state =====")

    # Create simulator from equilibrium state
    sim_c = LJSimulator(N=N, Lx=Lx, Ly=Ly, temperature=0.0, lattice_type='triangular')

    # Set positions and velocities from equilibrium state
    sim_c.x = equil_state_b['positions'][:, 0].copy()
    sim_c.y = equil_state_b['positions'][:, 1].copy()
    sim_c.vx = equil_state_b['velocities'][:, 0].copy()
    sim_c.vy = equil_state_b['velocities'][:, 1].copy()

    # Recalculate metrics
    sim_c.compute_acceleration()
    sim_c.compute_metrics()

    # Record initial energy for part (d)
    E0 = sim_c.total_energy

    # Temperature values to examine
    temperatures = []
    pressures = []
    energies = []

    # Record initial state
    temperatures.append(sim_c.temperature)
    pressures.append(sim_c.pressure)
    energies.append(sim_c.total_energy)

    # Run multiple temperature increases
    for temp_factor in [2, 4, 8, 16]:
        # Double temperature by scaling velocities
        scale_factor = np.sqrt(temp_factor)
        sim_c.vx *= scale_factor
        sim_c.vy *= scale_factor

        # Recalculate metrics
        sim_c.compute_metrics()

        print(f"\nAfter scaling velocities by {scale_factor}:")
        print(f"New temperature = {sim_c.temperature:.6f}")
        print(f"New total energy = {sim_c.total_energy:.6f}")

        # Clear history for this temperature
        sim_c.clear_history()

        # Run simulation to equilibrate at new temperature
        for _ in range(1000):
            sim_c.step()

        # Calculate mean values after equilibration (second half of simulation)
        halfway = len(sim_c.temperature_history) // 2
        mean_temp = np.mean(sim_c.temperature_history[halfway:])
        mean_press = np.mean(sim_c.pressure_history[halfway:])
        mean_energy = np.mean(sim_c.total_energy_history[halfway:])

        print(f"After equilibration:")
        print(f"Mean temperature = {mean_temp:.6f}")
        print(f"Mean pressure = {mean_press:.6f}")
        print(f"Mean total energy = {mean_energy:.6f}")

        # Record values
        temperatures.append(mean_temp)
        pressures.append(mean_press)
        energies.append(mean_energy)

    # Part (d) - Analysis of energy and pressure vs temperature
    print("\n===== Part (d) - Analysis of energy and pressure vs temperature =====")

    temperatures = np.array(temperatures)
    pressures = np.array(pressures)
    energies = np.array(energies)

    # Plot E(T) - E(0) vs T
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(temperatures, energies - E0, 'o-')

    # Fit a line to test proportionality
    z = np.polyfit(temperatures, energies - E0, 1)
    p = np.poly1d(z)
    plt.plot(temperatures, p(temperatures), 'r--', label=f'Fit: {z[0]:.3f}*T + {z[1]:.3f}')

    plt.xlabel('Temperature (T)')
    plt.ylabel('E(T) - E(0)')
    plt.title('Energy vs Temperature')
    plt.grid(True)
    plt.legend()

    # Plot P(T) vs T
    plt.subplot(2, 1, 2)
    plt.plot(temperatures, pressures, 'o-')

    # Fit a line to test proportionality
    z = np.polyfit(temperatures, pressures, 1)
    p = np.poly1d(z)
    plt.plot(temperatures, p(temperatures), 'r--', label=f'Fit: {z[0]:.3f}*T + {z[1]:.3f}')

    plt.xlabel('Temperature (T)')
    plt.ylabel('Pressure P(T)')
    plt.title('Pressure vs Temperature')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Calculate heat capacity (from slope of energy vs temperature)
    heat_capacity = z[0]
    print(f"Estimated heat capacity: {heat_capacity:.6f}")

    # Part (e) - Decreasing density until melting
    print("\n===== Part (e) - Decreasing density until melting =====")

    # Start from equilibrium configuration from part (b)
    sim_e = LJSimulator(N=N, Lx=Lx, Ly=Ly, temperature=0.0, lattice_type='triangular')

    # Set positions and velocities from equilibrium state
    sim_e.x = equil_state_b['positions'][:, 0].copy()
    sim_e.y = equil_state_b['positions'][:, 1].copy()
    sim_e.vx = equil_state_b['velocities'][:, 0].copy()
    sim_e.vy = equil_state_b['velocities'][:, 1].copy()

    # Recalculate metrics
    sim_e.compute_acceleration()
    sim_e.compute_metrics()

    # Store initial density
    initial_density = N / (sim_e.Lx * sim_e.Ly)

    # Density factors to try
    density_factors = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    msd_values = []

    for factor in density_factors:
        print(f"\nTesting density factor: {factor:.1f}")

        # Calculate new box size
        new_Lx = Lx * np.sqrt(factor)
        new_Ly = Ly * np.sqrt(factor)

        # Set new box size (this also rescales positions)
        sim_e.set_box_size(new_Lx, new_Ly)

        # Reset history for this density
        sim_e.clear_history()

        # Store reference positions
        ref_positions = np.column_stack((sim_e.x, sim_e.y))

        # Run simulation
        for _ in range(1000):
            sim_e.step()

        # Calculate new density
        density = N / (sim_e.Lx * sim_e.Ly)
        density_ratio = initial_density / density

        print(f"New density: {density:.6f} (ratio to initial: {density_ratio:.6f})")
        print(f"Final temperature: {sim_e.temperature:.6f}")
        print(f"Final pressure: {sim_e.pressure:.6f}")

        # Calculate MSD from initial positions at this density
        if len(sim_e.snapshots) > 0:
            final_msd = calculate_msd(sim_e, ref_positions)[-1]
            msd_values.append(final_msd)
            print(f"Final MSD from reference positions: {final_msd:.6f}")

    # Plot MSD vs density factor to help identify melting transition
    plt.figure(figsize=(8, 6))
    plt.plot(density_factors, msd_values, 'o-')
    plt.xlabel('Density Factor')
    plt.ylabel('Final MSD')
    plt.title('Mean Squared Displacement vs Density Factor')
    plt.grid(True)
    plt.show()
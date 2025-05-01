import numpy as np

class LJSimulator:
    def __init__(self, N=64, Lx=10.0, Ly=10.0, temperature=1.0, dt=0.01, lattice_type='square', snapshot_interval=10):
        self.N = N
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.initial_temperature = temperature
        self.snapshot_interval = snapshot_interval

        # Arrays for positions, velocities, and accelerations
        self.x = np.zeros(N)
        self.y = np.zeros(N)
        self.vx = np.zeros(N)
        self.vy = np.zeros(N)
        self.ax = np.zeros(N)
        self.ay = np.zeros(N)

        # Tracking
        self.time = 0.0
        self.step_count = 0
        self.potential_energy = 0.0
        self.kinetic_energy = 0.0
        self.total_energy = 0.0
        self.temperature = temperature
        self.pressure = 0.0
        self.virial = 0.0

        # History tracking
        self.time_history = []
        self.potential_energy_history = []
        self.kinetic_energy_history = []
        self.total_energy_history = []
        self.temperature_history = []
        self.pressure_history = []
        self.kinetic_energy_squared_history = []  # For heat capacity calculation
        self.snapshots = []

        # Set particle arrangement: square lattice, triangular lattice, or random (removed for brevity)
        if lattice_type == 'square':
            self.set_square_lattice()
        elif lattice_type == 'triangular':
            self.set_triangular_lattice()
        else:
            self.set_square_lattice()

        # Initialize velocities to get the desired temperature
        self.set_velocities()

        # Calculate initial accelerations and system properties
        self.compute_acceleration()
        self.compute_metrics()

        # Take initial snapshot
        self.take_snapshot()

        # Debugging
        # print(f"Initial temperature set to: {self.initial_temperature}")
        # print(f"Actual temperature from velocities: {np.sum(self.vx**2 + self.vy**2) / (2 * self.N)}")

        # print(f"Initial snapshot taken at time {self.time}")
        # print(f"Number of snapshots after init: {len(self.snapshots)}")


    ################################
    ##### Particle Arrangement #####
    ################################
    def set_square_lattice(self):
        """Place particles on a square lattice"""
        nx = int(np.sqrt(self.N))  # particles per row
        ny = int(np.ceil(self.N / nx))  # particles per column

        # Calculate spacing between particles
        dx = self.Lx / nx
        dy = self.Ly / ny

        # Place particles
        count = 0
        for ix in range(nx):
            for iy in range(ny):
                if count < self.N:
                    self.x[count] = dx * (ix + 0.5)  # Center in the cell
                    self.y[count] = dy * (iy + 0.5)
                    count += 1

    def set_triangular_lattice(self):
        """Place particles on a triangular lattice"""
        nx = int(np.sqrt(self.N))  # approximate particles per row
        ny = int(np.ceil(self.N / nx))  # particles per column

        # Calculate spacing between particles
        dx = self.Lx / nx
        dy = self.Ly / ny

        count = 0
        for ix in range(nx):
            for iy in range(ny):
                if count < self.N:
                    self.y[count] = dy * (iy + 0.5)

                    # Offset even/odd rows
                    if iy % 2 == 0:
                        self.x[count] = dx * (ix + 0.25)
                    else:
                        self.x[count] = dx * (ix + 0.75)

                    count += 1

    #############################
    ##### Simulator Control #####
    #############################
    def resize_box(self, Lx, Ly):
        return self.set_box_size(Lx, Ly)

    def set_box_size(self, Lx, Ly):
        """Set the size of the simulation box, rescaling particle positions"""

        # Calculate scaling factors
        scale_x = Lx / self.Lx
        scale_y = Ly / self.Ly

        # Rescale positions
        self.x *= scale_x
        self.y *= scale_y

        # Update box dimensions
        self.Lx = Lx
        self.Ly = Ly

        # Recalculate forces and system properties after changing box dimensions
        self.compute_acceleration()
        self.compute_metrics()

    def set_temperature(self, target_temperature):
        """
        Adjust system temperature by rescaling velocities
        """
        current_temperature = self.temperature
        if current_temperature > 0:
            # Calculate scaling factor
            scale_factor = np.sqrt(target_temperature / current_temperature)

            # Scale velocities
            self.vx *= scale_factor
            self.vy *= scale_factor

            # Update energy, temperature, and pressure
            self.compute_metrics()

    def set_velocities(self):
        """Set random velocities with zero center-of-mass momentum (Gould 8.3)"""
        # Generate random velocities
        self.vx = np.random.randn(self.N) - 0.5
        self.vy = np.random.randn(self.N) - 0.5

        # Set center-of-mass momentum to zero
        self.vx -= np.mean(self.vx)
        self.vy -= np.mean(self.vy)

        # Calculate current kinetic energy
        current_ke = 0.5 * np.sum(self.vx**2 + self.vy**2) / self.N

        # Scale velocities to match desired temperature
        scale_factor = np.sqrt(self.initial_temperature / current_ke)
        self.vx *= scale_factor
        self.vy *= scale_factor

    def remove_drift(self):
        """Remove net momentum by adjusting velocities to keep total linear momentum zero."""
        # Total momentum in x and y
        px = np.sum(self.vx)
        py = np.sum(self.vy)

        # Subtract average momentum per particle from each velocity
        self.vx -= px / self.N
        self.vy -= py / self.N

    def step(self):
        """Perform one time step using the Velocity Verlet algorithm"""

        # Debugging
        # print(f"Starting step {self.step_count} at time {self.time}")

        # First half of velocity update
        self.vx += 0.5 * self.dt * self.ax
        self.vy += 0.5 * self.dt * self.ay

        # Update positions
        self.x += self.dt * self.vx
        self.y += self.dt * self.vy

        # Apply periodic boundary conditions
        for i in range(self.N):
            self.x[i] = self.pbc_position(self.x[i], self.Lx)
            self.y[i] = self.pbc_position(self.y[i], self.Ly)

        # Calculate new accelerations
        self.compute_acceleration()

        # Second half of velocity update
        self.vx += 0.5 * self.dt * self.ax
        self.vy += 0.5 * self.dt * self.ay

        # Update time and step count
        self.time += self.dt
        self.step_count += 1

        # Update energy, temperature, and pressure
        self.compute_metrics()

        # Record history
        self.record_history()

        # Take snapshot if needed
        if self.step_count % self.snapshot_interval == 0:
            self.take_snapshot()

    ########################################
    ##### Periodic Boundary Conditions #####
    ########################################
    def pbc_separation(self, dr, L):
        """Apply minimum image convention for distance"""
        if dr > 0.5 * L:
            return dr - L
        elif dr < -0.5 * L:
            return dr + L
        return dr

    def pbc_position(self, r, L):
        """Apply periodic boundary conditions to position"""
        return r % L

    ###############################################
    ##### Lennard-Jones Particle Interactions #####
    ###############################################
    def compute_acceleration(self):
        """Calculate forces and accelerations using Lennard-Jones potential"""
        # Reset accelerations and potential energy
        self.ax = np.zeros(self.N)
        self.ay = np.zeros(self.N)
        self.potential_energy = 0.0
        self.virial = 0.0

        # Loop over all pairs of particles
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                # Calculate separation with periodic boundary conditions
                dx = self.pbc_separation(self.x[i] - self.x[j], self.Lx)
                dy = self.pbc_separation(self.y[i] - self.y[j], self.Ly)

                # Square distance
                r2 = dx**2 + dy**2

                # Compute Lennard-Jones force and potential
                if r2 > 0:  # Avoid division by zero
                    r2i = 1.0 / r2
                    r6i = r2i**3
                    f_mag= 48.0 * r2i * r6i * (r6i - 0.5)

                    # Update accelerations
                    self.ax[i] += f_mag* dx
                    self.ay[i] += f_mag* dy
                    self.ax[j] -= f_mag* dx
                    self.ay[j] -= f_mag* dy

                    # Update potential energy
                    self.potential_energy += 4.0 * r6i * (r6i - 1.0)

                    # Update virial (pressure term)
                    self.virial += f_mag * (dx*dx + dy*dy)

    ###########################################
    ##### History and Snapshot Management #####
    ###########################################
    def record_history(self):
        """Record current values in history"""
        self.time_history.append(self.time)
        self.potential_energy_history.append(self.potential_energy)
        self.kinetic_energy_history.append(self.kinetic_energy)
        self.total_energy_history.append(self.total_energy)
        self.temperature_history.append(self.temperature)
        self.pressure_history.append(self.pressure)

        # For heat capacity calculation
        self.kinetic_energy_squared_history.append(self.kinetic_energy**2)

    def clear_history(self):
        """Clear recorded history and snapshots"""
        self.time_history = []
        self.potential_energy_history = []
        self.kinetic_energy_history = []
        self.total_energy_history = []
        self.temperature_history = []
        self.pressure_history = []
        self.kinetic_energy_squared_history = []
        self.snapshots = []

    def take_snapshot(self):
        """Store current state in snapshots list"""
        positions = np.column_stack((self.x, self.y))
        velocities = np.column_stack((self.vx, self.vy))

        snapshot = {
            'Lx': self.Lx,
            'Ly': self.Ly,
            'time': self.time,
            'positions': positions.copy(),
            'velocities': velocities.copy(),
            'temperature': self.temperature,
            'pressure': self.pressure,
            'potential_energy': self.potential_energy,
            'kinetic_energy': self.kinetic_energy,
            'total_energy': self.total_energy
        }
        self.snapshots.append(snapshot)
        # print(f"Snapshot taken at time {self.time}")

    ################################
    ##### Metrics Calculations #####
    ################################
    def compute_metrics(self):
        """
        Calculate energy, temperature, and pressure (which depends on temperature)
        This method ensures that the latest temperature is used for the pressure calculation
        """
        self.compute_energy()
        self.compute_temperature()
        self.compute_pressure()  # depends on temperature!

    def compute_energy(self):
        """Calculate the kinetic energy and total energy"""
        self.kinetic_energy = 0.5 * np.sum(self.vx**2 + self.vy**2)
        self.total_energy = self.kinetic_energy + self.potential_energy

    def compute_temperature(self):
        """Calculate the current temperature based on equation (8.5)"""
        # For 2D system with N particles
        self.temperature = np.sum(self.vx**2 + self.vy**2) / (2 * self.N)

    def compute_pressure(self):
        """Calculate the pressure using the virial equation"""
        temperature = self.temperature
        volume = self.Lx * self.Ly
        # Calculate pressure using the virial equation
        # P = (N * T + 0.5 * virial) / V
        self.pressure = (self.N * temperature + 0.5 * self.virial) / volume
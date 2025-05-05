import numpy as np

class HourglassSimulator:
    def __init__(self, N=64, Lx=20.0, Ly=20.0, temperature=1.0, dt=0.001, gravity=0.1,
                 particle_radius=0.5, k=1.0, gamma=0.3,
                 snapshot_interval=10):

        # Init simulation variables
        self.N = N
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.gravity = gravity
        self.particle_radius = particle_radius
        self.k = k  # Spring constant
        self.gamma = gamma  # Damping coefficient
        self.initial_temperature = temperature
        self.snapshot_interval = snapshot_interval

        # Arrays for positions, radii, velocities, and accelerations
        self.x = np.zeros(N)
        self.y = np.zeros(N)
        self.r = np.zeros(N)
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

        # # Initialize velocities to get the desired temperature
        # self.set_velocities()

        # Calculate initial accelerations and system properties
        self.compute_acceleration()
        self.compute_metrics()

        # Take initial snapshot
        self.take_snapshot()

    ################################
    ##### Particle Arrangement #####
    ################################
    def initialize_particles(self):
        """Initialize particles with better spacing to reduce overlap"""
        # Set proper radii
        self.r = np.ones(self.N) * self.particle_radius

        # Create a more spaced out grid of particles
        particles_per_row = int(np.sqrt(self.N))
        rows = (self.N + particles_per_row - 1) // particles_per_row

        # Calculate spacing based on radius to minimize overlap
        x_spacing = self.Lx / (particles_per_row + 1)
        y_spacing = 2.5 * self.r[0]  # Extra vertical spacing to prevent immediate overlap

        count = 0
        for i in range(rows):
            for j in range(particles_per_row):
                if count < self.N:
                    self.x[count] = (j + 1) * x_spacing
                    self.y[count] = self.Ly - (i + 1) * y_spacing

                    # Add small random offsets to avoid perfect alignment and make it look more natural
                    self.x[count] += np.random.uniform(-0.1, 0.1) * self.r[0]

                    count += 1

        # Zero initial velocities - let gravity do the work
        self.vx = np.zeros(self.N)
        self.vy = np.zeros(self.N)

    #############################
    ##### Simulator Control #####
    #############################
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

        # Apply boundary conditions (replacing periodic boundaries)
        self.handle_boundaries()

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

    #############################################
    ##### Granular Forces Using Hooke's Law #####
    #############################################
    def compute_acceleration(self):
        """Calculate forces and accelerations using granular contact model with Hooke's law"""
        # Reset accelerations and potential energy
        self.ax = np.zeros(self.N)
        self.ay = np.zeros(self.N)
        self.potential_energy = 0.0
        self.virial = 0.0

        # Add constant gravitational acceleration
        self.ay -= self.gravity  # Apply gravity to all particles

        # Loop over all pairs of particles
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                # Calculate separation vector
                dx = self.x[i] - self.x[j]
                dy = self.y[i] - self.y[j]

                # Calculate distance and normalized direction vector
                r_ij = np.sqrt(dx**2 + dy**2)

                # Sum of radii - determines if particles are overlapping
                r_sum = self.r[i] + self.r[j]

                # Only calculate forces if particles overlap
                if r_ij < r_sum:
                    # Normalized direction vector
                    if r_ij > 0:  # Avoid division by zero
                        nx = dx / r_ij
                        ny = dy / r_ij

                        # Relative velocity
                        dvx = self.vx[i] - self.vx[j]
                        dvy = self.vy[i] - self.vy[j]

                        # Dot product of relative velocity and direction
                        v_dot_r = dvx * nx + dvy * ny

                        # Spring force component: -k(Ri + Rj - rij)(rij/|rij|)
                        spring_mag = self.k * (r_sum - r_ij)

                        # Damping force component: γ(vij·rij)(rij/rij²)
                        damping_mag = self.gamma * v_dot_r

                        # Total force magnitude
                        f_x = (spring_mag * nx) - (damping_mag * nx)
                        f_y = (spring_mag * ny) - (damping_mag * ny)

                        # Update accelerations (F = ma, assuming m=1)
                        self.ax[i] += f_x
                        self.ay[i] += f_y
                        self.ax[j] -= f_x
                        self.ay[j] -= f_y

                        # Update potential energy (spring potential energy)
                        self.potential_energy += 0.5 * self.k * (r_sum - r_ij)**2

                        # Update virial (for pressure calculation)
                        self.virial += f_x * dx + f_y * dy

        # Apply wall constraints
        self.apply_wall_constraints()

        # print(f"Gravity force: {self.gravity}")
        # print(f"Max particle force: {np.max(np.abs(self.ay - self.gravity))}")

    ##### Wall Constraints #####
    def add_left_wall(self, top_x, bottom_x, wall_width):
        """
        Add a diagonal left wall to the hourglass simulation.

        Parameters:
        -----------
        top_x : float
            X-coordinate of the top of the wall
        bottom_x : float
            X-coordinate of the bottom of the wall (should be > top_x to slope inward)
        wall_width : float
            Width/thickness of the wall
        """
        self.left_wall_top_x = top_x
        self.left_wall_bottom_x = bottom_x
        self.left_wall_width = wall_width

        # Calculate slope of the wall (m in y = mx + b)
        self.left_wall_slope = self.Ly / (bottom_x - top_x) if bottom_x != top_x else float('inf')

    def add_right_wall(self, top_x, bottom_x, wall_width):
        """
        Add a diagonal right wall to the hourglass simulation.

        Parameters:
        -----------
        top_x : float
            X-coordinate of the top of the wall
        bottom_x : float
            X-coordinate of the bottom of the wall (should be < top_x to slope inward)
        wall_width : float
            Width/thickness of the wall
        """
        self.right_wall_top_x = top_x
        self.right_wall_bottom_x = bottom_x
        self.right_wall_width = wall_width

        # Calculate slope of the wall (m in y = mx + b)
        self.right_wall_slope = self.Ly / (top_x - bottom_x) if top_x != bottom_x else float('inf')

    def handle_boundaries(self):
        """Handle particles going beyond boundaries, including both diagonal walls."""
        for i in range(self.N):
            # LEFT WALL COLLISION HANDLING
            if hasattr(self, 'left_wall_top_x') and hasattr(self, 'left_wall_bottom_x'):
                # Calculate current wall x-position at the particle's y-position
                # Linear interpolation between top and bottom wall positions
                # y = 0 at bottom, y = Ly at top
                wall_x_at_y = self.left_wall_top_x + (self.left_wall_bottom_x - self.left_wall_top_x) * (1 - self.y[i] / self.Ly)

                # Calculate the normal vector to the wall (perpendicular to the wall)
                # Wall vector points from bottom to top
                wall_vec_x = self.left_wall_bottom_x - self.left_wall_top_x
                wall_vec_y = self.Ly
                wall_length = np.sqrt(wall_vec_x**2 + wall_vec_y**2)

                # Normal vector (pointing right, away from wall)
                normal_x = wall_vec_y / wall_length
                normal_y = -wall_vec_x / wall_length

                # Check if particle is beyond the left wall (considering its radius)
                if self.x[i] < wall_x_at_y + self.left_wall_width + self.r[i]:
                    # Calculate penetration depth
                    penetration = (wall_x_at_y + self.left_wall_width + self.r[i]) - self.x[i]

                    # Move the particle to the wall surface (along the normal)
                    self.x[i] += penetration

                    # Reflect velocity with some energy loss (0.9 factor)
                    # Calculate the velocity component along the normal
                    v_dot_n = self.vx[i] * normal_x + self.vy[i] * normal_y

                    # Only reflect if particle is moving toward the wall
                    if v_dot_n < 0:
                        # Update velocity components to reflect off the wall
                        self.vx[i] -= 2 * v_dot_n * normal_x * 0.9
                        self.vy[i] -= 2 * v_dot_n * normal_y * 0.9

            # RIGHT WALL COLLISION HANDLING
            if hasattr(self, 'right_wall_top_x') and hasattr(self, 'right_wall_bottom_x'):
                # Calculate current wall x-position at the particle's y-position
                # Linear interpolation between top and bottom wall positions
                wall_x_at_y = self.right_wall_top_x + (self.right_wall_bottom_x - self.right_wall_top_x) * (1 - self.y[i] / self.Ly)

                # Calculate the normal vector to the wall (perpendicular to the wall)
                # Wall vector points from bottom to top
                wall_vec_x = self.right_wall_top_x - self.right_wall_bottom_x
                wall_vec_y = self.Ly
                wall_length = np.sqrt(wall_vec_x**2 + wall_vec_y**2)

                # Normal vector (pointing left, away from wall)
                normal_x = -wall_vec_y / wall_length
                normal_y = wall_vec_x / wall_length

                # Check if particle is beyond the right wall (considering its radius)
                if self.x[i] > wall_x_at_y - self.right_wall_width - self.r[i]:
                    # Calculate penetration depth
                    penetration = self.x[i] - (wall_x_at_y - self.right_wall_width - self.r[i])

                    # Move the particle to the wall surface (along the normal)
                    self.x[i] -= penetration

                    # Reflect velocity with some energy loss (0.9 factor)
                    # Calculate the velocity component along the normal
                    v_dot_n = self.vx[i] * normal_x + self.vy[i] * normal_y

                    # Only reflect if particle is moving toward the wall
                    if v_dot_n < 0:
                        # Update velocity components to reflect off the wall
                        self.vx[i] -= 2 * v_dot_n * normal_x * 0.9
                        self.vy[i] -= 2 * v_dot_n * normal_y * 0.9

            # Bounce off the bottom wall
            if self.y[i] < self.r[i]:
                self.y[i] = self.r[i]
                self.vy[i] = -self.vy[i] * 0.9

    def apply_wall_constraints(self):
        """Apply forces from walls to particles"""
        # For now, just implement simple box walls
        for i in range(self.N):
            # Left wall
            if self.x[i] < self.r[i]:
                penetration = self.r[i] - self.x[i]
                self.ax[i] += self.k * penetration - self.gamma * self.vx[i]

            # Right wall
            if self.x[i] > self.Lx - self.r[i]:
                penetration = self.x[i] - (self.Lx - self.r[i])
                self.ax[i] -= self.k * penetration - self.gamma * self.vx[i]

            # Bottom wall
            if self.y[i] < self.r[i]:
                penetration = self.r[i] - self.y[i]
                self.ay[i] += self.k * penetration - self.gamma * self.vy[i]

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
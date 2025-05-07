import numpy as np
import time
from numba import njit

class HourglassSimulator:
    """
    A modular granular materials simulator that can use either Hooke's law
    or Hertzian contact model for particle interactions.
    """
    def __init__(self,
                 N=200,                  # Number of particles
                 Lx=20.0,                # Box width
                 Ly=20.0,                # Box height
                 temperature=0.0,        # Initial temperature
                 dt=0.005,               # Time step size
                 gravity=5.0,            # Gravitational acceleration
                 particle_radius=0.5,    # Radius of particles
                 k=20.0,                 # Spring constant
                 gamma=2.0,              # Damping coefficient
                 contact_model="hertzian", # "hertzian" or "hooke"
                 neck_width=3.0,         # Width of hourglass neck
                 wall_width=0.5,         # Thickness of walls
                 friction_coef=0.5,      # Friction coefficient
                 restitution_coef=0.3,   # Coefficient of restitution
                 snapshot_interval=10,   # Interval for taking snapshots
                 respawn_particles=False # Whether to respawn particles
                ):

        # Simulation parameters
        self.N = N
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.gravity = gravity
        self.particle_radius = particle_radius
        self.k = k
        self.gamma = gamma
        self.friction_coef = friction_coef
        self.restitution_coef = restitution_coef
        self.contact_model = contact_model
        self.neck_width = neck_width
        self.wall_width = wall_width
        self.initial_temperature = temperature
        self.snapshot_interval = snapshot_interval
        self.respawn_particles = respawn_particles

        # Arrays for particles
        self.x = np.zeros(N)
        self.y = np.zeros(N)
        self.r = np.ones(N) * particle_radius
        self.vx = np.zeros(N)
        self.vy = np.zeros(N)
        self.ax = np.zeros(N)
        self.ay = np.zeros(N)

        # Respawn tracking
        self.respawn_count = np.zeros(N, dtype=int)
        self.total_respawns = 0

        # Tracking variables
        self.time = 0.0
        self.step_count = 0
        self.potential_energy = 0.0
        self.kinetic_energy = 0.0
        self.total_energy = 0.0
        self.temperature = 0.0
        self.pressure = 0.0
        self.virial = 0.0

        # History tracking
        self.time_history = []
        self.potential_energy_history = []
        self.kinetic_energy_history = []
        self.total_energy_history = []
        self.temperature_history = []
        self.pressure_history = []
        self.kinetic_energy_squared_history = []
        self.snapshots = []

        # Wall segments for ramps and hourglass
        self.left_wall_segments = []
        self.right_wall_segments = []

        print(f"Using {self.contact_model} contact model for particle interactions.")
        if self.respawn_particles:
            print(f"Particle respawning enabled. Particles will respawn when they hit the bottom.")

    ##############################
    ##### Particle Placement #####
    ##############################
    def place_particle_at_top(self, index):
        """
        Helper function to place a particle at the top of the hourglass
        Used by both initialization and respawning
        """
        # Get valid area for placement
        wall_width = self.wall_width
        left_top_x = 0.0
        right_top_x = self.Lx

        # If we have wall segments defined, get the top positions
        if self.left_wall_segments:
            top_segment = self.left_wall_segments[0]
            left_top_x = top_segment['top_x'] + wall_width

        if self.right_wall_segments:
            top_segment = self.right_wall_segments[0]
            right_top_x = top_segment['top_x'] - wall_width

        # Area where particles can be placed
        valid_width = right_top_x - left_top_x

        # Define the top area height (25% of total height)
        top_area_height = self.Ly * 0.25
        bottom_y = self.Ly - top_area_height

        # Calculate new position
        new_x = left_top_x + np.random.random() * valid_width
        new_y = bottom_y + np.random.random() * top_area_height

        # Set position and small random velocity
        self.x[index] = new_x
        self.y[index] = new_y
        self.vx[index] = np.random.uniform(-0.01, 0.01)
        self.vy[index] = 0.0

        return True

    def initialize_random_falling_particles(self):
        """Initialize particles randomly at the top of the hourglass"""
        # Set proper radii
        self.r = np.ones(self.N) * self.particle_radius

        # Minimum distance between particles
        min_distance = 2.2 * self.particle_radius

        # Initialize all particles
        for i in range(self.N):
            position_valid = False
            attempt = 0
            max_attempts = 100

            while not position_valid and attempt < max_attempts:
                # Place particle at top
                self.place_particle_at_top(i)

                # Check distance from previously placed particles
                position_valid = True
                for j in range(i):
                    dx = self.x[i] - self.x[j]
                    dy = self.y[i] - self.y[j]
                    distance = np.sqrt(dx**2 + dy**2)

                    if distance < min_distance:
                        position_valid = False
                        break

                attempt += 1

        # Initialize system
        self.compute_acceleration()
        self.compute_metrics()

    def respawn_particle(self, index):
        """Respawn a particle at the top of the hourglass"""
        if not self.respawn_particles:
            return

        # Place particle at top using the common function
        self.place_particle_at_top(index)

        # Update respawn counter
        self.respawn_count[index] += 1
        self.total_respawns += 1

    def set_velocities(self):
        """Set random velocities with zero center-of-mass momentum"""
        if self.initial_temperature > 0:
            # Generate random velocities
            self.vx = np.random.randn(self.N) - 0.5
            self.vy = np.random.randn(self.N) - 0.5

            # Set center-of-mass momentum to zero
            self.vx -= np.mean(self.vx)
            self.vy -= np.mean(self.vy)

            # Calculate current kinetic energy
            current_ke = 0.5 * np.sum(self.vx**2 + self.vy**2) / self.N

            # Scale velocities to match desired temperature
            if current_ke > 0:
                scale_factor = np.sqrt(self.initial_temperature / current_ke)
                self.vx *= scale_factor
                self.vy *= scale_factor
        else:
            # Zero initial velocities
            self.vx = np.zeros(self.N)
            self.vy = np.zeros(self.N)

    ############################################
    ##### Simplified Wall Segment Creation #####
    ############################################
    def draw_hourglass(self):
        """Creates an hourglass shape by placing converging walls in the middle"""
        # Calculate wall positions to create an hourglass shape
        left_wall_top_x = 0.0       # Left edge at top
        left_wall_middle_x = (self.Lx - self.neck_width) / 2  # Position at the neck
        left_wall_bottom_x = 0.0    # Left edge at bottom

        right_wall_top_x = self.Lx   # Right edge at top
        right_wall_middle_x = self.Lx - left_wall_middle_x  # Position at the neck
        right_wall_bottom_x = self.Lx  # Right edge at bottom

        # Clear any existing wall segments
        self.left_wall_segments = []
        self.right_wall_segments = []

        # Add top half of hourglass (top to middle)
        self.left_wall_segments.append({
            'top_x': left_wall_top_x,
            'bottom_x': left_wall_middle_x,
            'top_y': self.Ly,
            'bottom_y': self.Ly / 2,  # Middle height
            'width': self.wall_width
        })

        self.right_wall_segments.append({
            'top_x': right_wall_top_x,
            'bottom_x': right_wall_middle_x,
            'top_y': self.Ly,
            'bottom_y': self.Ly / 2,  # Middle height
            'width': self.wall_width
        })

        # Add bottom half of hourglass (middle to bottom)
        self.left_wall_segments.append({
            'top_x': left_wall_middle_x,
            'bottom_x': left_wall_bottom_x,
            'top_y': self.Ly / 2,  # Middle height
            'bottom_y': 0,
            'width': self.wall_width
        })

        self.right_wall_segments.append({
            'top_x': right_wall_middle_x,
            'bottom_x': right_wall_bottom_x,
            'top_y': self.Ly / 2,  # Middle height
            'bottom_y': 0,
            'width': self.wall_width
        })

    #############################
    ##### Physics Simulation ####
    #############################
    @staticmethod
    @njit
    def compute_particle_forces_hertzian(x, y, vx, vy, r, N, k, gamma, friction_coef, gravity,
                                    ax_out, ay_out, potential_energy, virial):
        """
        Compute forces between particles using Hertzian contact model (non-linear spring).
        This function is JIT-compiled with Numba for performance.
        """
        # Reset outputs
        for i in range(N):
            ax_out[i] = 0.0
            ay_out[i] = -gravity  # Apply gravity first

        potential_energy[0] = 0.0
        virial[0] = 0.0

        # Loop over all pairs of particles
        for i in range(N - 1):
            for j in range(i + 1, N):
                # Calculate separation vector
                dx = x[i] - x[j]
                dy = y[i] - y[j]

                # Calculate distance
                r_ij = np.sqrt(dx**2 + dy**2)

                # Sum of radii
                r_sum = r[i] + r[j]

                # Only calculate forces if particles overlap
                if r_ij < r_sum:
                    # Overlap distance
                    overlap = r_sum - r_ij

                    # Normalized direction vector
                    if r_ij > 0:  # Avoid division by zero
                        nx = dx / r_ij
                        ny = dy / r_ij

                        # Relative velocity
                        dvx = vx[i] - vx[j]
                        dvy = vy[i] - vy[j]

                        # Normal component of relative velocity
                        v_n = dvx * nx + dvy * ny

                        # Tangential velocity components
                        v_tx = dvx - v_n * nx
                        v_ty = dvy - v_n * ny
                        v_t_mag = np.sqrt(v_tx**2 + v_ty**2)

                        # Hertzian contact force (non-linear spring)
                        normal_force_mag = k * np.sqrt(overlap)

                        # Damping force in normal direction
                        damping_force_mag = gamma * v_n

                        # Total normal force
                        f_n = normal_force_mag - damping_force_mag if v_n < 0 else normal_force_mag

                        # Force components in normal direction
                        f_nx = f_n * nx
                        f_ny = f_n * ny

                        # Tangential (friction) force
                        if v_t_mag > 1e-6:  # Avoid division by zero
                            # Direction of tangential velocity
                            tx = v_tx / v_t_mag
                            ty = v_ty / v_t_mag

                            # Tangential force magnitude (limited by Coulomb friction)
                            f_t = min(friction_coef * abs(f_n), gamma * v_t_mag)

                            # Tangential force components (opposing direction of motion)
                            f_tx = -f_t * tx
                            f_ty = -f_t * ty
                        else:
                            f_tx = 0.0
                            f_ty = 0.0

                        # Total force components
                        fx = f_nx + f_tx
                        fy = f_ny + f_ty

                        # Update accelerations (F = ma, assuming m=1)
                        ax_out[i] += fx
                        ay_out[i] += fy
                        ax_out[j] -= fx
                        ay_out[j] -= fy

                        # Update potential energy (Hertzian, 2/5*k*overlap^(5/2))
                        potential_energy[0] += 0.4 * k * overlap**2.5

                        # Update virial (for pressure calculation)
                        virial[0] += fx * dx + fy * dy

    @staticmethod
    @njit
    def compute_particle_forces_hooke(x, y, vx, vy, r, N, k, gamma, friction_coef, gravity,
                                 ax_out, ay_out, potential_energy, virial):
        """
        Compute forces between particles using Hooke's law (linear spring).
        Using the equation f(r_ij) = -k(R_i + R_j - r_ij)r_ij/|r_ij| + γ(v_ij·r_ij)r_ij/r_ij²
        This function is JIT-compiled with Numba for performance.
        """
        # Reset outputs
        for i in range(N):
            ax_out[i] = 0.0
            ay_out[i] = -gravity  # Apply gravity first

        potential_energy[0] = 0.0
        virial[0] = 0.0

        # Loop over all pairs of particles
        for i in range(N - 1):
            for j in range(i + 1, N):
                # Calculate separation vector
                dx = x[i] - x[j]
                dy = y[i] - y[j]

                # Calculate distance
                r_ij = np.sqrt(dx**2 + dy**2)

                # Sum of radii
                r_sum = r[i] + r[j]

                # Only calculate forces if particles overlap
                if r_ij < r_sum:
                    # Overlap distance
                    overlap = r_sum - r_ij

                    # Unit vector in direction of separation
                    if r_ij > 0:  # Avoid division by zero
                        nx = dx / r_ij
                        ny = dy / r_ij

                        # Relative velocity
                        dvx = vx[i] - vx[j]
                        dvy = vy[i] - vy[j]

                        # Dot product of velocity and direction
                        v_dot_r = dvx * nx + dvy * ny

                        # Spring force component: -k(R_i + R_j - r_ij)(r_ij/|r_ij|)
                        spring_force_mag = k * overlap
                        spring_fx = spring_force_mag * nx
                        spring_fy = spring_force_mag * ny

                        # Damping force: γ(v_ij·r_ij)r_ij/|r_ij|²
                        damping_mag = gamma * v_dot_r
                        damping_fx = damping_mag * nx
                        damping_fy = damping_mag * ny

                        # Total force
                        fx = spring_fx - damping_fx
                        fy = spring_fy - damping_fy

                        # Update accelerations (F = ma, assuming m=1)
                        ax_out[i] += fx
                        ay_out[i] += fy
                        ax_out[j] -= fx
                        ay_out[j] -= fy

                        # Update potential energy (Hookean, 0.5*k*overlap²)
                        potential_energy[0] += 0.5 * k * overlap**2

                        # Update virial (for pressure calculation)
                        virial[0] += fx * dx + fy * dy

    def compute_acceleration(self):
        """Calculate forces and accelerations using the selected contact model"""
        # Arrays to hold output values
        potential_energy = np.zeros(1)
        virial = np.zeros(1)

        # Call the appropriate force calculation method based on contact model
        if self.contact_model == "hertzian":
            self.compute_particle_forces_hertzian(
                self.x, self.y, self.vx, self.vy, self.r, self.N,
                self.k, self.gamma, self.friction_coef, self.gravity,
                self.ax, self.ay, potential_energy, virial
            )
        else:  # Hooke's law
            self.compute_particle_forces_hooke(
                self.x, self.y, self.vx, self.vy, self.r, self.N,
                self.k, self.gamma, self.friction_coef, self.gravity,
                self.ax, self.ay, potential_energy, virial
            )

        # Apply wall constraints
        self.apply_wall_constraints()

        # Update energy and virial
        self.potential_energy = potential_energy[0]
        self.virial = virial[0]

    def handle_wall_segments(self):
        """Handle particle collisions with wall segments using friction"""
        for i in range(self.N):
            # LEFT WALL SEGMENTS
            if self.left_wall_segments:
                for segment in self.left_wall_segments:
                    # Only process if particle is in this segment's y-range
                    if segment['bottom_y'] <= self.y[i] <= segment['top_y']:
                        # Calculate wall x-position at this y-coordinate
                        segment_height = segment['top_y'] - segment['bottom_y']
                        if segment_height > 0:  # Avoid division by zero
                            y_frac = (self.y[i] - segment['bottom_y']) / segment_height
                            wall_x_at_y = segment['bottom_x'] + y_frac * (segment['top_x'] - segment['bottom_x'])

                            # Calculate normal and tangent vectors
                            wall_vec_x = segment['top_x'] - segment['bottom_x']
                            wall_vec_y = segment['top_y'] - segment['bottom_y']
                            wall_length = np.sqrt(wall_vec_x**2 + wall_vec_y**2)

                            if wall_length > 0:  # Avoid division by zero
                                # Normal vector (pointing right, away from wall)
                                normal_x = wall_vec_y / wall_length
                                normal_y = -wall_vec_x / wall_length

                                # Tangent vector (pointing along wall)
                                tangent_x = wall_vec_x / wall_length
                                tangent_y = wall_vec_y / wall_length

                                # Check if particle is overlapping with wall
                                wall_inner_x = wall_x_at_y + segment['width']

                                # Distance from particle center to wall
                                dx = self.x[i] - wall_inner_x

                                # If particle is penetrating the wall
                                if dx < self.r[i]:
                                    # Calculate penetration depth
                                    penetration = self.r[i] - dx

                                    # Decompose velocity into normal and tangential components
                                    v_normal = self.vx[i] * normal_x + self.vy[i] * normal_y
                                    v_tangent = self.vx[i] * tangent_x + self.vy[i] * tangent_y

                                    # Only apply forces if penetrating or moving toward wall
                                    if penetration > 0 or v_normal < 0:
                                        # Use Hertzian or Hooke's law for normal force
                                        if self.contact_model == "hertzian":
                                            normal_force = self.k * np.sqrt(max(0, penetration))
                                        else:
                                            normal_force = self.k * penetration

                                        # Damping in normal direction (only when moving towards wall)
                                        normal_damping = self.gamma * min(0, v_normal)

                                        # Total normal force
                                        total_normal_force = normal_force - normal_damping

                                        # Apply normal force
                                        self.ax[i] += total_normal_force * normal_x
                                        self.ay[i] += total_normal_force * normal_y

                                        # Apply friction force (tangential)
                                        if abs(v_tangent) > 1e-6:
                                            # Friction magnitude (limited by Coulomb law)
                                            friction_mag = min(
                                                self.friction_coef * total_normal_force,
                                                self.gamma * abs(v_tangent)
                                            )

                                            # Apply friction in opposite direction of tangential velocity
                                            friction_dir = -1 if v_tangent > 0 else 1
                                            self.ax[i] += friction_mag * friction_dir * tangent_x
                                            self.ay[i] += friction_mag * friction_dir * tangent_y

                                    # Prevent extreme penetration by moving particle
                                    if penetration > 0.5 * self.r[i]:
                                        self.x[i] = wall_inner_x + self.r[i]

            # RIGHT WALL SEGMENTS
            if self.right_wall_segments:
                for segment in self.right_wall_segments:
                    # Only process if particle is in this segment's y-range
                    if segment['bottom_y'] <= self.y[i] <= segment['top_y']:
                        # Calculate wall x-position at this y-coordinate
                        segment_height = segment['top_y'] - segment['bottom_y']
                        if segment_height > 0:  # Avoid division by zero
                            y_frac = (self.y[i] - segment['bottom_y']) / segment_height
                            wall_x_at_y = segment['bottom_x'] + y_frac * (segment['top_x'] - segment['bottom_x'])

                            # Calculate normal and tangent vectors
                            wall_vec_x = segment['top_x'] - segment['bottom_x']
                            wall_vec_y = segment['top_y'] - segment['bottom_y']
                            wall_length = np.sqrt(wall_vec_x**2 + wall_vec_y**2)

                            if wall_length > 0:  # Avoid division by zero
                                # Normal vector (pointing left, away from wall)
                                normal_x = -wall_vec_y / wall_length
                                normal_y = wall_vec_x / wall_length

                                # Tangent vector (pointing along wall)
                                tangent_x = wall_vec_x / wall_length
                                tangent_y = wall_vec_y / wall_length

                                # Check if particle is overlapping with wall
                                wall_inner_x = wall_x_at_y - segment['width']

                                # Distance from particle center to wall
                                dx = wall_inner_x - self.x[i]

                                # If particle is penetrating the wall
                                if dx < self.r[i]:
                                    # Calculate penetration depth
                                    penetration = self.r[i] - dx

                                    # Decompose velocity into normal and tangential components
                                    v_normal = self.vx[i] * normal_x + self.vy[i] * normal_y
                                    v_tangent = self.vx[i] * tangent_x + self.vy[i] * tangent_y

                                    # Only apply forces if penetrating or moving toward wall
                                    if penetration > 0 or v_normal < 0:
                                        # Use Hertzian or Hooke's law for normal force
                                        if self.contact_model == "hertzian":
                                            normal_force = self.k * np.sqrt(max(0, penetration))
                                        else:
                                            normal_force = self.k * penetration

                                        # Damping in normal direction (only when moving towards wall)
                                        normal_damping = self.gamma * min(0, v_normal)

                                        # Total normal force
                                        total_normal_force = normal_force - normal_damping

                                        # Apply normal force
                                        self.ax[i] += total_normal_force * normal_x
                                        self.ay[i] += total_normal_force * normal_y

                                        # Apply friction force (tangential)
                                        if abs(v_tangent) > 1e-6:
                                            # Friction magnitude (limited by Coulomb law)
                                            friction_mag = min(
                                                self.friction_coef * total_normal_force,
                                                self.gamma * abs(v_tangent)
                                            )

                                            # Apply friction in opposite direction of tangential velocity
                                            friction_dir = -1 if v_tangent > 0 else 1
                                            self.ax[i] += friction_mag * friction_dir * tangent_x
                                            self.ay[i] += friction_mag * friction_dir * tangent_y

                                    # Prevent extreme penetration by moving particle
                                    if penetration > 0.5 * self.r[i]:
                                        self.x[i] = wall_inner_x - self.r[i]

    def apply_wall_constraints(self):
        """Apply forces and constraints from box walls"""
        # Handle bottom wall special case
        for i in range(self.N):
            # Bottom wall - apply forces or respawn
            if self.y[i] < self.r[i]:
                if self.respawn_particles:
                    # Respawn the particle back to the top
                    self.respawn_particle(i)
                else:
                    # Apply normal forces to bounce off bottom
                    penetration = self.r[i] - self.y[i]

                    if penetration > 0:
                        # Normal force based on contact model
                        if self.contact_model == "hertzian":
                            normal_force = self.k * np.sqrt(penetration)
                        else:
                            normal_force = self.k * penetration

                        # Damping force (only when moving toward the wall)
                        normal_damping = self.gamma * min(0, self.vy[i])

                        # Total normal force
                        total_normal_force = normal_force - normal_damping

                        # Apply normal force
                        self.ay[i] += total_normal_force

                        # Apply friction force (if moving horizontally)
                        if abs(self.vx[i]) > 1e-6:
                            # Friction magnitude (limited by Coulomb friction)
                            friction_mag = min(
                                self.friction_coef * total_normal_force,
                                self.gamma * abs(self.vx[i])
                            )

                            # Apply friction in opposite direction of motion
                            friction_dir = -1 if self.vx[i] > 0 else 1
                            self.ax[i] += friction_mag * friction_dir

                        # Prevent extreme penetration
                        if penetration > 0.5 * self.r[i]:
                            self.y[i] = self.r[i]
                            # Reduce vertical velocity with coefficient of restitution
                            if self.vy[i] < 0:
                                self.vy[i] = -self.vy[i] * self.restitution_coef

        # Handle wall segments for the hourglass shape
        self.handle_wall_segments()

    def step(self):
        """Perform one time step using the Velocity Verlet algorithm"""
        # First half of velocity update
        self.vx += 0.5 * self.dt * self.ax
        self.vy += 0.5 * self.dt * self.ay

        # Update positions
        self.x += self.dt * self.vx
        self.y += self.dt * self.vy

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

    ################################
    ##### Metrics Calculations #####
    ################################
    def compute_metrics(self):
        """
        Calculate energy, temperature, and pressure
        """
        self.compute_energy()
        self.compute_temperature()
        self.compute_pressure()

    def compute_energy(self):
        """Calculate the kinetic energy and total energy"""
        self.kinetic_energy = 0.5 * np.sum(self.vx**2 + self.vy**2)
        self.total_energy = self.kinetic_energy + self.potential_energy

    def compute_temperature(self):
        """Calculate temperature based on average kinetic energy per particle"""
        if self.N > 0:
            self.temperature = self.kinetic_energy / self.N

    def compute_pressure(self):
        """Calculate pressure using the virial equation"""
        if self.N > 0:
            volume = self.Lx * self.Ly
            self.pressure = (self.N * self.temperature + 0.5 * self.virial) / volume

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
            'radii': self.r.copy(),
            'temperature': self.temperature,
            'pressure': self.pressure,
            'potential_energy': self.potential_energy,
            'kinetic_energy': self.kinetic_energy,
            'total_energy': self.total_energy
        }
        self.snapshots.append(snapshot)

    def get_respawn_stats(self):
        """Return statistics about particle respawning"""
        if not self.respawn_particles:
            return {"enabled": False, "total_respawns": 0}

        return {
            "enabled": True,
            "total_respawns": self.total_respawns,
            "particles_respawned": np.sum(self.respawn_count > 0),
            "max_respawns_per_particle": np.max(self.respawn_count),
            "avg_respawns_per_particle": np.mean(self.respawn_count)
        }

    #############################
    ##### Data Analysis #########
    #############################
    def calculate_flow_rate(self, time_window=1.0, y_threshold=None):
        """
        Calculate the flow rate of particles through a horizontal line

        Parameters:
        -----------
        time_window : float
            Time window over which to calculate the flow rate
        y_threshold : float
            Y-coordinate of the horizontal line (defaults to the neck position)

        Returns:
        --------
        Dict with time points and corresponding flow rates
        """
        if y_threshold is None:
            y_threshold = self.Ly / 2  # Default to the neck position

        # Only use snapshots with enough time resolution
        if not self.snapshots:
            return {"times": [], "flow_rates": []}

        # Track particles that have passed through the threshold
        particles_passed = set()
        times = []
        counts = []
        flow_rates = []

        # Start with the first snapshot
        previous_time = self.snapshots[0]['time']
        previous_count = 0

        # Count particles crossing the threshold in each snapshot
        for snapshot in self.snapshots:
            current_time = snapshot['time']
            positions = snapshot['positions']

            # Count particles below the threshold
            for i in range(self.N):
                if positions[i, 1] < y_threshold and i not in particles_passed:
                    particles_passed.add(i)

            # Record time and cumulative count
            times.append(current_time)
            counts.append(len(particles_passed))

            # Calculate flow rate for time windows
            if current_time - previous_time >= time_window:
                flow_rate = (len(particles_passed) - previous_count) / (current_time - previous_time)
                flow_rates.append((current_time, flow_rate))

                # Update previous values
                previous_time = current_time
                previous_count = len(particles_passed)

        return {
            "times": [t for t, _ in flow_rates],
            "flow_rates": [r for _, r in flow_rates],
            "cumulative_times": times,
            "cumulative_counts": counts
        }
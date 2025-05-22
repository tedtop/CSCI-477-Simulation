import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
import time
from IPython.display import HTML

# Using Numba for acceleration
from numba import njit
NUMBA_AVAILABLE = True

class RespawningHourglassSimulator:
    def __init__(self, N=200, Lx=20.0, Ly=20.0, temperature=0.0, dt=0.005, gravity=5.0,
                 particle_radius=0.5, k=20.0, gamma=2.0, neck_width=3.0,
                 wall_width=0.5, friction_coef=0.5, restitution_coef=0.3,
                 snapshot_interval=10, respawn_particles=False):
        """
        Initialize the granular hourglass simulator.

        Parameters:
        -----------
        N : int
            Number of particles
        Lx, Ly : float
            Dimensions of the simulation box
        temperature : float
            Initial temperature (random motion)
        dt : float
            Time step size
        gravity : float
            Gravitational acceleration
        particle_radius : float
            Radius of particles
        k : float
            Spring constant for particle interactions
        gamma : float
            Damping coefficient
        neck_width : float
            Width of the hourglass neck
        wall_width : float
            Thickness of the walls
        friction_coef : float
            Coefficient of friction (0-1)
        restitution_coef : float
            Coefficient of restitution for wall collisions (0-1)
        snapshot_interval : int
            Interval for taking snapshots
        respawn_particles : bool
            Whether to respawn particles that reach the bottom back to the top
        """
        # Basic simulation parameters
        self.N = N
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.gravity = gravity
        self.particle_radius = particle_radius
        self.k = k  # Higher spring constant for stiffer contacts
        self.gamma = gamma  # Higher damping for energy dissipation
        self.friction_coef = friction_coef  # Friction coefficient
        self.restitution_coef = restitution_coef  # Coefficient of restitution
        self.neck_width = neck_width
        self.wall_width = wall_width
        self.snapshot_interval = snapshot_interval
        self.initial_temperature = temperature
        self.respawn_particles = respawn_particles

        # Arrays for positions, radii, velocities, and accelerations
        self.x = np.zeros(N)
        self.y = np.zeros(N)
        self.r = np.ones(N) * particle_radius
        self.vx = np.zeros(N)
        self.vy = np.zeros(N)
        self.ax = np.zeros(N)
        self.ay = np.zeros(N)

        # Respawn tracking
        self.respawn_count = np.zeros(N, dtype=int)  # Track how many times each particle respawns
        self.total_respawns = 0  # Track total respawns across all particles

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

        # Wall segments for hourglass
        self.left_wall_segments = []
        self.right_wall_segments = []

        # Print status
        print("Numba acceleration is available and will be used.")
        if self.respawn_particles:
            print(f"Particle respawning enabled. Particles will respawn when they hit the bottom.")

    ################################
    ##### Particle Arrangement #####
    ################################
    def initialize_particles(self):
        """Initialize particles in a grid formation"""
        # Set proper radii
        self.r = np.ones(self.N) * self.particle_radius

        # Create a more spaced out grid of particles
        particles_per_row = int(np.sqrt(self.N))
        rows = (self.N + particles_per_row - 1) // particles_per_row

        # Calculate spacing based on radius to minimize overlap
        x_spacing = self.Lx / (particles_per_row + 1)
        y_spacing = 2.5 * self.r[0]  # Extra vertical spacing

        count = 0
        for i in range(rows):
            for j in range(particles_per_row):
                if count < self.N:
                    self.x[count] = (j + 1) * x_spacing
                    self.y[count] = self.Ly - (i + 1) * y_spacing

                    # Add small random offsets to avoid perfect alignment
                    self.x[count] += np.random.uniform(-0.1, 0.1) * self.r[0]

                    count += 1

        # Zero initial velocities - let gravity do the work
        self.vx = np.zeros(self.N)
        self.vy = np.zeros(self.N)

        # Initialize system
        self.set_velocities()
        self.compute_acceleration()
        self.compute_metrics()

    def initialize_random_falling_particles(self):
        """Initialize particles randomly at the top of the hourglass"""
        # Set proper radii
        self.r = np.ones(self.N) * self.particle_radius

        # Get wall parameters to avoid placing particles inside walls
        wall_width = self.wall_width
        left_top_x = 0.0
        right_top_x = self.Lx

        # If we have wall segments defined, get the top positions
        if self.left_wall_segments:
            top_segment = self.left_wall_segments[0]  # Get top segment
            left_top_x = top_segment['top_x'] + wall_width  # Add width to get inner edge

        if self.right_wall_segments:
            top_segment = self.right_wall_segments[0]  # Get top segment
            right_top_x = top_segment['top_x'] - wall_width  # Subtract width

        # Area where particles can be placed
        valid_width = right_top_x - left_top_x

        # Define the top area height (25% of total height)
        top_area_height = self.Ly * 0.25
        bottom_y = self.Ly - top_area_height

        # Minimum distance between particles
        min_distance = 2.2 * self.particle_radius

        # Initialize all particles
        for i in range(self.N):
            position_valid = False
            attempt = 0
            max_attempts = 100

            while not position_valid and attempt < max_attempts:
                # Random position within valid area
                x = left_top_x + np.random.random() * valid_width
                y = bottom_y + np.random.random() * top_area_height

                # Check distance from previously placed particles
                position_valid = True
                for j in range(i):
                    dx = x - self.x[j]
                    dy = y - self.y[j]
                    distance = np.sqrt(dx**2 + dy**2)

                    if distance < min_distance:
                        position_valid = False
                        break

                attempt += 1

            # Use the position whether valid or not after max attempts
            self.x[i] = x
            self.y[i] = y

        # Apply small random velocities to help break symmetry
        self.vx = np.random.uniform(-0.01, 0.01, self.N)
        self.vy = np.zeros(self.N)

        # Initialize system
        self.compute_acceleration()
        self.compute_metrics()

    def respawn_particle(self, index):
        """Respawn a particle at the top of the hourglass"""
        # Get valid area for respawning
        wall_width = self.wall_width
        left_top_x = 0.0
        right_top_x = self.Lx

        # If we have wall segments defined, get the top positions
        if self.left_wall_segments:
            top_segment = self.left_wall_segments[0]  # Get top segment
            left_top_x = top_segment['top_x'] + wall_width  # Add width to get inner edge

        if self.right_wall_segments:
            top_segment = self.right_wall_segments[0]  # Get top segment
            right_top_x = top_segment['top_x'] - wall_width  # Subtract width

        # Valid width where particles can be placed
        valid_width = right_top_x - left_top_x

        # Define top area for respawning (5-15% from the top)
        top_margin = 0.05 * self.Ly  # 5% from the top
        bottom_margin = 0.15 * self.Ly  # 15% from the top

        # Calculate new position
        new_x = left_top_x + np.random.random() * valid_width
        new_y = self.Ly - top_margin - np.random.random() * (bottom_margin - top_margin)

        # Update particle position and reset velocity
        self.x[index] = new_x
        self.y[index] = new_y
        self.vx[index] = np.random.uniform(-0.01, 0.01)
        self.vy[index] = 0.0

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

    #############################
    ##### Hourglass Shapes ######
    #############################
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
    def compute_particle_forces(x, y, vx, vy, r, N, k, gamma, friction_coef, gravity,
                              ax_out, ay_out, potential_energy, virial):
        """
        Optimized particle force calculation using Numba
        Hertzian contact model with tangential friction
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

    def compute_acceleration(self):
        """Calculate forces and accelerations using Hertzian contact model with friction"""
        # Arrays to hold output values
        potential_energy = np.zeros(1)
        virial = np.zeros(1)

        # Call optimized force calculation
        self.compute_particle_forces(
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
                                        # Hertzian normal force
                                        normal_force = self.k * np.sqrt(max(0, penetration))

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
                                        # Hertzian normal force
                                        normal_force = self.k * np.sqrt(max(0, penetration))

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
                        # Normal force (Hertzian contact)
                        normal_force = self.k * np.sqrt(penetration)

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
            self.temperature = np.sum(self.vx**2 + self.vy**2) / (2 * self.N)

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
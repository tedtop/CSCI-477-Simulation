import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


class LJParticles:
    def __init__(self, N=64, L=10.0, temperature=1.0, dt=0.01, seed=None):
        # Initialize system parameters
        self.N = N
        self.L = L  # Square box (L×L)
        self.dt = dt
        self.initial_temperature = temperature

        # Arrays for positions, velocities, and accelerations
        self.x = np.zeros(N)
        self.y = np.zeros(N)
        self.vx = np.zeros(N)
        self.vy = np.zeros(N)
        self.ax = np.zeros(N)
        self.ay = np.zeros(N)

        # Tracking
        self.time = 0.0
        self.potential_energy = 0.0
        self.kinetic_energy = 0.0
        self.total_energy = 0.0

        self.seed = seed

        # Initialize particles on a square lattice with 8×8 grid
        self.set_square_lattice()

        # Initialize velocities to get the desired temperature
        self.set_velocities()

        # Calculate initial accelerations and energy
        self.compute_acceleration()
        self.compute_energy()

    def set_square_lattice(self):
        """Place particles on a square lattice"""
        nx = int(np.sqrt(self.N))  # 8 particles per row
        ny = nx  # 8 particles per column

        # Calculate spacing between particles
        dx = self.L / nx
        dy = self.L / ny

        # Place particles
        count = 0
        for ix in range(nx):
            for iy in range(ny):
                if count < self.N:
                    self.x[count] = dx * (ix + 0.5)  # Center in the cell
                    self.y[count] = dy * (iy + 0.5)
                    count += 1

    def set_velocities(self):
        """Set random velocities with zero center-of-mass momentum"""
        # Generate random velocities

        if self.seed is not None:
            np.random.seed(self.seed)

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
                dx = self.pbc_separation(self.x[i] - self.x[j], self.L)
                dy = self.pbc_separation(self.y[i] - self.y[j], self.L)

                # Square distance
                r2 = dx**2 + dy**2

                # Compute Lennard-Jones force and potential
                if r2 > 0:  # Avoid division by zero
                    r2i = 1.0 / r2
                    r6i = r2i**3
                    f_mag = 48.0 * r2i * r6i * (r6i - 0.5)

                    fx = f_mag * dx
                    fy = f_mag * dy

                    # Update accelerations
                    self.ax[i] += fx
                    self.ay[i] += fy
                    self.ax[j] -= fx
                    self.ay[j] -= fy

                    # Update potential energy
                    self.potential_energy += 4.0 * r6i * (r6i - 1.0)
                    self.virial += dx * fx + dy * fy

    def compute_energy(self):
        """Calculate the kinetic energy and total energy"""
        # KE = (1/2)mv²
        # v² = vx² + vy² (the squared velocity is the sum of squared components)
        self.kinetic_energy = 0.5 * np.sum(self.vx**2 + self.vy**2)
        self.total_energy = self.kinetic_energy + self.potential_energy

    def get_temperature(self):
        """Calculate the current temperature based on equation (8.5)"""
        # For 2D system with N particles
        return np.sum(self.vx**2 + self.vy**2) / (2 * self.N)

    def step(self):
        """Perform one time step using the Velocity Verlet algorithm"""
        # First half of velocity update
        self.vx += 0.5 * self.dt * self.ax
        self.vy += 0.5 * self.dt * self.ay

        # Update positions
        self.x += self.dt * self.vx
        self.y += self.dt * self.vy

        # Apply periodic boundary conditions
        for i in range(self.N):
            self.x[i] = self.pbc_position(self.x[i], self.L)
            self.y[i] = self.pbc_position(self.y[i], self.L)

        # Calculate new accelerations
        self.compute_acceleration()

        # Second half of velocity update
        self.vx += 0.5 * self.dt * self.ax
        self.vy += 0.5 * self.dt * self.ay

        # Update energy and time
        self.compute_energy()
        self.time += self.dt

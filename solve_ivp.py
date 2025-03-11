import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Define the n-body ODE function compatible with scipy.integrate
def n_body_scipy(t, y, params):
    """
    ODE function for the n-body problem for scipy.integrate.solve_ivp

    Parameters:
    t: time (scalar)
    y: state vector [positions, velocities]
    params: dictionary of parameters

    Returns:
    dydt: derivatives [velocities, accelerations]
    """
    m = params['m']
    dimensions = params['dimensions']
    fix_first = params.get('fix_first', False)

    n_bodies = len(m)
    pos = y[:n_bodies*dimensions].reshape(n_bodies, dimensions)
    vel = y[n_bodies*dimensions:].reshape(n_bodies, dimensions)

    # Initialize acceleration array
    acc = np.zeros_like(pos)

    # Calculate accelerations
    for i in range(n_bodies):
        if fix_first and i == 0:
            continue
        for j in range(n_bodies):
            if i != j:
                r_ij = pos[j] - pos[i]
                r_norm = np.linalg.norm(r_ij)
                # Gravitational force: F = G*m_i*m_j/r^2 * unit_vector
                # Here G=1 for simplicity
                acc[i] += m[j] * r_ij / (r_norm**3)

    # Flatten velocities and accelerations for output
    dydt = np.concatenate([vel.flatten(), acc.flatten()])
    return dydt

# Set up the problem
dt = 0.001
t_span = [0, 20]
y0 = np.array([0, 0, 2, 0, -1, 0, 0, 0, 0.62, 0, 0, -1])  # x0, y0, x1, y1, x2, y2, vx0, vy0, vx1, vy1, vx2, vy2
p_he = {'m': np.array([2, 1, 1]), 'dimensions': 2, 'fix_first': True}

# Solve using scipy.integrate.solve_ivp
sol = integrate.solve_ivp(
    n_body_scipy,
    t_span,
    y0,
    method='RK45',  # This is similar to your EulerRichardson
    args=(p_he,),
    rtol=1e-8,
    atol=1e-8,
    max_step=0.05,
    first_step=dt,
    dense_output=True  # For smooth interpolation in animation
)

# Extract solution
t = sol.t
y = sol.y.T  # Transpose to match your original shape

# Animation function (assuming show_anim is defined elsewhere)
def show_anim(t, y, p_he=None):
    """
    Create animation of the n-body solution
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True)

    n_bodies = p_he['m'].size
    dimensions = p_he['dimensions']

    # Initialize lines and points for each body
    lines = []
    points = []

    colors = ['blue', 'red', 'green']

    for i in range(n_bodies):
        line, = ax.plot([], [], '-', alpha=0.3, color=colors[i % len(colors)])
        point, = ax.plot([], [], 'o', color=colors[i % len(colors)])
        lines.append(line)
        points.append(point)

    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        return lines + points

    def animate(i):
        frame = min(i * 10, len(t) - 1)  # Skip frames for faster animation

        for j in range(n_bodies):
            # Get position history up to current frame
            pos_x = y[:frame, j * dimensions]
            pos_y = y[:frame, j * dimensions + 1]

            # Current position
            current_x = y[frame, j * dimensions]
            current_y = y[frame, j * dimensions + 1]

            # Update trail and current position
            lines[j].set_data(pos_x, pos_y)
            points[j].set_data(current_x, current_y)

        return lines + points

    anim = FuncAnimation(fig, animate, frames=len(t)//10,
                         init_func=init, blit=True, interval=30)

    return anim

# Create animation
anim = show_anim(t, y, p_he)
HTML(anim.to_html5_video())
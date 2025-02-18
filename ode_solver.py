import numpy as np
from scipy.stats import linregress
from matplotlib import pyplot as plt

def Euler(dt, f, t, y, args):
    return y + f(t, y, *args) * dt

def EulerCromer(dt, f, t, y, args):
    y_end = y + f(t, y, *args) * dt  # First Euler step to get the endpoint
    return y + f(t + dt, y_end, *args) * dt  # Second step: update using the endpoint

def EulerRichardson(dt, f, t, y, args):
    y_mid = y + f(t, y, *args) * (dt / 2)  # First Euler step to get the midpoint
    y_next = y + f(t + dt / 2, y_mid, *args) * dt  # Second step: update using the midpoint
    return y_next

def solve_ode(f, tspan, y0, method=Euler, *args, **options):
    """
    Given a function f that returns derivatives,
        dy / dt = f(t, y)
    and an initial state:
        y(tspan[0]) = y0
    This function will return the set of intermediate states of y
    from t0 (tspan[0]) to tf (tspan[1]).

    INPUTS:
    f - function handle that returns derivatives of y at time t.
        Can accept additional parameters passed via *args.
    tspan - iterable containing [t0, tf], initial and final times.
    y0 - Initial state of the system (numpy array).
    method - Integration method (Euler, Euler-Cromer, or Euler-Richardson).
    *args - Additional parameters for function f.
    **options - Optional keyword arguments:
        first_step - Initial time step (default: 0.01).

    OUTPUTS:
    t, y - Numpy arrays of time steps and corresponding y values.
    """
    t0, tf = tspan[0], tspan[1]
    dt = options.get("first_step", 0.01)

    y = [y0]
    t = [t0]

    while t[-1] < tf:
        current_t, current_y = t[-1], y[-1]
        next_y = method(dt, f, current_t, current_y, *args)

        y.append(next_y)
        t.append(current_t + dt)

    return np.array(t), np.array(y)

def error_scale(steps, errors, plot=True):
    """
    INPUTS:
    steps - Vector of steps (ideally log-spaced).
    errors - Vector of errors (e.g., |y - y_analytic|).
    plot - Boolean flag to plot results.
    """
    steps_log = np.log10(steps)
    errors_log = np.log10(errors)

    slope, intercept, _, _, _ = linregress(steps_log, errors_log)

    if plot:
        plt.scatter(steps_log, errors_log, label="Error Data")
        plt.plot(steps_log, slope * steps_log + intercept, "r--", label=f"Fit: α = {slope:.2f}")
        plt.xlabel("log10(Δt)")
        plt.ylabel("log10(Error)")
        plt.title("Error Scaling")
        plt.legend()
        plt.show()

    return slope

def error(y, y_a):
    return np.sum((y - y_a) ** 2)  # L2 norm

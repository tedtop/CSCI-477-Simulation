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

def RungeKutta(dt, f, t, y, args):
    """
    h: time step (dt)
    f: function to integrate
    t: current time
    y: current state
    args: additional arguments to f
    """
    k1 = f(t, y, *args)
    k2 = f(t + dt / 2, y + k1 * dt / 2, *args)
    k3 = f(t + dt / 2, y + k2 * dt / 2, *args)
    k4 = f(t + dt, y + k3 * dt, *args)

    return y + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

def RK45(dt, f, t, y, args, tol=1e-5, h_max=0.1, h_min=1e-6):
    """
    One step of the Dormand-Prince 5(4) method with adaptive stepping.
    This function advances the solution by approximately dt, but may take
    multiple internal sub-steps to maintain accuracy.

    Parameters:
    -----------
    dt : float
        Desired (approximate) step size
    f : callable
        Function that defines the ODE system
    t : float
        Current time
    y : array_like
        Current state vector
    args : tuple
        Additional arguments for f
    tol : float
        Error tolerance for adaptive stepping
    h_max : float
        Maximum internal step size
    h_min : float
        Minimum internal step size

    Returns:
    --------
    next_y : ndarray
        The state at approximately t + dt
    """
    # Target end time
    t_end = t + dt

    # Current state and time
    t_current = t
    y_current = np.array(y, dtype=float)

    # Dormand-Prince 5(4) coefficients
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])

    # a_ij coefficients (rows correspond to i, columns to j)
    a = np.zeros((7, 6))
    # i=2, j=1 (row 1, col 0 in zero-indexed array)
    a[1, 0] = 1/5
    # i=3, j=1,2
    a[2, 0] = 3/40
    a[2, 1] = 9/40
    # i=4, j=1,2,3
    a[3, 0] = 44/45
    a[3, 1] = -56/15
    a[3, 2] = 32/9
    # i=5, j=1,2,3,4
    a[4, 0] = 19372/6561
    a[4, 1] = -25360/2187
    a[4, 2] = 64448/6561
    a[4, 3] = -212/729
    # i=6, j=1,2,3,4,5
    a[5, 0] = 9017/3168
    a[5, 1] = -355/33
    a[5, 2] = 46732/5247
    a[5, 3] = 49/176
    a[5, 4] = -5103/18656
    # i=7, j=1,2,3,4,5,6
    a[6, 0] = 35/384
    a[6, 1] = 0
    a[6, 2] = 500/1113
    a[6, 3] = 125/192
    a[6, 4] = -2187/6784
    a[6, 5] = 11/84

    # 5th order method coefficients (b_i)
    b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])

    # 4th order method coefficients (b_i*)
    b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

    # Initial step size for the adaptive process
    h = min(dt/2, h_max)  # Start with a conservative step

    # Take internal steps until we reach t_end
    while t_current < t_end:
        # Ensure we don't overshoot t_end
        if t_current + h > t_end:
            h = t_end - t_current

        # Calculate the k values for the Dormand-Prince method
        k = np.zeros((7, len(y_current)))
        k[0] = h * np.array(f(t_current, y_current, *args))

        for i in range(1, 7):
            y_temp = y_current.copy()
            for j in range(i):
                y_temp += a[i, j] * k[j]
            k[i] = h * np.array(f(t_current + c[i] * h, y_temp, *args))

        # Calculate 5th and 4th order solutions
        y5 = y_current.copy()
        y4 = y_current.copy()

        for i in range(7):
            y5 += b5[i] * k[i]
            y4 += b4[i] * k[i]

        # Calculate the error estimate
        # Use relative error with a small absolute error floor to avoid division by zero
        error_vec = np.abs(y5 - y4) / (1.0 + np.abs(y_current))
        error = np.max(error_vec) / h  # Take the maximum error component

        # Determine if the step is acceptable
        if error <= tol:
            # Accept this step
            t_current += h
            y_current = y5.copy()  # Use the 5th order solution

            # If we've reached the end, we're done
            if abs(t_current - t_end) < 1e-10:
                break

        # Calculate new step size using PI controller formula
        if error < 1e-15:  # Avoid division by zero or very small numbers
            h_new = h_max
        else:
            # Standard step size control formula
            h_new = 0.9 * h * (tol / error)**0.2

        # Apply step size constraints
        h = min(max(h_new, h_min), h_max)

    # Return the final state
    return y_current

def solve_ode(f, tspan, y0, method=Euler, *args, **options):
    """
    Given a function f that returns derivatives,
    dy / dt = f(t, y)
    and an inital state:
    y(tspan[0]) = y0

    This function will return the set of intermediate states of y
    from t0 (tspan[0]) to tf (tspan[1])

    The function is called as follows:

    INPUTS

    f - the function handle to the function that returns derivatives of the
        vector y at time t. The function can also accept parameters that are
        passed via *args, eg f(t,y,g) could accept the acceleration due to gravity.

    tspan - a indexed data type that has [t0 tf] as its two members.
            t0 is the initial time
            tf is the final time

    y0 - The initial state of the system, must be passed as a numpy array.

    method - The method of integrating the ODEs. This can be Euler, EulerCromer,
             EulerRichardson, RungeKutta, or RK45.

    *args - a tuple containing as many additional parameters as you would like for
            the function handle f.

    **options - a dictionary containing all the keywords that might be used to control
                function behavior. For now, there is only one:

                first_step - the initial time step for the simulation.
                tol - tolerance for adaptive methods (RK45)
                h_max - maximum step size for adaptive methods
                h_min - minimum step size for adaptive methods

    OUTPUTS

    t,y

    The returned states will be in the form of a numpy array
    t containing the times the ODEs were solved at and an array
    y with shape tsteps,N_y where tsteps is the number of steps
    and N_y is the number of equations. Observe this makes plotting simple:

    plt.plot(t,y[:,0])

    would plot positions.
    """
    t0, tf = tspan[0], tspan[1]
    dt = options.get("first_step", 0.01)

    y = [y0]
    t = [t0]

    while t[-1] < tf:
        current_t, current_y = t[-1], y[-1]

        # Call the appropriate integration method with the correct parameters
        if method.__name__ == "RK45":
            # For RK45, pass the additional adaptive stepping parameters
            next_y = method(dt, f, current_t, current_y, args,
                           tol=options.get("tol", 1e-5),
                           h_max=options.get("h_max", 0.1),
                           h_min=options.get("h_min", 1e-6))
        else:
            # For standard methods, use the regular parameter set
            next_y = method(dt, f, current_t, current_y, args)

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

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

    method - The method of integrating the ODEs. This week will be one of Euler,
             Euler-Cromer, or Euler-Richardson

    *args - a tuple containing as many additional parameters as you would like for
            the function handle f.

    **options - a dictionary containing all the keywords that might be used to control
                function behavior. For now, there is only one:

                first_step - the initial time step for the simulation.


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
        next_y = method(dt, f, current_t, current_y, *args) # this shouldn't unpack args (*)

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

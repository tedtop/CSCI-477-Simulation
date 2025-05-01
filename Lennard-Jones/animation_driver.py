#!/usr/bin/python
from  particles import *
from particleInitialize import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import sys, time

dt = 0.01       # Time step
STEPS = 3500    # Total steps before animation repeats
TRANSIENT = 500  # An initial period that is discarded

# Forces between particles
f = LennardJonesForce()
# Create paricle object storing particles
p = Particles(f)
# Add particles, set initial conditions 
particleInitialize(p,'p8.11b',f)

# Integrator
integrate = VerletIntegrator(dt)

# Begin animated plot
#fig,ax = plt.subplots(1,2)
fig, axd = plt.subplot_mosaic(
    [["atoms", "atoms","totalE"],
     ["temp", "pressure","Cp"]],
    layout="constrained",
    width_ratios=[1.0,1.0,1.0],
)
atoms = []
for i in range(p.x.size):
   c = Circle((p.x[i],p.y[i]),radius = .5*2**(1/6.),facecolor='orange',edgecolor='red',alpha=.5)
   axd["atoms"].add_patch(c)
   atoms.append(c)

axd["atoms"].set_xlim((-p.Lx/2,p.Lx/2))
axd["atoms"].set_ylim((-p.Ly/2,p.Ly/2))
axd["atoms"].grid()
axd["atoms"].set_aspect('equal')
axd["atoms"].set_title('Atomic Positions')
axd["atoms"].set_xlabel('Horizontal Distance')
axd["atoms"].set_ylabel('Vertical Distance')

# Discard a transient period and compute the limits on plots:
for i in range(TRANSIENT+1):
    integrate(f,p)

# Time Series Plots 
temp, = axd["temp"].plot(p.time,p.temperature,'r')
axd["temp"].set_title("Temperature vs. Time")
axd["temp"].grid()

pressure, = axd["pressure"].plot(p.time,p.pressure,'b')
axd["pressure"].set_title("Pressure vs. Time")
axd["pressure"].grid()

totalE, = axd["totalE"].plot(p.time,p.total_energy,'k')
axd["totalE"].set_title("Total Energy vs. Time")
axd["totalE"].grid()

Cp, = axd["Cp"].plot(p.time,p.heat_capacity,'y')
axd["Cp"].set_title("Heat Capacity vs. Time")
axd["Cp"].grid()



def update(frame):
    # Do the physics updates
    integrate(f,p) # Step forward in time

    # slow, steady increase in KE
    if frame%50==0 and\
            2 * p.mean_temperature[TRANSIENT] > p.mean_temperature[-1]:
        p.vx *= 1.05
        p.vy *= 1.05

    # After increasing KE, make the box bigger to create a phase change
    if 2 * p.mean_temperature[TRANSIENT] < p.mean_temperature[-1]\
            and frame % 50 == 0\
            and p.Lx < 8*1.1:
        p.Lx *= 1.01
        p.Ly *= 1.01
        axd["atoms"].relim()
        axd["atoms"].autoscale_view()

    # Plotting:
    axd["temp"].set_xlim(p.time[TRANSIENT],p.time[-1])
    temp.set_xdata(p.time[TRANSIENT:])
    temp.set_ydata(p.mean_temperature[TRANSIENT:])
    axd["temp"].relim()
    axd["temp"].autoscale_view()

    axd["pressure"].set_xlim(p.time[TRANSIENT],p.time[-1])
    pressure.set_xdata(p.time[TRANSIENT:])
    pressure.set_ydata(p.mean_pressure[TRANSIENT:])
    axd["pressure"].relim()
    axd["pressure"].autoscale_view()

    axd["totalE"].set_xlim(p.time[TRANSIENT],p.time[-1])
    totalE.set_xdata(p.time[TRANSIENT:])
    totalE.set_ydata(p.mean_energy[TRANSIENT:])
    axd["totalE"].relim()
    axd["totalE"].autoscale_view()

    axd["Cp"].set_xlim(p.time[TRANSIENT],p.time[-1])
    Cp.set_xdata(p.time[TRANSIENT:])
    Cp.set_ydata(p.heat_capacity[TRANSIENT:])
    axd["Cp"].relim()
    axd["Cp"].autoscale_view()


    for i,a in enumerate(atoms):
        a.center = (p.x[i],p.y[i])

    # This sequence of returned lists is critical, and mysterious
    return [Cp] + [pressure] + [temp] + [totalE] + atoms

# Create the animation
anim = FuncAnimation(fig, update, frames=STEPS, interval=20, blit=True)
plt.show()

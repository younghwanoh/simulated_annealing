
# ackley multimodal function
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import basinhopping
import numpy as np

fig = plt.figure()
fig.set_size_inches(12, 7)
mpl.rcParams.update({'font.size': 16})

trace_all = []
th_trace = 0
# objective function
def objective(input_vector):
    x, y = input_vector[0], input_vector[1]
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

def fitness(input_vector):
    global trace_all
    x, y = input_vector[0], input_vector[1]
    trace_all.append((x,y))
    return objective(input_vector)

# filter large jump with l2 distance
def sample(trace_all, th):
    trace = []
    prev_x = prev_y = -9999999
    for x,y in trace_all:
        diff_x = x - prev_x
        diff_y = y - prev_y
        jmp_distance = np.sqrt(diff_x**2+diff_y**2)
        assert jmp_distance >= 0, jmp_distance
        if jmp_distance > th and prev_x != -9999999:
            trace.append(((prev_x, prev_y),(x,y)))
        prev_x, prev_y = x, y
    return trace

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective([x,y])
# 3d-plot
# figure = plt.figure()
# axis = figure.add_subplot(111, projection='3d')
# axis.plot_surface(x, y, results, cmap='jet')

# 2d-plat with heatmap
plt.contourf(x,y,results)
plt.colorbar()

global_min = np.min(results)

# define the bounds on the search
bounds = [[r_min, r_max], [r_min, r_max]]

# perform the simulated annealing search
initial_guess = np.random.uniform(low=r_min, high=r_max, size=2)
result = basinhopping(fitness, initial_guess, niter=100)
# trace = sample(trace_all, th_trace)

# plot everything
pos = np.linspace(0, 1, len(trace_all))
for idx, elem in enumerate(trace_all):
    plt.plot(*elem, color=mpl.colormaps["hot"](pos[idx]), markersize=3, marker="o", linewidth=0)
plt.plot(*result.x, marker="*", color="yellow", linewidth=0, markersize=20, mew=1.5, mec="black")
plt.ylim([r_min,r_max])
plt.xlim([r_min,r_max])
plt.savefig("mesh_basin.png")



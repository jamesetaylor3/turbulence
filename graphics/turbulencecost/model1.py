import matplotlib.pyplot as plt
import numpy as np
import math

bound_options = {
    "Light" : (0.64, 0.36, 0.17, 0.12),
    "Medium" : (0.79, 0.44, 0.20, 0.16),
    "Heavy" : (0.96, 0.54, 0.24, 0.18)
}
def f(x, bounds):
    if bounds[0] < x:
        return 100
    elif bounds[1] < x <= bounds[0]:
        bounds_range = bounds[0] - bounds[1]
        percent_in = (x - bounds[1]) / bounds_range
        return 33.333 * (1 + percent_in)
    elif bounds[2] < x <= bounds[1]:
        bounds_range = bounds[1] - bounds[2]
        percent_in = (x - bounds[2]) / bounds_range
        return 16.666 * (1 + percent_in)
    elif bounds[3] < x <= bounds[2]:
        bounds_range = bounds[2] - bounds[3]
        percent_in = (x - bounds[3]) / bounds_range
        return 8.333 * (1 + percent_in)
    else:
        return 0

for size, bounds in bound_options.iteritems():


    edr = np.arange(0, 1, 0.005)
    c = [f(val, bounds) for val in edr]

    plt.plot(edr, c)

    plt.title('Turbulence Cost For ' + size + ' Plane')
    plt.xlabel('EDR')
    plt.ylabel('Cost')

    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import math

bound_options = {
    "Light" : (0.64, 0.12),
    "Medium" : (0.79, 0.16),
    "Heavy" : (0.96, 0.18)
}
def f(x):
    a = 100 / (math.e - 1)
    b = -a
    return a*math.exp(x) + b


edr = np.arange(0, 1, 0.005)
c = [f(val) for val in edr]

plt.plot(edr, c)

plt.title('Turbulence Cost For Plane')
plt.xlabel('EDR')
plt.ylabel('Cost')

plt.show()

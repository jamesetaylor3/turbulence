import matplotlib.pyplot as plt
import numpy as np
import math

t_f = 13
c_f = 100
c_i = 2

a = (c_f - c_i) / (math.e ** t_f  - 1)
b = c_i - a

t = np.arange(0, 13, 0.065)
c = a * math.e ** t + b

plt.plot(t, c)

plt.title('Fuel Cost Over Time')
plt.xlabel('Time')
plt.ylabel('Fuel Cost')
plt.legend(['c = a*e^x+b'], loc='upper left')

plt.show()

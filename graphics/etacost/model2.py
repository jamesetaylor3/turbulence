import matplotlib.pyplot as plt
import numpy as np
import math

c_f = 5

t = np.arange(-0.5, 0.5, 0.0025)
a = (c_f * math.sqrt(math.e)) / (math.e - 1)
b = -a / math.sqrt(math.e)
c = a * math.e ** t + b

plt.plot(t, c)
plt.title('ETA Cost vs Time Gained in ETA')
plt.xlabel('Time Gained in ETA (min)')
plt.ylabel('ETA Cost')
plt.legend(['c = a*t + b'], loc='upper left')
plt.axis([-0.5, 0.5, 0, 5])
plt.show()

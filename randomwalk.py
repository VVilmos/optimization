import time
import matplotlib
import matplotlib.pyplot as plt
import random
import math

import matplotlib.pyplot as plt
import numpy

# Turning on interactive mode
plt.ion()

# Creating a sample figure and axes
fig, ax = plt.subplots()
ax.set_xlim(0, 600)
ax.set_ylim(0, 600)

x = numpy.array([300, 150])
y = numpy.array([400, 200])

print(plt.get_backend())

def randomwalk(list_x, list_y):
    for u in range(2):
        phi = random.uniform(0, 2*math.pi)
        speed = random.randint(1, 50)
        a = speed*math.sin(phi)
        b = speed*math.cos(phi)
        list_x[u] += a
        list_y[u] += b
        numpy.clip(x, 0, 600)
        numpy.clip(y, 0, 600)


for i in range(100):
    ax.scatter(x, y)
    plt.draw()
    plt.pause(0.5)
    ax.clear()
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 600)
    randomwalk(x,y)


plt.ioff()
plt.show()

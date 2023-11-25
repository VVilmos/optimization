import matplotlib.pyplot as plt
import random
import numpy as np

from matplotlib import cm

# 3D plotting example
#generating the dataset


#7 different values for switching cost
sw_values = [0, 50, 100, 150, 200, 250, 300]
#18 different values for handover cost
ho_values = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]


ho = []
#generating the dataset
for i in range(7):
    for j in range(18):
        ho.append(ho_values[j])

sw = []
for i in range(7):
    for j in range(18):
        sw.append(sw_values[i])


print(len(sw))
print(len(ho))

n=[]
start = [280, 240, 200, 160, 120, 80, 40]

for i in range(7):
    diff = start[i] // 18
    n.append(start[i])
    for j in range(17):
        n.append(n[len(n)-1] -diff)

print(len(n))


#plotting the dataset

plt.style.use('_mpl-gallery')

# Make data
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(ho_values, sw_values)
Z = np.array(n).reshape(7,18)
print(Y)
print(Z)

# Plot the surface with the calculated n values
def surf_plot(ho_values, sw_values, n):
    X,Y = np.meshgrid(ho_values, sw_values)
    Z = np.array(n).reshape(7,18)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, rcount=18, ccount=7, cmap=cm.coolwarm)
    ax.set_xlabel('Handover Cost')
    ax.set_ylabel('Switching Cost')
    ax.set_zlabel('Number of')
    plt.show()



# Plot the surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, rcount=18, ccount=7, cmap=cm.coolwarm)


ax.set_xlabel('Handover Cost')
ax.set_ylabel('Switching Cost')
ax.set_zlabel('Number of')

plt.show()

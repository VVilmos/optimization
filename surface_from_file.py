import numpy
import pandas
import matplotlib.pyplot as plt
from matplotlib import cm

x_values = pandas.read_csv('sw_values.csv', header=None).to_numpy()
y_values = pandas.read_csv('ho_values.csv', header=None).to_numpy()
n = pandas.read_csv('ho_result.csv', header=None).to_numpy()


def surf_plot(x_values, y_values, n):

    average_x_values = []
    for i in range(0, len(x_values)-1, 2):
        average_x_values.append((x_values[i]+x_values[i+1])/2)

    print(average_x_values)

    average_y_values = []
    for i in range(0, len(y_values)-1, 2):
        average_y_values.append((y_values[i]+y_values[i+1])/2)

    print(average_y_values)
    X,Y = numpy.meshgrid(x_values, y_values)
    aX, aY = numpy.meshgrid(average_x_values, average_y_values)

    average_Z = numpy.zeros((len(average_y_values),len(average_x_values)))

    Z = numpy.array(n).reshape(len(y_values),len(x_values))

    for j in range(0, len(average_x_values)):
        for i in range(0, len(average_y_values)):
            average_Z[i][j] = (Z[2*i][2*j] + Z[2*i+1][2*j] + Z[2*i][2*j+1] + Z[2*i+1][2*j+1])/4
            '''sum_Z = 0
            for x in range(4):
                 for y in range(4):
                    sum_Z += Z[4*i + x][4*j + y]
            average_Z[i][j] = sum_Z / 16'''



    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #ax.plot_surface(X, Y, Z,cmap=cm.coolwarm)
    ax.plot_surface(aX, aY, average_Z,cmap=cm.coolwarm)
    ax.set_xlabel('Handover Cost')
    ax.set_ylabel('Switching Cost')
    ax.set_zlabel('Number of Handovers')
    plt.show()


surf_plot(x_values, y_values, n)
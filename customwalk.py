from docplex.mp.model import Model
import random
import math
import pandas
import matplotlib
import matplotlib.pyplot as plt
import numpy 


list_x = numpy.array([])
list_y = numpy.array([])

fig, ax = plt.subplots()

#reading the data from the csv file

#Location of users is read from the csv file
b = []
users_x = numpy.array([])
users_y = numpy.array([])
for u in range(1):
    x =  random.uniform(0 , 600)
    y = random.uniform(0 , 600)
    users_x = numpy.append(users_x, x)
    users_y = numpy.append(users_y, y)
    ax.scatter(x, y, c='r', s=25, alpha=1, zorder = 10)

last_speed = []
last_phi = []
#initializing the users' speed and direction with radnom values
def init_user():
    for u in range(90):
        phi = random.uniform(0, 2*math.pi)
        speed = random.randint(1, 70)
        last_speed.append(speed)
        last_phi.append(phi)



#random walk mobility model
#the user moves in the direction of the last movement with 60% probability
def randomwalk(list_x, list_y ):
    for i in range(30):
        for u in range(1):
            phi = random.uniform(last_phi[u] - 0.25*math.pi, last_phi[u] + 0.25*math.pi)
            speed = random.randint(1, 100)
            phi = 0.4*phi + 0.6*last_phi[u]
            speed = 0.4*speed + 0.6*last_speed[u]
            prev_x = list_x[u]
            prev_y = list_y[u]
            list_x[u] += speed*math.cos(phi)
            list_y[u] += speed*math.sin(phi)
            plot(prev_x, prev_y, list_x[u], list_y[u], 0.8-i/40)
            last_speed[u] = speed
            last_phi[u] = phi
            numpy.clip(list_x, 0, 600, out=list_x)
            numpy.clip(list_y,0, 600, out=list_y)
            if (list_x[u] == 0 or list_x[u] == 600 or list_y[u] == 0 or list_y[u] == 590):
                #last_phi[u] = random.uniform(0, 2*math.pi)
                last_phi[u] += last_phi[u] + math.pi




def plot(x1, y1, x2, y2, alfa):
    plt.xlim(0, 600)
    plt.ylim(0, 600)
    ax.arrow(x1 + (x2-x1)*0.1, y1 + (y2-y1)*0.1, (x2-x1) + (x1-x2)*0.3, (y2-y1) + (y1-y2)*0.3, head_width=10, head_length=5, fc='k', ec='k', alpha = alfa, zorder=0)
    ax.scatter(x2,y2, s=25, alpha = alfa+0.1, c='r', zorder = 8, marker = '+')

init_user()
randomwalk(users_x, users_y )
plt.show()
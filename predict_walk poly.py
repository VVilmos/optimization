import random
import math
import matplotlib.pyplot as plt
import numpy 
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

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


prev_x = users_x[0]
prev_y = users_y[0]
last_speed = []
last_phi = []
#initializing the users' speed and direction with radnom values
def init_user():
    for u in range(90):
        phi = random.uniform(0, 2*math.pi)
        speed = random.randint(1, 70)
        last_speed.append(speed)
        last_phi.append(phi)


last_position = [0, 0]
def predict_next(x1, y1, x2, y2, x3, y3):
    poly = PolynomialFeatures(degree=2)
    regressor = linear_model.LinearRegression()
    x_train = numpy.array([x1, x2, x3]).reshape(-1, 1)
    x_train = poly.fit_transform(x_train)
    y_train = numpy.array([y1, y2, y3]).reshape(-1, 1)
    regressor.fit(x_train, y_train)
    next_x = x3 + (x3-x1)/2
    next_x = poly.fit_transform([[next_x]])

    curve_x = numpy.linspace(x1-20, x3, 20)
    curve_x = poly.fit_transform(curve_x.reshape(-1, 1))
    curve_y = regressor.predict(curve_x)
    ax.plot(curve_x, curve_y, c='grey', linewidth=0.5, zorder=1)
    next_y = regressor.predict(next_x)
    return next_y[0]

#random walk mobility model
#the user moves in the direction of the last movement with 60% probability
def randomwalk(list_x, list_y ):
    for i in range(3):
        for u in range(1):
            phi = random.uniform(last_phi[u] - 0.25*math.pi, last_phi[u] + 0.25*math.pi)
            speed = random.randint(1, 100)
            phi = 0.4*phi + 0.6*last_phi[u]
            speed = 0.4*speed + 0.6*last_speed[u]
            if (i == 0):   
                prev_x = list_x[u]
                prev_y = list_y[u]
            pprev_x = prev_x
            pprev_y = prev_y
            prev_x = list_x[u]
            prev_y = list_y[u]
            list_x[u] += speed*math.cos(phi)
            list_y[u] += speed*math.sin(phi)
            last_speed[u] = speed
            last_phi[u] = phi
            numpy.clip(list_x, 0, 600, out=list_x)
            numpy.clip(list_y,0, 600, out=list_y)


            plot(prev_x, prev_y, list_x[u], list_y[u], 0.8-i/40)
            next_y = (predict_next(pprev_x, pprev_y,prev_x, prev_y,  list_x[u], list_y[u]))
            next_x = (list_x[u]-pprev_x) /2 + list_x[u]
            ax.plot([pprev_x, prev_x, next_x], [pprev_y, prev_y, next_y[0]], c='grey', linewidth=0.5, zorder=1)
            ax.scatter(next_x, next_y[0], s=25, alpha = 0.8-i/40, c='b', zorder = 8, marker = '+')


            if (list_x[u] == 0):
                last_phi[u] = random.uniform(0, math.pi)
            if (list_x[u] == 600):
                last_phi[u] = random.uniform(math.pi, 2*math.pi)
            if (list_y[u] == 0):
                last_phi[u] = random.uniform(-0.5*math.pi, 0.5*math.pi)
            if (list_y[u] == 600):
                last_phi[u] = random.uniform(0.5*math.pi, 1.5*math.pi)




def plot(x1, y1, x2, y2, alfa):
    plt.xlim(-30, 630)
    plt.ylim(-30, 630)
   # ax.arrow(x1 + (x2-x1)*0.1, y1 + (y2-y1)*0.1, (x2-x1) + (x1-x2)*0.3, (y2-y1) + (y1-y2)*0.3, head_width=10, head_length=5, fc='k', ec='k', alpha = alfa, zorder=0)
    ax.scatter(x2,y2, s=25, alpha = alfa+0.1, c='r', zorder = 8, marker = 'o')
    #ax.plot([x1, x2], [y1, y2], c='grey', linewidth=0.5, zorder=1)

init_user()
randomwalk(users_x, users_y )

plt.show()
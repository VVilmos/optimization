import random
import math
import matplotlib.pyplot as plt
import numpy 
from sklearn import linear_model

def predict2(X, Y, ax ):

    phi = math.atan2(X[1] - X[0], Y[1] - Y[0])
    vertical = phi < 0.25*math.pi or phi > 1.75*math.pi or (0.75*math.pi < phi and phi < 1.25*math.pi)

    if (vertical):
         X, Y = Y, X

    #fitting the data to a linear regression model
    regressor = linear_model.LinearRegression()
    x_train = numpy.array(X).reshape(-1, 1)
    y_train = numpy.array(Y).reshape(-1, 1)
    regressor.fit(x_train, y_train)

    #predict the next point
    next_x = X[-1] + X[-1] - X[-2]
    predict_x = numpy.array(next_x).reshape(-1, 1)
    predict_y = regressor.predict(predict_x)
    next_y = predict_y[0][0]


    #plotting the line
    line_x = numpy.linspace(X[0]-1, X[-1]+1, 20).reshape(-1, 1)
    line_y = regressor.predict(line_x)
    #ax.plot(line_x.ravel(), line_y.ravel(), c='grey', linewidth=0.5, zorder=1)

    if (vertical):
        next_x, next_y = next_y, next_x
    return next_x, next_y


fig, ax = plt.subplots()
ax.set_xlim(-100, 700)
ax.set_ylim(-100, 700)

#initializing the user position
user_x = random.uniform(0, 600)
user_y = random.uniform(0, 600)
speed = random.randint(1, 70)
phi = random.uniform(0, 2*math.pi)
last_phi = phi
last_speed = speed
ax.scatter(user_x, user_y, c='r', alpha=1, zorder=10)

#random walk mobility model
#the user moves in the direction of the last movement with 60% probability
def randomwalk(start_x, start_y, last_phi, last_speed):
    prev_x = start_x
    prev_y = start_y
    for i in range(100):
            phi = random.uniform(last_phi - 0.25*math.pi, last_phi + 0.25*math.pi)
            speed = random.randint(1, 120)
            phi = 0.4*phi + 0.6*last_phi
            speed = 0.4*speed + 0.6*last_speed
            x = prev_x + speed*math.sin(phi)
            y = prev_y + speed*math.cos(phi)

            #if the user hits the boundary, it is reflected
            if (x < 0):
                phi = random.uniform(0, math.pi)
            if (x > 600):
                phi = random.uniform(math.pi, 2*math.pi)
            if (y < 0):
                phi = random.uniform(-0.5*math.pi, 0.5*math.pi)
            if (y > 600):
                phi = random.uniform(0.5*math.pi, 1.5*math.pi)


            #plotting the user's next position
            ax.scatter(x, y, c='blue', alpha=1, zorder=10, marker='o', s=5)
            ax.plot([prev_x, x], [prev_y, y], c='grey', linewidth=0.5, zorder=1)
            #difference between the predicted and the actual position
            if (i > 0): ax.plot([x, next_x], [y, next_y], c='red', linewidth=0.5, zorder=1)
            next_x, next_y = predict2([prev_x, x], [prev_y, y], ax)
            if (i < 99): ax.scatter(next_x, next_y, c='red', alpha=1, zorder=10, marker='x', s=9)
            plt.draw()
            plt.pause(0.5)

            #update for the next iteration
            last_speed = speed
            last_phi = phi
            prev_x = x
            prev_y = y
randomwalk(user_x, user_y, last_phi, last_speed)

plt.show()
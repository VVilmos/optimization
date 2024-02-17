import random
import math
import matplotlib.pyplot as plt
import numpy 
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

def predict4(X, Y):
    #fitting the data to a polynomial regression model
    poly = PolynomialFeatures(degree=3)
    regressor = linear_model.LinearRegression()
    x_train = numpy.array(X).reshape(-1, 1)
    x_train = poly.fit_transform(x_train)
    y_train = numpy.array(Y).reshape(-1, 1)
    regressor.fit(x_train, y_train)

    #predict the next point
    next_x = X[-1] + (X[-1] - X[-2] + X[-2] - X[-3] + X[-3] - X[-4])/3
    predict_x = numpy.array(next_x).reshape(-1, 1)
    predict_x = poly.fit_transform(predict_x)
    predict_y = regressor.predict(predict_x)
    next_y = predict_y[0][0]
    #plotting the curve
    curve_x = numpy.linspace(X[0]-1, X[-1]+1, 20).reshape(-1, 1)
    curve_x_poly = poly.fit_transform(curve_x)
    curve_y = regressor.predict(curve_x_poly)
    #ax.plot(curve_x.ravel(), curve_y.ravel(), c='green', linewidth=0.5, zorder=1)
    return next_x, next_y

fig, ax = plt.subplots()
ax.set_xlim(-100, 700)
ax.set_ylim(-100, 700)

#initializing the user position
user_x = random.uniform(100, 150)
user_y = random.uniform(0, 600)
speed = random.randint(1, 100)
phi = random.uniform(0, math.pi)
last_phi = phi
last_speed = speed
ax.scatter(user_x, user_y, c='b', alpha=1, s=5, marker='^')

#random walk mobility model
#the user moves in the direction of the last movement with 60% probability
def randomwalk(start_x, start_y, last_phi, last_speed):
    prev_x = start_x
    prev_y = start_y
    pprev_x = start_x
    pprev_y = start_y
    ppprev_x = start_x
    ppprev_y = start_y
    for i in range(10):
            phi = random.uniform(last_phi - 0.25*math.pi, last_phi + 0.25*math.pi)
            speed = random.randint(1, 100)
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
            if (i > 2): ax.plot([x, next_x], [y, next_y], c='red', linewidth=0.5, zorder=1)
            next_x, next_y = predict4([ppprev_x, pprev_x, prev_x, x], [ppprev_y, pprev_y, prev_y, y])
            if (1 < i < 9): 
                 ax.scatter(next_x, next_y, c='red', alpha=1, zorder=10, marker='x', s=9)
            #plt.draw()
            #plt.pause(2)

            #update for the next iteration
            last_speed = speed
            last_phi = phi
            ppprev_x = pprev_x
            ppprev_y = pprev_y
            pprev_x = prev_x
            pprev_y = prev_y
            prev_x = x
            prev_y = y


randomwalk(user_x, user_y, last_phi, last_speed)

plt.show()
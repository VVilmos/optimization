import sklearn.preprocessing
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


list_x = np.array([])
list_y = np.array([])

fig, ax = plt.subplots()

#filling the list with sample data
for i in range(1, 10):
    list_x = np.append(list_x, i)
    list_y = np.append(list_y, 2*i + random.uniform(-50 ,50))

#plotting the points separately
for i in range(9):
    ax.scatter(list_x[i], list_y[i], c='r',  alpha=1, zorder = 10)


def predict3(X, Y):
    #fitting the data to a polynomial regression model
    poly = PolynomialFeatures(degree=2)
    regressor = linear_model.LinearRegression()
    x_train = np.array(X).reshape(-1, 1)
    x_train = poly.fit_transform(x_train)
    y_train = np.array(Y).reshape(-1, 1)
    regressor.fit(x_train, y_train)

    #plotting the curve
    curve_x = np.linspace(X[0]-1, X[-1]+1, 20).reshape(-1, 1)
    curve_x_poly = poly.fit_transform(curve_x)
    curve_y = regressor.predict(curve_x_poly)
    ax.plot(curve_x.ravel(), curve_y.ravel(), c='grey', linewidth=0.5, zorder=1)

predict3(list_x[2:5], list_y[2:5])


def predict2(X, Y):
    #fitting the data to a linear regression model
    regressor = linear_model.LinearRegression()
    x_train = np.array(X).reshape(-1, 1)
    y_train = np.array(Y).reshape(-1, 1)
    regressor.fit(x_train, y_train)

    #plotting the line
    line_x = np.linspace(X[0]-1, X[-1]+1, 20).reshape(-1, 1)
    line_y = regressor.predict(line_x)
    ax.plot(line_x.ravel(), line_y.ravel(), c='blue', linewidth=0.5, zorder=1)

predict2(list_x[1:3], list_y[1:3])

def predict4(X, Y):
    #fitting the data to a polynomial regression model
    poly = PolynomialFeatures(degree=3)
    regressor = linear_model.LinearRegression()
    x_train = np.array(X).reshape(-1, 1)
    x_train = poly.fit_transform(x_train)
    y_train = np.array(Y).reshape(-1, 1)
    regressor.fit(x_train, y_train)

    #plotting the curve
    curve_x = np.linspace(X[0]-1, X[-1]+1, 20).reshape(-1, 1)
    curve_x_poly = poly.fit_transform(curve_x)
    curve_y = regressor.predict(curve_x_poly)
    ax.plot(curve_x.ravel(), curve_y.ravel(), c='green', linewidth=0.5, zorder=1)

predict4(list_x[5:9], list_y[5:9])
    
plt.show()
import pandas
import math
import numpy
import sklearn.tree
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import tree


def supportvector(features, target):
     features_train, features_test, target_train, target_test = sklearn.model_selection.train_test_split(features, target, test_size=0.2, random_state=42)
     regressor = svm.SVR()

     #preproceccing???

     regressor.fit(features_train, target_train)
     target_pred = regressor.predict(features_test)

     print("MSE of SVM: ", mean_squared_error(target_test, target_pred))
    
     return regressor

def randomforest(features, target):
    features_train, features_test, target_train, target_test = sklearn.model_selection.train_test_split(features, target, test_size=0.2, random_state=42)
    regressor = RandomForestRegressor(n_estimators=40, random_state=0)

    #preproceccing???
    regressor.fit(features_train, target_train)
    target_pred = regressor.predict(features_test)
    print("MSE of Random Forest: ", mean_squared_error(target_test, target_pred))
    return regressor

def decicisontree(features, target):
    features_train, features_test, target_train, target_test = sklearn.model_selection.train_test_split(features, target, test_size=0.2, random_state=42)
    regressor = tree.DecisionTreeRegressor()
    #preproceccing???
    regressor.fit(features_train, target_train)
    target_pred = regressor.predict(features_test)
    print("MSE of Decision Tree: ", mean_squared_error(target_test, target_pred))
    return regressor

def lasso(features, target):
    features_train, features_test, target_train, target_test = sklearn.model_selection.train_test_split(features, target, test_size=0.2, random_state=42)
    regressor = sklearn.linear_model.Lasso(alpha=0.1)
    #preproceccing???
    regressor.fit(features_train, target_train)
    target_pred = regressor.predict(features_test)
    print("MSE of Lasso: ", mean_squared_error(target_test, target_pred))
    return regressor

def predict2(X, Y, ):

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

def predict3(X, Y):
    phi = math.atan2(X[-1] - X[-2], Y[-1] - Y[-2])
    vertical = phi < 0.25*math.pi or phi > 1.75*math.pi or (0.75*math.pi < phi and phi < 1.25*math.pi)

    if (not vertical):
            X, Y = Y, X

    #fitting the data to a polynomial regression model
    poly = PolynomialFeatures(degree=2)
    regressor = linear_model.LinearRegression()
    x_train = numpy.array(X).reshape(-1, 1)
    x_train = poly.fit_transform(x_train)
    y_train = numpy.array(Y).reshape(-1, 1)
    regressor.fit(x_train, y_train)

    #predict the next point
    next_x = X[-1] + (X[-1] - X[-2] + X[-2] - X[-3])/2
    predict_x = numpy.array(next_x).reshape(-1, 1)
    predict_x = poly.fit_transform(predict_x)
    predict_y = regressor.predict(predict_x)
    next_y = predict_y[0][0]
    #ax.scatter(next_x, next_y, c='blue', alpha=1, zorder=10)

    #plotting the curve
    curve_x = numpy.linspace(X[0]-1, X[-1]+1, 20).reshape(-1, 1)
    curve_x_poly = poly.fit_transform(curve_x)
    curve_y = regressor.predict(curve_x_poly)
    #ax.plot(curve_x.ravel(), curve_y.ravel(), c='grey', linewidth=0.5, zorder=1)
    
    if (not vertical):
        next_x, next_y = next_y, next_x

    return next_x, next_y

def predict4(X, Y):
    phi = math.atan2(X[-1] - X[-2], Y[-1] - Y[-2])
    vertical = phi < 0.25*math.pi or phi > 1.75*math.pi or (0.75*math.pi < phi and phi < 1.25*math.pi)

    if (vertical):
            X, Y = Y, X

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
    if (vertical):
        next_x, next_y = next_y, next_x
    
    return next_x, next_y




#reading the data
df = pandas.read_csv('train_x.csv')
features = df[['prev_x', 'prev_y', 'phi', 'speed']]
target_x = df['x']
df = pandas.read_csv('train_y.csv')
target_y = df['y']

#training the models
regressor_x = randomforest(features, target_x)
regressor_y = randomforest(features, target_y)
regressor_x2 = decicisontree(features, target_x)
regressor_y2 = decicisontree(features, target_y)
regressor_x3 = lasso(features, target_x)
regressor_y3 = lasso(features, target_y)
regressor_x4 = supportvector(features, target_x)
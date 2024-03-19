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
    regressor = RandomForestRegressor(n_estimators=50, random_state=0)

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

def predict2(X, Y ):

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

    if (vertical):
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
    
    if (vertical):
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

def predict(regressor_x, regressor_y, prev_x, prev_y, phi, speed):
    df = pandas.DataFrame({'prev_x': [prev_x], 'prev_y': [prev_y], 'phi': [phi], 'speed': [speed]})
    next_x = regressor_x.predict(df)
    next_y = regressor_y.predict(df)
    return next_x[0], next_y[0]



def calculate_difference(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def eval(x, y, features, target_x, target_y, total_errors): 

    #training the models
    regressor_x = randomforest(features, target_x)
    regressor_y = randomforest(features, target_y)
    regressor_x2 = decicisontree(features, target_x)
    regressor_y2 = decicisontree(features, target_y)
    regressor_x3 = lasso(features, target_x)
    regressor_y3 = lasso(features, target_y)
    regressor_x4 = supportvector(features, target_x)
    regressor_y4 = supportvector(features, target_y)

    total_path = 0
    errors = {'linear': 0, 'poly3': 0, 'poly4': 0, 'randomforest': 0, 'decisiontree': 0, 'lasso': 0, 'supportvector': 0}
    #total_errors = {'linear': 0, 'poly3': 0, 'poly4': 0, 'randomforest': 0, 'decisiontree': 0, 'lasso': 0, 'supportvector': 0}
    x_pp = {'linear': x[3], 'poly3': x[3], 'poly4': x[3], 'randomforest': x[3], 'decisiontree': x[3], 'lasso': x[3], 'supportvector': x[3]}
    y_pp = {'linear': y[3], 'poly3': y[3], 'poly4': y[3], 'randomforest': y[3], 'decisiontree': y[3], 'lasso': y[3], 'supportvector': y[3]}

    for i in range(3, 8999):
        #calculate the total length of the path
        total_path += math.dist([x[i], y[i]], [x[i+1], y[i+1]])

        phi = math.atan2(x[i] - x[i-1], y[i] - y[i-1])
        speed = math.dist([x[i], y[i]], [x[i-1], y[i-1]])
        x_pred, y_pred = predict(regressor_x, regressor_y, x[i], y[i], phi, speed)
        #calculate the length of prediction path
        total_errors['randomforest'] += math.dist([x_pp['randomforest'], y_pp['randomforest']], [x_pred, y_pred])
        errors['randomforest'] += math.dist([x[i+1], y[i+1]], [x_pred, y_pred])
        x_pp['randomforest'] = x_pred
        y_pp['randomforest'] = y_pred


        x_pred, y_pred = predict(regressor_x2, regressor_y2, x[i], y[i], phi, speed)
        #calculate the length of prediction path
        total_errors['decisiontree'] += math.dist([x_pp['decisiontree'], y_pp['decisiontree']], [x_pred, y_pred])
        errors['decisiontree'] += math.dist([x[i+1], y[i+1]], [x_pred, y_pred])
        x_pp['decisiontree'] = x_pred
        y_pp['decisiontree'] = y_pred

        x_pred, y_pred = predict(regressor_x3, regressor_y3, x[i], y[i], phi, speed)
        errors['lasso'] += math.dist([x[i+1], y[i+1]], [x_pred, y_pred])
        total_errors['lasso'] += math.dist([x_pp['lasso'], y_pp['lasso']], [x_pred, y_pred])
        x_pp['lasso'] = x_pred
        y_pp['lasso'] = y_pred

        x_pred, y_pred = predict(regressor_x4, regressor_y4, x[i], y[i], phi, speed)
        errors['supportvector'] += math.dist([x[i+1], y[i+1]], [x_pred, y_pred])
        total_errors['supportvector'] += math.dist([x_pp['supportvector'], y_pp['supportvector']], [x_pred, y_pred])
        x_pp['supportvector'] = x_pred
        y_pp['supportvector'] = y_pred

        x_pred, y_pred = predict2(x[i-1:i+1], y[i-1:i+1])
        errors['linear'] += math.dist([x[i+1], y[i+1]], [x_pred, y_pred])
        total_errors['linear'] += math.dist([x_pp['linear'], y_pp['linear']], [x_pred, y_pred])
        x_pp['linear'] = x_pred
        y_pp['linear'] = y_pred


        x_pred, y_pred = predict3(x[i-2:i+1], y[i-2:i+1])
        errors['poly3'] += math.dist([x[i+1], y[i+1]], [x_pred, y_pred])
        total_errors['poly3'] += math.dist([x_pp['poly3'], y_pp['poly3']], [x_pred, y_pred])
        x_pp['poly3'] = x_pred
        y_pp['poly3'] = y_pred

        x_pred, y_pred = predict4(x[i-3:i+1], y[i-3:i+1])
        errors['poly4'] += math.dist([x[i+1], y[i+1]], [x_pred, y_pred])
        total_errors['poly4'] += math.dist([x_pp['poly4'], y_pp['poly4']], [x_pred, y_pred])
        x_pp['poly4'] = x_pred
        y_pp['poly4'] = y_pred


    print("Linear: ", errors['linear'])
    print("Poly3: ", errors['poly3'])
    print("Poly4: ", errors['poly4'])
    print("Random Forest: ", errors['randomforest'])
    print("Decision Tree: ", errors['decisiontree'])
    print("Lasso: ", errors['lasso'])
    print("Support Vector: ", errors['supportvector'])
    return total_path, total_errors, errors

total_errors = {'linear': 0, 'poly3': 0, 'poly4': 0, 'randomforest': 0, 'decisiontree': 0, 'lasso': 0, 'supportvector': 0}
errors = {'linear': 0, 'poly3': 0, 'poly4': 0, 'randomforest': 0, 'decisiontree': 0, 'lasso': 0, 'supportvector': 0}
output = pandas.DataFrame(errors, index=[0])
percentages = pandas.DataFrame(total_errors, index=[0])

for i in {50, 100, 150}:
     print(i)

for i in {50, 100, 150}:
    #reading the data
    df = pandas.read_csv(f"train_x_{i}.csv")
    features = df[['prev_x', 'prev_y', 'phi', 'speed']]
    target_x = df['x']
    df = pandas.read_csv(f'train_y_{i}.csv')
    target_y = df['y']

    #reading the data from the csv file
    df = pandas.read_csv(f'walk_{i}.csv')
    arr = df.to_numpy()
    x = numpy.array(arr[:, 0])
    y = numpy.array(arr[:, 1])
    
    total_path, total_errors, errors = eval(x, y, features, target_x, target_y, total_errors)
    output.loc[i/50] = errors
    p = numpy.empty(0)
    for a in total_errors.keys():
        p = numpy.append(p, (total_path-total_errors[a])/total_path)
    percentages.loc[i/50] = p

output.to_csv('errors.csv', index=False)
percentages.to_csv('percentages.csv', index=False)
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas 
import sklearn.model_selection
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
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


#reading the data
df = pandas.read_csv('train_x.csv')
features = df[['prev_x', 'prev_y', 'phi', 'speed']]
target_x = df['x']
df = pandas.read_csv('train_y.csv')
target_y = df['y']



#predict the next point
def predict(regressor_x, regressor_y, prev_x, prev_y, phi, speed):
    df = pandas.DataFrame({'prev_x': [prev_x], 'prev_y': [prev_y], 'phi': [phi], 'speed': [speed]})
    next_x = regressor_x.predict(df)
    next_y = regressor_y.predict(df)
    return next_x[0], next_y[0]


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
    regressor_x = lasso(features, target_x)
    regressor_y = lasso(features, target_y)
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
            next_x, next_y = predict(regressor_x, regressor_y, x, y, phi, speed)
            if (i < 99): ax.scatter(next_x, next_y, c='red', alpha=1, zorder=10, marker='x', s=9)
            plt.draw()
            plt.pause(0.5)

            #update for the next iteration
            last_speed = speed
            last_phi = phi
            prev_x = x
            prev_y = y
#randomwalk(user_x, user_y, last_phi, last_speed)

plt.show()
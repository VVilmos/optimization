import random
import math
import matplotlib.pyplot as plt
import numpy 
from sklearn  import tree
from sklearn import linear_model
import pandas
import sklearn
from sklearn.metrics import mean_squared_error


#training a decision tree model
#creating regressor for x
tree_regressor_x = tree.DecisionTreeRegressor()

df = pandas.read_csv('train_x.csv')
features = df[['prev_x', 'prev_y', 'phi', 'speed']]
target_x = df['x']
features_train, features_test, target_x_train, target_x_test = sklearn.model_selection.train_test_split(features, target_x, test_size=0.2, random_state=42)

scaler = sklearn.preprocessing.StandardScaler()
print(features_train)


#scaling the targets???
tree_regressor_x.fit(features_train, target_x_train)

#evaluate the model
target_x_pred = tree_regressor_x.predict(features_test)

print(mean_squared_error(target_x_test, target_x_pred))

#create regressor for y
tree_regressor_y = tree.DecisionTreeRegressor()
df = pandas.read_csv('train_y.csv')
features = df[['prev_x', 'prev_y', 'phi', 'speed']]
target_y = df['y']
features_train, features_test, target_y_train, target_y_test = sklearn.model_selection.train_test_split(features, target_y, test_size=0.2, random_state=42)

#i could transform the target values as well, but it is not necessary for decision trees
tree_regressor_y.fit(features_train, target_y_train)

#evaluate the model
target_y_pred = tree_regressor_y.predict(features_test)
print(mean_squared_error(target_y_test, target_y_pred))

#predict the next point

def predict(prev_x, prev_y, phi, speed):
    df = pandas.DataFrame({'prev_x': [prev_x], 'prev_y': [prev_y], 'phi': [phi], 'speed': [speed]})
    next_x = tree_regressor_x.predict(df)
    next_y = tree_regressor_y.predict(df)
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
            next_x, next_y = predict(x, y, phi, speed)
            if (i < 9): ax.scatter(next_x, next_y, c='red', alpha=1, zorder=10, marker='x', s=9)
            plt.draw()
            plt.pause(0.5)

            #update for the next iteration
            last_speed = speed
            last_phi = phi
            prev_x = x
            prev_y = y
randomwalk(user_x, user_y, last_phi, last_speed)

plt.show()
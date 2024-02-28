import math
import sklearn.preprocessing
import numpy
import pandas
import sklearn.model_selection
import sklearn.tree
import sklearn.ensemble


#training a decision tree model
df = pandas.read_csv('train_x.csv')

features = df[['prev_x', 'prev_y', 'phi', 'speed']]
target_x = df['x']

df = pandas.read_csv('train_y.csv')
target_y = df['y'] 

features_train, features_test, target_x_train, target_x_test = sklearn.model_selection.train_test_split(features, target_x, test_size=0.2, random_state=42)
features_train, features_test, target_y_train, target_y_test = sklearn.model_selection.train_test_split(features, target_y, test_size=0.2, random_state=42)
regressor_y = sklearn.ensemble.RandomForestRegressor()
regressor_y.fit(features_train, target_y_train)

pred_y = regressor_y.predict(features_test)
print("MSE of Decision Tree for y: ", sklearn.metrics.mean_squared_error(target_y_test, pred_y))

regressor_x = sklearn.ensemble.RandomForestRegressor()
regressor_x.fit(features_train, target_x_train)

pred_x = regressor_x.predict(features_test)
print("MSE of Decision Tree for x: ", sklearn.metrics.mean_squared_error(target_x_test, pred_x))
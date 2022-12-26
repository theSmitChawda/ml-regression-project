# -*- coding: utf-8 -*-
"""
Created on Mon Oct 7 15:40:45 2022

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model

from sklearn.model_selection import cross_val_score

df = pd.read_csv("cars.csv") # Use Pandas to read in the car dataset "cars.csv"
new_df = df[['Engine Information.Number of Forward Gears','Fuel Information.City mpg','Engine Information.Engine Statistics.Horsepower','Engine Information.Engine Statistics.Torque']]
mask = np.random.rand(len(new_df)) < 0.7
train = new_df[mask]
test = new_df[~mask]

model = linear_model.LinearRegression()

var_k_group = 4

train_x = np.asanyarray(train[['Gears','Torque']])
test_x = np.asanyarray(test[['Gears','Torque']])

train_y = np.asanyarray(train[['CityMPG']])
test_y_hat = 0
test_y = np.asanyarray(test[['CityMPG']])

model.fit (train_x, train_y)
print ('Beta0: ', model.intercept_)

for i in range(0, len(train_x)):
    if i == len(train_x[0]):
        break
    print("Beta",(i+1),": ",model.coef_[0][i])

test_y_hat = model.predict(test_x)
print("\n-------------------------------")
print("Predicted outcome: ",test_y_hat)    
print("\nMean Absolute Error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Mean Squared Error: %.2f" % np.mean((test_y_hat - test_y) ** 2))

print("\n-------------------------------")
scores = cross_val_score(model, train_x, train_y, cv=var_k_group, scoring='r2')
print('The mean R^2 score using K-fold is: %.3f' % np.mean(scores))
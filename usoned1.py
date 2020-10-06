# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 00:30:19 2020

@author: ardag
"""



import numpy as np
np.set_printoptions(precision=2)
 

import random

def f(x):
    res = -(x* 10) + 45
    error = res * random.uniform(-0.18, 0.18) 
    return res + error

values = []
''' now using f(x) we are going to create 300 values. '''
for i in range(0, 50):
    x = random.uniform(1, 50)
    x = np.round(x)
    y = f(x)
    y = np.round(y)
    values.append((x, y))

from sklearn import linear_model
regr = linear_model.LinearRegression(
        fit_intercept=True, 
        normalize=False, 
        copy_X=True, 
        n_jobs=26)
''' split the values into two series 
    instead a list of tuples  '''
x, y = zip(*values)
max_x = max(x)
min_x = min(x)
cut_value = 10

''' split the values in train data and test data. '''
train_data_X = map(lambda x: [x], list(x[:-cut_value]))
train_data_X = np.array(list(train_data_X))
train_data_Y = list(y[:-cut_value])
train_data_Y = np.array(list(train_data_Y))

test_data_X = map(lambda x: [x], list(x[-cut_value:]))
test_data_X = np.array(list(test_data_X))
test_data_Y = list(y[-cut_value:])
test_data_Y = np.array(list(test_data_Y))

''' feed the linear regression with the train data 
    to obtain a model. '''
regr.fit(train_data_X, train_data_Y)

''' check that the coeffients are the expected ones. '''
''' y = mx + b '''
m = regr.coef_[0]
b = regr.intercept_
print(' y = {0} * x + {1}'.format(m, b))

import matplotlib.pyplot as plt

''' now we are going to plot the points 
    and the model obtained '''
plt.scatter(train_data_X, train_data_Y, color='blue')
plt.scatter(test_data_X,  test_data_Y,  color='green')
plt.plot([min_x, max_x], [b, m*max_x + b], 'r')
plt.title('Fitted linear regression (blue: train, green: test)', fontsize=16)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)

import numpy as np
''' Mean squared error '''
print("Mean squared error: %.2f" % np.mean((
        regr.predict(test_data_X) - test_data_Y) ** 2))
''' Variance : 1 is perfect prediction '''
print('Variance score: %.2f' % regr.score(
        test_data_X, test_data_Y))
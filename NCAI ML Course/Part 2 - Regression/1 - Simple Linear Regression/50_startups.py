# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 21:40:12 2020

@author: Rafay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Importing the dataset
dataset = pd.read_csv('P:/Machine-Learning-Course-NCAI/assignment_2_data/50_Startups.csv')
############# NEW YORK #####################################################################################

N = dataset.loc[dataset.State=='New York', :]
n_y = N.iloc[:, -1].values
q =np.arange(17)
n_x = q.reshape(-1, 1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
n_x_train, n_x_test, n_y_train, n_y_test = train_test_split(n_x, n_y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(n_x_train, n_y_train)

# predicting and Visualising the linear results
Y_pred = regressor.predict(n_x_test)
plt.scatter(n_x_train, n_y_train, color = 'red')
plt.plot(n_x_train, regressor.predict(n_x_train), color = 'yellow')
plt.title('Profit of startups in New York (Linear Regression)')
plt.xlabel('New York')
plt.ylabel('Profit / $')
plt.show()
# since data is not linear so we ove towards polynomial regression
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
n_x_poly = poly_reg.fit_transform(n_x)
poly_reg.fit(n_x_poly, n_y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(n_x_poly, n_y)

# Visualising the Polynomial Regression results
plt.scatter(n_x, n_y, color = 'red')
plt.plot(n_x, lin_reg_2.predict(poly_reg.fit_transform(n_x)), color = 'blue')
plt.title('Profit of startups in New York (Polynomial Regression)')
plt.xlabel('New York')
plt.ylabel('Profit / $')
plt.show()

print('Profit of startups in NYC (20):')
print(regressor.predict([[20]]))
######################################################################################################################

###########  CALIFORNIA  #########################################################################################
C = dataset.loc[dataset.State=='California', :]
cali_y = C.iloc[:, -1].values
r =np.arange(17)
cali_x = r.reshape(-1, 1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
cali_x_train, cali_x_test, cali_y_train, cali_y_test = train_test_split(cali_x, cali_y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor2 = LinearRegression()
regressor2.fit(cali_x_train, cali_y_train)

# Predicting the Test set results
Y_pred2 = regressor2.predict(cali_x_test)

#visualising the dataset and linear regression
plt.scatter(cali_x_train, cali_y_train, color = 'green')
plt.plot(cali_x_train, regressor.predict(cali_x_train), color = 'yellow')
plt.title('Profit of startups in California (Linear Regression)')
plt.xlabel('California')
plt.ylabel('Profit / $')
plt.show()

# since data is not linear so we ove towards polynomial regression
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree = 6)
cali_x_poly = poly_reg2.fit_transform(cali_x)
poly_reg.fit(cali_x_poly, cali_y)

lin_reg_3 = LinearRegression()
lin_reg_3.fit(cali_x_poly, cali_y)

# Visualising the Polynomial Regression results
plt.scatter(cali_x, cali_y, color = 'green')
plt.plot(cali_x, lin_reg_3.predict(poly_reg2.fit_transform(cali_x)), color = 'blue')
plt.title('Profit of startups in California (Polynomial Regression)')
plt.xlabel('California')
plt.ylabel('Profit / $')
plt.show()

print('Profit of startups in California (20):')
print(regressor2.predict([[20]]))

################################################################################
####################   FLORIDA  #####################################3
F = dataset.loc[dataset.State=='Florida', :]
fl_y = F.iloc[:, -1].values
w =np.arange(16)
fl_x = w.reshape(-1, 1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
fl_x_train, fl_x_test, fl_y_train, fl_y_test = train_test_split(fl_x, fl_y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor3 = LinearRegression()
regressor3.fit(fl_x_train, fl_y_train)

# Predicting and visualisisng the Test set results
Y_pred3 = regressor3.predict(fl_x_test)

plt.scatter(fl_x_train, fl_y_train, color = 'purple')
plt.plot(fl_x_train, regressor3.predict(fl_x_train), color = 'yellow')
plt.title('Profit of startups in Florida (Linear Regression)')
plt.xlabel('Florida')
plt.ylabel('Profit / $')
plt.show()

# since data is not linear so we ove towards polynomial regression
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg4 = PolynomialFeatures(degree = 5)
fl_x_poly = poly_reg4.fit_transform(fl_x)
poly_reg.fit(fl_x_poly, fl_y)

lin_reg_4 = LinearRegression()
lin_reg_4.fit(fl_x_poly, fl_y)

# Visualising the Polynomial Regression results
plt.scatter(fl_x, fl_y, color = 'purple')
plt.plot(fl_x, lin_reg_4.predict(poly_reg4.fit_transform(fl_x)), color = 'blue')
plt.title('Profit of startups in Florida (Polynomial Regression)')
plt.xlabel('Florida')
plt.ylabel('Profit / $')
plt.show()

print('Profit of startups in Florida (20):')
print(regressor3.predict([[20]]))
print("\n")
################################################################################
#COMPARING PROFITS OF 3 STATES

print('Profit of startups in Florida (11):')
print(regressor3.predict([[11]]))
print('Profit of startups in California (11):')
print(regressor2.predict([[11]]))
print('Profit of startups in NYC (11):')
print(regressor.predict([[11]]))
print("\n")

from sklearn import metrics
print('Accuracies of results of the following states')
print("\n")
print('NEW YORK')
print('Mean Absolute Error:', metrics.mean_absolute_error(n_y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(n_y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(n_y_test, Y_pred)))
print("\n")

print('CALIFORNIA')
print('Mean Absolute Error:', metrics.mean_absolute_error(cali_y_test, Y_pred2))
print('Mean Squared Error:', metrics.mean_squared_error(cali_y_test, Y_pred2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(cali_y_test, Y_pred2)))
print("\n")

print('FLORIDA')
print('Mean Absolute Error:', metrics.mean_absolute_error(fl_y_test, Y_pred3))
print('Mean Squared Error:', metrics.mean_squared_error(fl_y_test, Y_pred3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(fl_y_test, Y_pred3)))


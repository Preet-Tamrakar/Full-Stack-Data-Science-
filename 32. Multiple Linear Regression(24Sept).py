# MULTIPLE LINER REGRESSION 

# Importing Libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the Dataset
dataset = pd.read_csv(r"E:\WORK\FSDS\Daily Notes\ML Dataset\Investment.csv")

# Feature Selection
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

x = pd.get_dummies(x, dtype=int)

# Split the dataset into training & testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Fit the Liner Regression Model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#----- We build MLR Model

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

# ------------------ Backward Elimination using OLS ------------------
# Adding a constant column (intercept term) manually for statsmodels
x = np.append(arr = np.full((50,1),42467).astype(int),values=x, axis =1)

import statsmodels.api as sm

#------  Model with all features
x_opt = x[:,[0,1,2,3,4,5]]
# OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

#------ Model after removing feature with highest p-value
import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,5]]
# OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

#------ Further elimination
import statsmodels.api as sm
x_opt = x[:,[0,1,3]]
# OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

#------ Final model with only two predictors
import statsmodels.api as sm
x_opt = x[:,[0,1]]
# OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

#------ Bias and Variance ------------------
# R^2 Score on training data (goodness of fit)

bias =regressor.score(x_train, y_train)
bias

variance = regressor.score(x_train, y_train)
variance
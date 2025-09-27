# SIMPLE LINEAR REGRESSION 

# Importing Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Load the Dataset
dataset = pd.read_csv(r"E:\WORK\FSDS\Daily Notes\ML Dataset\Salary_Data.csv")

# Feature Selection
x = dataset.iloc[:, :-1]   # Years of Experience (Independent Variable)
y = dataset.iloc[:, -1]    # Salary (Dependent Variable)

# Split the dataset into training & testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Fit the Liner Regression Model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Training set results
y_pred = regressor.predict(x_test)

# Compare predicted & Actual salaries from the test set
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

# Visualizing the Testing set results
plt.scatter(x_test, y_test, color='red') # Real salary data (testing)
plt.plot(x_train, regressor.predict(x_train), color='blue') # Regression Line
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Training set results
plt.scatter(x_train, y_train, color='red') # Real salary data (testing)
plt.plot(x_train, regressor.predict(x_train), color='blue') # Regression Line
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Getting Slope and Intercept
m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

# Predicting the Salaries
y_12 = m * 12 + c # Prediction of 12 year exp 
print(y_12)

y_20 = m * 20 + c # Prediction of 20 year exp 
print(y_20)

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)



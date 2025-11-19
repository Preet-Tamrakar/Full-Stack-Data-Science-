import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv(r"E:\WORK\FSDS\Daily Notes\ML Dataset\emp_sal.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Linear Model -- Linear Algo (degree - 1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Linear Regression Visualization
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Linear Regression Graph')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Polynomial Model (Bydefault degree - 2)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)

# Again linear model build with 2nd degree 
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Polynomial Visualization
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Prediction
lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)


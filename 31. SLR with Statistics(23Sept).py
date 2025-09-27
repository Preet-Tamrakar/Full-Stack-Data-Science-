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

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# STATS WITH ML 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
 
# MEAN

# This will give mean 
dataset.mean()

# This will give us mean of that particular column
dataset['Salary'].mean()

#------------------------------------------------------------------------------
 
# MEDIAN

# This will give median of entire dataset
dataset.median()

# This will give us Median of that particular column
dataset['Salary'].median()

#------------------------------------------------------------------------------
 
# MODE

# This will give us Mode of that particular column
dataset.mode()

# This will give mode of that particular column
dataset['Salary'].mode()

#------------------------------------------------------------------------------
 
# VARIANCE

# This will give variance of entire dataset
dataset.var()

# This will give variance of that particular column
dataset['Salary'].var()

#------------------------------------------------------------------------------
 
# STANDARD DEVIATION

# This will give standard deviation of entire dataset
dataset.std()

# This will give standard deviation of that particular column
dataset['Salary'].std()

#------------------------------------------------------------------------------
 
# Cofficient of Variation (CV)

# for calculating cv we have to import a library
from scipy.stats import variation
variation(dataset.values) # This will give cv of entire dataset

# This will give us cv of that particular column
variation(dataset['Salary'])

#------------------------------------------------------------------------------
 
# CORRELATION

# This will give correlation of entire dataset
dataset.corr()

# This will give us correlation between Salary & YearsExperience 
dataset['Salary'].corr(dataset['YearsExperience'])

#------------------------------------------------------------------------------
 
# SKEWNESS

# THis will give skewness of entire dataset
dataset.skew()

# This will give us skewness of that particular column
dataset['Salary'].skew()

#------------------------------------------------------------------------------
 
# STANDARD ERROR

# This will give Standard Error of Entire Dataset
dataset.sem()

# This will give us standard error of that particular column
dataset['Salary'].sem()

#------------------------------------------------------------------------------
 
# Z-score 

# for calculationg Z-score we have to import a library
import scipy.stats as stats

dataset.apply(stats.zscore) # This will give Z-score of entire dataset

# This will give us Z-score of that particular column
stats.zscore(dataset['Salary'])

#------------------------------------------------------------------------------
 
# Sum of Squer Regression (SSR)
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

# Sum of Squer Error (SSE)
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

# Total Sum of Squer (SST)
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

# R2 SQUER
r_square = 1 - (SSR/SST)
r_square

#------------------------------------------------------------------------------
 
# CHECK MODEL PERFORMANCE 

from sklearn.metrics import mean_squared_error
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)

print(train_mse)
print(test_mse)

#------------------------------------------------------------------------------
 
# Save the Trained Model to Disk
import pickle
filename = 'SLR_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as SLR_model.pkl")

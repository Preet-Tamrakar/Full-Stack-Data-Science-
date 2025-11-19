import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv(r"E:\WORK\FSDS\Daily Notes\ML Dataset\emp_sal.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# KNN Regression Model 

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor()
knn_model.fit(x,y)

knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred)

#------------------------------------------------------------

# n_neighbors=6
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=6)
knn_model.fit(x,y)

knn_model_pred = knn_model.predict([[6.5]])
print("n_neighbors=6",knn_model_pred)

#------------------------------------------------------------

# n_neighbors=5
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(x,y)

knn_model_pred = knn_model.predict([[6.5]])
print("n_neighbors=5",knn_model_pred)

#------------------------------------------------------------

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=4)
knn_model.fit(x,y)

knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred) 

#------------------------------------------------------------

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(x,y)

knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred)

#------------------------------------------------------------

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='brute', p=1)
knn_model.fit(x,y)

knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred) 
#------------------------------------------------------------

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=6, weights='distance', algorithm='brute', p=1)
knn_model.fit(x,y)

knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred) 

#------------------------------------------------------------

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='brute', p=2)
knn_model.fit(x,y)

knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred) 
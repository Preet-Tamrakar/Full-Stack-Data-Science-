import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv(r"E:\WORK\FSDS\Daily Notes\ML Dataset\emp_sal.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# SVR Model
from sklearn.svm import SVR
svr_model = SVR()
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print(svr_model_pred) # 130001

#-------------------------------------------------------------

## Sigmoid

# With Sigmoid with degree 3 
svr_model = SVR(kernel='sigmoid',degree=3,gamma='auto',C=10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print("Sigmoid3 :", svr_model_pred) # Predicted: 129999

#-------------------------------------------------------------

# Sigmoid with degree 4 
svr_model = SVR(kernel='sigmoid',degree=4,gamma='auto',C=10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print("Sigmoid4 :", svr_model_pred) # Predicted: 129999

#-------------------------------------------------------------

# With Sigmoid with degree 5
svr_model = SVR(kernel='sigmoid',degree=5,gamma='auto',C=10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print("Sigmoid5 :", svr_model_pred) # Predicted: 129999

#-------------------------------------------------------------

## Poly

# Poly with degree=2
svr_model = SVR(kernel='poly',degree=2,gamma='auto',C=10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print("Poly2 :", svr_model_pred) # Predicted : 162812

#-------------------------------------------------------------

# Poly with degree=3
svr_model = SVR(kernel='poly',degree=3,gamma='auto',C=10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print("Poly3 :", svr_model_pred) # Predicted : 213026

#-------------------------------------------------------------

# Poly with degree=4
svr_model = SVR(kernel='poly',degree=4,gamma='auto',C=10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print("Poly4 :", svr_model_pred) # Predicted : 175705 -- Right Prediction

#-------------------------------------------------------------

# Poly with degree=5
svr_model = SVR(kernel='poly',degree=5,gamma='auto',C=10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print("Poly5 :", svr_model_pred) # Predicted : 160107

#-------------------------------------------------------------

## Rbf

# Rbf with degree=3
svr_model = SVR(kernel='rbf',degree=3,gamma='auto',C=10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print("Rbf3 :", svr_model_pred) # Predicted : 130015
 
#-------------------------------------------------------------

# Rbf with degree=4
svr_model = SVR(kernel='rbf',degree=4,gamma='auto',C=10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print("Rbf4 :", svr_model_pred) # Predicted : 130015

#-------------------------------------------------------------

# Rbf with degree=5
svr_model = SVR(kernel='rbf',degree=5,gamma='auto',C=10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print("Rbf5 :", svr_model_pred) # Predicted : 130015

#-------------------------------------------------------------

## Linear

# With Linear with degree 3 
svr_model = SVR(kernel='linear',degree=3,gamma='auto',C=10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print("linear3 :", svr_model_pred) # Predicted: 130250
 
#-------------------------------------------------------------

# With Linear with degree 4 
svr_model = SVR(kernel='linear',degree=4,gamma='auto',C=10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print("linear4 :", svr_model_pred) # Predicted: 130250

#-------------------------------------------------------------

# With Linear with degree 5
svr_model = SVR(kernel='linear',degree=5,gamma='auto',C=10.0)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
print("linear5 :", svr_model_pred) # Predicted: 130250

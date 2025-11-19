import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv(r"E:\WORK\FSDS\Daily Notes\ML Dataset\emp_sal.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

## Decision Tree

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(x,y)

dt_model_pred = dt_model.predict([[6.5]])
print(dt_model_pred)

###----------------------------------------------------------------------------

## Random Forest
#### Group of Decision Tree is called Random Forest 

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(x,y)

rf_model_pred = rf_model.predict([[6.5]])
print(rf_model_pred)

#------------------------------------------------------------------------------

# random_state=0
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(x,y)

rf_model_pred = rf_model.predict([[6.5]])
print(rf_model_pred)

#------------------------------------------------------------------------------

# n_estimators=27
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=27, random_state=0)
rf_model.fit(x,y)

rf_model_pred = rf_model.predict([[6.5]])
print(rf_model_pred)

#------------------------------------------------------------------------------

# n_estimators=30
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=30, random_state=0)
rf_model.fit(x,y)

rf_model_pred = rf_model.predict([[6.5]])
print(rf_model_pred)

#------------------------------------------------------------------------------

# n_estimators=25
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=25, random_state=0)
rf_model.fit(x,y)

rf_model_pred = rf_model.predict([[6.5]])
print(rf_model_pred)

#------------------------------------------------------------------------------

# n_estimators=20
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=20, random_state=0)
rf_model.fit(x,y)

rf_model_pred = rf_model.predict([[6.5]])
print(rf_model_pred)

#------------------------------------------------------------------------------

# n_estimators=23
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=23, random_state=0)
rf_model.fit(x,y)

rf_model_pred = rf_model.predict([[6.5]])
print(rf_model_pred)

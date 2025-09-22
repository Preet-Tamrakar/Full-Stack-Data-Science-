# IMPORTING THE LIBRARY

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

#-----------------------------

# IMPORT THE DATASET & DEVIDEED DATASET INTO INDEPRNDENT & DEPENDENT

dataset = pd.read_csv(r"E:\WORK\FSDS\Daily Notes\ML Dataset\Data.csv")

x = dataset.iloc[:, :-1].values

y = dataset.iloc[:,3].values

# -------------------------------

from sklearn.impute import SimpleImputer 
imputer = SimpleImputer()

imputer = imputer.fit(x[:,1:3])

x[:, 1:3] = imputer.transform(x[:, 1:3])

# HOW TO ENCODE CATEGORICAL DATA & CREATE A DUMMY VARIABLE 

from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()

x[:,0] = labelencoder_x.fit_transform(x[:,0])

#----------------------------------

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

#---------------------------------

# SPLITTING THE DATASET IN TRAINING SET & TESTING SET

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# If we remove random_state then model not behave as accurate.
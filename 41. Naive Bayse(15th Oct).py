# Naive Bayes

# Importing the Libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv(r"E:\WORK\FSDS\Daily Notes\14th- knn\14th- knn\Social_Network_Ads.csv")

x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training Set & Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#------------------------------------------------------------------------------
#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler()
#------------------------------------------------------------------------------
#from sklearn.preprocessing import Normalizer
#sc = Normalizer()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train the Naive Bayes Model on the Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
 
# This is to get the Models Accuracy 
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print('Accuracy=',ac) 

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr

bias = classifier.score(x_train,y_train)
print('bias=', bias)

variance = classifier.score(x_test,y_test)
print('variance=', variance)

#------------------------------------------------------------------------------
# OUTPUTS
#------------------------------------------------------------------------------
# 1) Bernoulli NB - 
#       standardscaler -- 82.50
#       minmaxscale -- 72.50
#       without scale -- 72.50
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 2) Gausian NB - 
#       without scale -- 92.5
#       with standardscale -- 91.25
#       with minmaxscale -- 91.25
#       with Normalizer -- 72.50
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 2) Multinomial NB - (it won't consider -ve values)
#       with standardscale -- doesn't support 
#       with minmaxscale -- 72.50
#       with Normalizer -- 72.50
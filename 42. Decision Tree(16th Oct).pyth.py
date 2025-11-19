# Decision Tree

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

# Train the Model on the Training Set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=20, min_samples_leaf=8)
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

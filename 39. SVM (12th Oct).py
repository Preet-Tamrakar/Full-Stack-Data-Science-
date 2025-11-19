# SVM (Support Vector Machine)

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv(r"E:\WORK\FSDS\Daily Notes\ML Dataset\logit classification.csv")

x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training Set & Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Training the SVM Model on the Training Set
from sklearn.svm import SVC
classifier_svm = SVC()
classifier_svm.fit(x_train, y_train)

# Predicting the Test Set Result
y_pred = classifier_svm.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

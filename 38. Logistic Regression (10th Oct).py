# Logistic Regression 

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

# Training the Logistic Regression Model on the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# Predicting the Test Set Result
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Model Accuracy 
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("Model Accuracy: ",ac)

# Generating and displaying a detailed classification report 
# (includes Precision, Recall, F1-Score, and Support)
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

# Calculate Bias (Training Score)
bias = classifier.score(x_train, y_train)
print("bias: " ,bias)

# Calculate Variance (Testing Score)
variance = classifier.score(x_test, y_test)
print("variance: ",variance)
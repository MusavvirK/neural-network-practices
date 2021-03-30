# -*- coding: utf-8 -*-


import pandas as pd 
import numpy as np
import sklearn

data = pd.read_csv("heart failure classification dataset.csv")

data.head()

data[['serum_sodium']].head()

from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(data[['serum_sodium']])
data['serum_sodium'] = impute.transform(data[['serum_sodium']])

from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(data[['time']])
data['time'] = impute.transform(data[['time']])

data[['serum_sodium']].head()

data.shape

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
data['gender'] = enc.fit_transform(data['sex'])
print(data[['sex','gender']].head())

data['smoker'] = enc.fit_transform(data['smoking'])
print(data[['smoker','smoking']].head())

data = data.drop(['sex','smoking'], axis = 1)

data.head()

from sklearn.model_selection import train_test_split
X = data.drop("DEATH_EVENT", axis=1)
y = data["DEATH_EVENT"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score_LRC=accuracy_score(y_pred,y_test)
print(score_LRC)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy',random_state=1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
score_DTC=accuracy_score(y_pred,y_test)
print(score_DTC)

data = {'Logistic Regression':score_LRC, 'Decision Tree':score_DTC} 
classifiers = list(data.keys()) 
values = list(data.values()) 
   
fig = plt.figure(figsize = (10, 6)) 
  
# creating the bar plot 
plt.bar(classifiers, values, color ='blue',  
        width = 0.4) 
  
plt.xlabel("Classifier") 
plt.ylabel("Accuracy Score") 
plt.title("Classifier Score Accuracy Comparison") 
plt.show()

# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('glass-source-classification-dataset.csv')

df.sample(10)

df = df.fillna(0)
df["Type"] = df["Type"].astype('category')
df["Type"] = df["Type"].cat.codes
df.sample(10)

df['Ba'] = df['Ba'].map({'Does not exist':0, 'exists': 1})
df['Fe'] = df['Fe'].map({'Does not exist':0, 'exists': 1})
df.tail()

y = df['Type']
df=((df-df.min())/(df.max()-df.min()))
df.sample(10)

predictors = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
X = df[predictors]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn import svm
clfSVM = svm.SVC(decision_function_shape='ovo')
clfSVM.fit(X_train, y_train)
y_pred_SVM = clfSVM.predict(X_test)
score_SVM = accuracy_score(y_pred_SVM, y_test)
print(score_SVM)

from sklearn.ensemble import RandomForestClassifier
clfRFC = RandomForestClassifier(n_estimators=9)
clfRFC.fit(X_train, y_train)
y_pred_RFC = clfRFC.predict(X_test)
score_RFC = accuracy_score(y_pred_RFC, y_test)
print(score_RFC)

from sklearn.neural_network import MLPClassifier
clfMLP = MLPClassifier(hidden_layer_sizes=200, activation='relu', solver='adam', max_iter=1000, alpha=0.0001, batch_size=50, learning_rate_init=0.005, n_iter_no_change=10, tol=1e-4, verbose=True)
clfMLP.fit(X_train, y_train)
y_pred_MLP = clfMLP.predict(X_test)
score_MLP = accuracy_score(y_pred_MLP, y_test)
print(score_MLP)

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca)

clfSVM_pca = svm.SVC(decision_function_shape='ovo')
clfSVM_pca.fit(X_train_pca, y_train)
y_pred_SVM_pca = clfSVM_pca.predict(X_test_pca)
score_SVM_pca_reduced = accuracy_score(y_pred_SVM_pca, y_test)
print(score_SVM_pca_reduced)

clfRFC_pca = RandomForestClassifier(n_estimators=9)
clfRFC_pca.fit(X_train_pca, y_train)
y_pred_RFC_pca = clfRFC_pca.predict(X_test_pca)
score_RFC_pca_reduced = accuracy_score(y_pred_RFC_pca, y_test)
print(score_RFC_pca_reduced)

clfMLP_pca = MLPClassifier(hidden_layer_sizes=200, activation='relu', solver='adam', max_iter=1000, alpha=0.0001, batch_size=50, learning_rate_init=0.005, n_iter_no_change=10, tol=1e-4, verbose=True)
clfMLP_pca.fit(X_train_pca, y_train)
y_pred_MLP_pca = clfMLP_pca.predict(X_test_pca)
score_MLP_pca_reduced = accuracy_score(y_pred_MLP_pca, y_test)
print(score_MLP_pca_reduced)

import matplotlib.pyplot as plt

normalized_original_dataset = [score_SVM, score_RFC, score_MLP]
pca_reduced_dataset = [score_SVM_pca_reduced, score_RFC_pca_reduced, score_MLP_pca_reduced]

fig, ax = plt.subplots(figsize =(14, 8))
bar_width = 0.3
X = np.arange(3)

p1 = plt.bar(X, normalized_original_dataset, bar_width, color='blue', label='original processed dataset')
p2 = plt.bar(X + bar_width, pca_reduced_dataset, bar_width, color='orange', label='pca reduced dataset')

plt.xlabel('Classifiers')
plt.ylabel('Accuracy Scores')
plt.title('Accuracy Score Comparison')
plt.xticks(X + (bar_width/2) , ("SVM", "Random Forest", 
"MLP Neural Network"))
plt.legend()

plt.tight_layout()
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Sun May 17 00:20:29 2020

@author: Ankush
"""


#IMPORTING THE LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
np.random.seed(2)

#IMPORTING THE DATASET
data = pd.read_csv('creditcard.csv')

#DATA EXPLORATION
data.head()

#PRE-PROCESSING
from sklearn.preprocessing import StandardScaler
data['normalisedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'], axis = 1)
data.head()
data = data.drop(['Time'], axis = 1)

#SPLITTING THE FEATURE AND TARGET VARAIBLES
x = data.iloc[:, data.columns!='Class']
y = data.iloc[:, data.columns=='Class']
x.head()
y.head()

#SPLITTING THE DATASET INTO TRAINING AND TESTING
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
X_train.shape
X_test.shape

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rand = RandomForestClassifier(n_estimators = 100)
rand.fit(X_train, y_train.values.ravel())

score = rand.score(X_test, y_test)
y_pred = rand.predict(X_test)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot = True)

#PREDICTING ON WHOLE DATASET
y_pred1 = rand.predict(x)
cm1 = confusion_matrix(y, y_pred1.round())
plt.figure(figsize=(10,7))
sns.heatmap(cm1, annot = True)
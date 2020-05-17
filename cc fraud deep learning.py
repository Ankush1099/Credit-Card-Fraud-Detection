# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:49:55 2020

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
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#DEEP NEURAL NETWORK
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential([
    Dense(units = 16, input_dim = 29, activation = 'relu'),
    Dense(units = 24, activation = 'relu'),
    Dropout(0.5),
    Dense(20, activation = 'relu'),
    Dense(24, activation = 'relu'),
    Dense(1, activation = 'sigmoid'),
])
model.summary()

#TRAINING
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 15, epochs = 5)

score = model.evaluate(X_test, y_test)

#PREDICTING
y_pred = model.predict(X_test)
y_test = pd.DataFrame(y_test)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot = True)

#PREDICTING ON WHOLE DATASET
y_pred1 = model.predict(x)
y_expected = pd.DataFrame(y)
cm1 = confusion_matrix(y_expected, y_pred1.round())
plt.figure(figsize=(10,7))
sns.heatmap(cm1, annot = True)

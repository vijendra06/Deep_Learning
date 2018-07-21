#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:45:07 2018

@author: vijendrasharma
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset= pd.read_csv('Churn_Modelling.csv')
x= dataset.iloc[:, 3:13 ].values
y= dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le1= LabelEncoder()
x[:, 1]= le1.fit_transform(x[:,1])

le2= LabelEncoder()
x[:, 2]= le2.fit_transform(x[:,2])

ohe= OneHotEncoder( categorical_features= [1])
x= ohe.fit_transform(x).toarray()
x= x[:, 1:]

from sklearn.cross_validation import train_test_split
xtrain,xtest, ytrain,ytest= train_test_split( x, y, test_size= 0.2, random_state= 0)

from sklearn.preprocessing import StandardScaler
scx= StandardScaler()
xtrain= scx.fit_transform(xtrain)
xtest= scx.transform(xtest)


import keras
from keras.models import Sequential
from keras.layers import Dense

clf= Sequential()
clf.add( Dense(output_dim=6, init='uniform', activation='relu' , input_dim=11  ))

clf.add( Dense(output_dim=6, init='uniform', activation='relu' ))

clf.add( Dense(output_dim=1, init='uniform', activation='sigmoid' ))

clf.compile( optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])

clf.fit( xtrain, ytrain, batch_size=20, epochs=40)

ypred= clf.predict( xtest)
ypred= (ypred > 0.5)

#acc= clf.score( xtest, ytest)
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(ytest, ypred)

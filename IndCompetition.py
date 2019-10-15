#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:26:14 2019

@author: shuchitakapoor
"""

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from category_encoders import TargetEncoder


#Reading training and testing data
dataset=pd.read_csv('/Users/shuchitakapoor/Downloads/tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv')  #training data
dataset1=pd.read_csv('/Users/shuchitakapoor/Downloads/tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv')  #testing data
dataset2=pd.read_csv('/Users/shuchitakapoor/Downloads/tcdml1920-income-ind/tcd ml 2019-20 income prediction submission file.csv')        #submission file

#Filling missing values in traning dataset using bfill method
dataset = dataset.fillna(method='bfill')

#Removing outliers(income which is more than 5M)
dataset = dataset[dataset['Income in EUR']<5000000]

#Segregating into independent and dependent variables
X = dataset[[ 'Gender','University Degree','Profession', 'Age', 'Country','Size of City','Year of Record']] #Independent variable
y = dataset['Income in EUR']                                                                                #Dependent variable

#Target encoding used to encode categorical features against y(dependent variable)
enc = TargetEncoder(cols=['Gender', 'University Degree', 'Profession','Country']).fit(X, y)
ds = enc.transform(X, y)

#Scaling the two features using standard scaler methond
ss=StandardScaler()
scaler = ss.fit_transform(ds[['Size of City','Year of Record']])
scaler=np.transpose(scaler)
ds['Size of City']=scaler[0]
ds['Year of Record']=scaler[1]

#Splitting the dataset into training(80%) and testing(20%) dataset
X_train, X_test, y_train, y_test = train_test_split(ds, y, test_size=0.2)


#Gradient Boosting Regressor used as a regression algorithm to train the data and predict dependent variable using testing data
params = {'n_estimators': 1000, 'max_depth': 5, 'min_samples_split': 5,
          'learning_rate': 0.1, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#Calculating Root Mean Squared Error
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Filling missing values in testing dataset using bfill method
dataset1 = dataset1.fillna(method='bfill')

#Selecting features to predict income for testing data
X = dataset1[[ 'Gender','University Degree','Profession', 'Age', 'Country','Size of City','Year of Record']]   #Independent variable

#Target encoding used to encode categorical features of testing dataset using the model used on training dataset
ds1 = enc.transform(X)

#Scaling the two features using standard scaler methond
scaler = ss.fit_transform(ds1[['Size of City','Year of Record']])
scaler=np.transpose(scaler)
ds1['Size of City']=scaler[0]
ds1['Year of Record']=scaler[1]

#Predicting data using Gradient Boosting Regressor
y_pred1 = clf.predict(ds1)

#Writing the predicted income in submission file
dataset2['Income'] = y_pred1
dataset2.to_csv('/Users/shuchitakapoor/Downloads/tcdml1920-income-ind/tcd ml 2019-20 income prediction submission file.csv',index=False)
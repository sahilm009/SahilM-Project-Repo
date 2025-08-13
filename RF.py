# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 00:02:34 2022

@author: SMedepalli
"""
# Importing the libraries
import numpy as np # for array operations
import pandas as pd # for working with DataFrames
import requests, io # for HTTP requests and I/O commands
import matplotlib.pyplot as plt # for data visualization


# scikit-learn modules
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.metrics import mean_squared_error # for calculating the cost function
from sklearn.ensemble import RandomForestRegressor # for building the model
import seaborn as sns
from sklearn import metrics

x = pd.read_csv("mlpX.csv")
y = pd.read_csv("mlpY.csv")

# Splitting the dataset into training and testing set (80/20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 28)

def custom_loss()

# Initializing the Random Forest Regression model with 10 decision trees
model = RandomForestRegressor(n_estimators = 10, random_state = 0)

# Fitting the Random Forest Regression model to the data
model.fit(x_train, y_train.values.ravel()) 

# Predicting the target values of the test set
y_pred = model.predict(x_test)

#plt.plot(y_test,y_pred)
sns.regplot(y_test, y_pred, scatter_kws={"s": 100})
#plt.plot(y_pred,y_test)
# RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)),'.3f'))
print("\nRMSE:\n",rmse)
print("\nMSE:\n",metrics.mean_squared_error(y_test,y_pred))


f=open("MLP_DAL_(noconstr)_predictions.csv", "a")
for i in range(len(y_pred)):
   f.write(str(y_pred[i]))
   f.write("\n")
f.close()  


f=open("TestDataFile.csv", "a")
#for i in range(len(y_test)):
   #new_y_test=y_test.transpose() 
   #ny_test=y_test.reshape(-1,1)
f.write(y_test[y_test.columns[0]].to_string())
f.write("\n")
f.close()  

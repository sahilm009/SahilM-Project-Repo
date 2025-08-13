# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 00:41:57 2021
@author: Nicholas
"""

#MARS Implemented with Lasso

import csv
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Collects the Data from the CSV File
x_ = np.array([])
y_ = []
z_ = np.array([])


with open('MarsXdat.csv') as datfile:
    reader=csv.reader(datfile)
    for row_var in reader:
        ##if row_var[0] != 'Cycle no.' and row_var[1] != 'Ratio no.' and row_var[2] != 'Capacitance':
            x_=np.append(x_,float(row_var[0]))
            ##y_.append(float(row_var[1]))
            ##z_.append(float(row_var[2]))
            
            
            

with open('MarsZdat.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        z_=np.append(z_,float(row[0]))
        

#while(x_.size>z_.size):
    #z_=np.append(z_,float(z_[z_.size-1]+1))
   
#x_.reshape(1,-1)
#Creates Training and Testing Data Split
#scaler=MinMaxScaler()
#X_scaled=scaler.fit_transform(x_)
X_train, X_test, z_train, z_test = train_test_split(x_, z_, test_size=0.8, random_state=0) 



X_train=X_train.reshape(-1,1)   
z_train=z_train.reshape(-1,1) 
X_test=X_test.reshape(-1,1)
z_test=z_test.reshape(-1,1)
lasso = Lasso(.001)

#model=Lasso()
lasso.fit(X_train, z_train)

z_pred = lasso.predict(X_test)


#plt.yscale("log")
plt.plot(x_, z_, color = "orange")
plt.plot(z_test,z_pred,'go')
plt.legend(["Real Data", "Predicted"])

#Calculates MSE for Comparing Algorithms

from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(z_test, z_pred)
print('MSE:', MSE)

f=open("MLP_DAL_(noconstr)_predictions.csv", "a")
for i in range(len(z_pred)):
   f.write(str(z_pred[i]))
   f.write("\n")
f.close()   

f=open("TestDataFile.csv", "a")
#for i in range(len(y_test)):
   #new_y_test=y_test.transpose() 
   #ny_test=y_test.reshape(-1,1)
#f.write(z_test[z_test.columns[0]].to_string())
#f.write("\n")
for i in range(len(z_test)):
   f.write(str(z_test[i]))
   f.write("\n")
f.close() 
f.close()  

#MLA Citation
#Sarem, Sarem, and Giridhar. Numbers and Code, 11 Sept. 2018, 
#numbersandcode.com/non-greedy-mars-regression. 
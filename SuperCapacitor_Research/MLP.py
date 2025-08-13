

"""
import numpy as np
from numpy import array
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


#importing and formatting data
data = pd.read_csv("MLPdata.csv", error_bad_lines=False)
train = data.iloc[:150]
test = data.iloc[150:]
training_seq = data.iloc[:150].f.to_numpy()

#constraint
n_steps = 1
# split into samples
X, y = split_sequence(training_seq, n_steps)

# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
y_test = data.iloc[150:].f.to_numpy()


predictions = []
for i in range(1, len(y_test)):
    input_vals = array([y_test[i - 1]])
    input_vals = input_vals.reshape((1, n_steps))
    pred_vals = model.predict(input_vals, verbose=0)
    for values in pred_vals:
        for value in values:
            if value <= y_test[i - 50]:
                predictions.append(value)
            else:
                predictions.append(y_test[i - 50])

numpy_pred_arr = np.asanyarray(predictions)


fig3 = plt.figure(num=3, dpi=300, facecolor='w', edgecolor='k')
plt.plot(train.index, training_seq, label='Training')
plt.plot(test.index, y_test, label='Test')
plt.plot(test.index[1:], numpy_pred_arr, linewidth=.5, label='Predictions')
plt.legend(['Training', 'Test', 'Predicitons'], loc='upper right')
plt.ylabel('Specific Capacitance (uF/cm$^2$)')
plt.xlabel('Cycle No.')
plt.title('Specific Capacitance - LS_24h(0.1:10)') 
"""
from sklearn import datasets
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.style.use('ggplot')


X = pd.read_csv("mlpX.csv")
y = pd.read_csv("mlpY.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
model = MLPRegressor()
model.fit(X_train, y_train.values.ravel())
print(model)

expected_y  = y_test
predicted_y = model.predict(X_test)

print(metrics.r2_score(expected_y, predicted_y))
print(metrics.mean_squared_error(expected_y, predicted_y))

##print(predicted_y)
f=open("MLP_DAL_(noconstr)_predictions.csv", "a")
for i in range(len(predicted_y)):
   f.write(str(predicted_y[i]))
   f.write("\n")
f.close()   

"""
f=open("MLP_DAL_(noconstr)_test.csv", "a")
expected_y.to_csv(f)
f.close()

for i in range(len(expected_y)):
   f.write(str(expected_y[i]))
   f.write("\n")
f.close()
"""

f=open("TestDataFile.csv", "a")
#for i in range(len(y_test)):
   #new_y_test=y_test.transpose() 
   #ny_test=y_test.reshape(-1,1)
f.write(y_test[y_test.columns[0]].to_string())
f.write("\n")
f.close()  


plt.figure(figsize=(10,10))
ax=plt.axis()

sns.regplot(expected_y, predicted_y, scatter_kws={"s": 100})
plt.plot(y,y)
plt.ylabel('Predicted')
plt.xlabel('Measured')
#ax.set_facecolor("white")
#plt.plot(X,y)
#plt.plot(X_test,predicted_y)



    
    
    
    
    
    
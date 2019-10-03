#Finding the optimal number of iterations/epochs required by gradient descent to achieve Library level accuracy for Linear Regression with learning rate 0.1

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
#Importing the dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,[0]].values
y=dataset.iloc[:,[1]].values

#Description of dataset: 
#Given experience of 30 employees(Independent variable x)
#We have to predict their salary(Dependent variable y) based upon their experience

#Feature scaling
def scaler(X):
    minimum=min(X)
    factor=max(X)-minimum
    for i in X:
        i[0]=(i[0]-minimum)/factor
    del(minimum,factor,i)
    return X

X=scaler(X)
X_pred=X
y=scaler(y)

#Adding constant feature
c=np.ones((len(X),2))
for i in range(len(c)):
    c[i][1]=X[i][0]
X=c
del(c,i)
    
#Sklearn linear model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_pred,y)
y_pred_lib=regressor.predict(X_pred)

#Linear Regression Algorithm using gradient deescent
w=2*np.random.random((2,1))-1
a=0.01
epochs=50
y_pred=np.zeros((30,1))
while round(np.mean(y_pred),3)!=round(np.mean(y_pred_lib),3):
    epochs+=1
    for i in range(epochs):
        output=np.dot(X,w)
        error=output-y
        w+=-a*((1/len(y))*(np.dot(X.T,error)))
    y_pred=np.dot(X,w)
#If executing this on command prompt, comment out plotting part
    plt.scatter(X_pred,y)
    plt.plot(X_pred,y_pred_lib,color='r',label='sklearn algorithm')
    plt.plot(X_pred,y_pred,color='b',label='our algorithm at epoch {}'.format(epochs))
    plt.title('Current algorithm VS sklearn algorithm')
    plt.legend()
    plt.show()
del(a,i)
print('Optimal number of epochs=',epochs)
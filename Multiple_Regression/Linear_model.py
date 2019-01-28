import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Gradient Descent for Multiple Regression
class Multiple_Regression:
    def __init__(self,reg_param=0):
        self.reg_param=reg_param
    '''def add_bias(X):
        x=np.ones((X.shape[0],X.shape[1]+1))
        for i in range(X.shape[1]):
            x[:,i+1]=X[:,i]
        X=x
        return X'''
    def scale(self,x):
        o=np.ones((x.shape))
        for i in range(x.shape[1]):
            o[:,i]=(x[:,i]-min(x[:,i]))/(max(x[:,i])-min(x[:,i]))
        return o
    def add_polynomial_features(self,X,degree):
        x=np.ones((len(X),degree))
        for i in range(1,degree+1):
            x[:,i-1]=X[:,0]**i
        return x
        
    def fit(self, X,y):
        x=np.ones((X.shape[0],X.shape[1]+1))
        for i in range(X.shape[1]):
            x[:,i+1]=X[:,i]
        X=x
        self.w=2*np.random.random((X.shape[1],1))-1
        a=0.1
        epochs=50000
        for i in range(epochs):
            output=np.dot(X,self.w)
            error=output-y
            m=len(X)
            self.w=self.w*(1-(a*self.reg_param/m))-a*(1/m)*(np.dot(X.T,error))
    def predict(self,X):
        x=np.ones((X.shape[0],X.shape[1]+1))
        for i in range(X.shape[1]):
            x[:,i+1]=X[:,i]
        X=x
        return np.dot(X,self.w)
    

def main():
    #importing the dataset
    '''dataset=pd.read_csv('50_Startups.csv')
    X=dataset.iloc[:,:3].values
    y=dataset.iloc[:,[-1]].values'''
    
    dataset=pd.read_csv('Position_Salaries.csv')
    X=dataset.iloc[:,1:2].values
    y=dataset.iloc[:,[-1]].values
    '''
    dataset=pd.read_csv('Salary_Data.csv')
    X=dataset.iloc[:,[0]].values
    y=dataset.iloc[:,[-1]].values'''
    
    '''#Adding features to polynomial data
    degree=10
    x=np.ones((len(X),degree))
    for i in range(1,degree+1):
        x[:,i-1]=X[:,0]**i
    X=x
    del(x)
    del(i)'''
     
    regressor=Multiple_Regression()
    #Adding polynomial features
    X=regressor.add_polynomial_features(X,10)
    
    #Scaling the featured
    X=regressor.scale(X)
    y=regressor.scale(y)
    #Fitting the data
    regressor.fit(X,y)
    y_pred=regressor.predict(X)
    
    #Plotting the results
    plt.scatter(X[:,0],y,color='r')
    plt.plot(X[:,0],y_pred)
    plt.show()
    
if __name__=='__main__':
    main()
        


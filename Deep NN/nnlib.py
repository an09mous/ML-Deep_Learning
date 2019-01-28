class nn:
    #Scaling function
    def scale(self,X,y):
        mean=np.mean(X,axis=0)
        var=np.var(X,axis=0)
        X=X-mean
        X/=(var)**0.5
        return X.T,y.T,mean,var
    def scale_transform(self,X,mean,var):
        X=X-mean
        X/=(var)**0.5
        return X.T
    
    #Activation functions and their derivatives
    #Sigmoid
    def sigmoid(self,z):
        return 1/(1+(np.exp(-z)))
    def sigmoid_derivative(self,a):
        return self.sigmoid(a)*(1-self.sigmoid(a))
    
    #tanh
    def tanh(self,z):
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    def tanh_derivative(self,a):
        return 1-self.tanh(a)**2
    
    #relu
    def relu(self,z):
        out=copy.deepcopy(z)
        out[out<0]=0
        return out
    def relu_derivative(self,a):
        out=copy.deepcopy(a)
        out[out>0]=1
        out[out<=0]=0
        return out
    
    #leaky_relu
    def leaky_relu(self,z,c=0.01):
        out=copy.deepcopy(z)
        return np.where(out>0, out, out * c)
    def leaky_relu_derivative(self,a,c=0.01):
        out=np.ones_like(a)
        out[a<0]=c
        return out
    
    #Loss function and its derivative
    def log_loss(self,y,a):
        return (-y *np.log(a+10**-8)-(1-y)*np.log(1 - a+10**-8)).mean()
    
    def log_loss_derivative(self,y,a):
        return (-y/a)+((1-y)/(1-a))
    
    #Random Weights Initialisation function
    def init_weights(self,nodes,features):
        return np.random.randn(nodes,features)
    
    def __init__(self,nodes, activations):
        np.random.seed(0)
        self.nodes=nodes
        self.activations=activations    
        self.act_func={'sigmoid':self.sigmoid,'tanh':self.tanh,'relu':self.relu,'leaky_relu':self.leaky_relu}
        self.act_func_der={'sigmoid':self.sigmoid_derivative,'tanh':self.tanh_derivative,'relu':self.relu_derivative,
                  'leaky_relu':self.leaky_relu_derivative}
    
    def fit(self,X,y,alpha=0.1,epochs=1000):
        X,y,self.mean,self.var=self.scale(X,y)
        Layers_num=len(self.nodes)  #Number of Layers
        m=X.shape[1]    #Number of training examples
        
        self.w=[0]   #Weight matrix
        self.b=[0]   #Bias matrix
        features=X.shape[0]     #Contain number of features to be feeded into each layer of nodes
        #Initialsing Weights and Bias of each layer
        for layer in range(Layers_num):
            self.w.append(self.init_weights(self.nodes[layer],features))
            self.b.append(np.zeros((self.nodes[layer],1)))
            features=self.nodes[layer]
        
        self.loss=[] #Loss vector
        
        for epoch in range(epochs):
            z=[0]
            a=[X]   #Stores activations of each layer
            
            #Forward Propagation
            for layer in range(1,Layers_num+1):
                z.append(np.dot(self.w[layer],a[layer-1])+self.b[layer])
                a.append(self.act_func[self.activations[layer-1]](z[layer]))
                
            self.loss.append(self.log_loss(y,a[-1]))    #Computing the loss
            
            #Backward Propagation
            da=self.log_loss_derivative(y,a[-1])
            for layer in range(Layers_num,0,-1):
                dg=self.act_func_der[self.activations[layer-1]]
                dz=da*dg(z[layer])
                dw=(1/m)*np.dot(dz,a[layer-1].T)
                db=(1/m)*np.sum(dz,axis=1,keepdims=True)
                da=np.dot(self.w[layer].T,dz)
                self.w[layer]-=alpha*dw
                self.b[layer]-=alpha*db
                
    def predict(self,X):
        X=self.scale_transform(X,self.mean,self.var)
        Layers_num=len(self.nodes)
        z=[0]
        a=[X]
        for layer in range(1,Layers_num+1):
            z.append(np.dot(self.w[layer],a[layer-1])+self.b[layer])
            a.append(self.act_func[self.activations[layer-1]](z[layer]))
        
        output=copy.deepcopy(a[-1])
        output[output<0.5]=0
        output[output>=0.5]=1
        return output.T
            
            
            
'''if __name__=='__main__':  
    #Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    import pandas as pd     
    import copy
    #Importing the Dataset
    dataset=pd.read_csv('Social_Network_Ads.csv')
    X=dataset.iloc[:,2:4].values
    y=dataset.iloc[:,[-1]].values
    
    #Splitting the dataset into training and test set
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    
    n=nn([4,1],['relu','sigmoid'])
    n.fit(X_train,y_train,epochs=2000)
    y_pred=n.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    accuracy=(cm[0][0]+cm[1][1])/len(y_test)
    print('Accuracy=',accuracy)
    plt.plot(n.loss)'''
    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd     
import copy

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

n=nn([8,4,1],['relu','relu','sigmoid'])
n.fit(X_train,y_train,0.1,10000)
y_pred=n.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
accuracy=(cm[0][0]+cm[1][1])/len(y_test)
print('Accuracy=',accuracy)
plt.plot(n.loss)
    
    

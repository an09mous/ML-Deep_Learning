import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

#Loading images dataset
digits=datasets.load_digits()
X=digits.images
y=digits.target

print(X[0].shape)   #Al images are already in grayscale and are of same size
plt.imshow(X[0])
plt.show()

#Now converting these 2d images to 1d(Flattening)
X=[img.flatten() for img in X]

#Splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Applying classification algorithm
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
clf.fit(X_train,y_train)

#Predicting test set
y_pred=clf.predict(X_test)

#Creating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Finding accuracy over test results
from sklearn.metrics import accuracy_score
a=accuracy_score(y_test,y_pred)
print('Accuracy=',a)

#Manually predicting the digits
test_img=digits.images[223] #Any random image
plt.imshow(test_img)
plt.show()
print('Predicted value:',clf.predict([test_img.flatten()]))
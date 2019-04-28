 

import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.tree import  DecisionTreeClassifier 
from sklearn import metrics

def load_data():  
    data = np.genfromtxt("spam\spambase.data",delimiter=",") 
    X = data[:,:-1]
    Y = data[:,-1]
    return X,Y  
X,Y = load_data()

"""1. train_test_split"""
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)  
clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_pred= clf.predict(x_test) 

acc = metrics.accuracy_score(y_test, y_pred)  
print(acc) 

 
 








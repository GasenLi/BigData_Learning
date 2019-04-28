# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:59:56 2019

@author: Adiministrator
"""
import numpy as np

hf1 = {'Sunny':0, 'Overcast':1, 'Rain':2}
hf2 = {'Hot':0, 'Mild':1, 'Cool':2}
hf3 = {'High':0, 'Normal':1,}
hf4 = {'Light':0, 'Strong':1}
hc = {'Yes':0, 'No':1}
x = []
y = []
def process_data(filename):
    fi = open(filename,'r')
    for line in fi:        
        line = line.strip()
        
        #ins = []
        tokens =  line.split(',')
#        print(tokens[0], tokens[1], tokens[2], tokens[3])
        id1,id2,id3,id4 =  hf1[tokens[0]],hf2[tokens[1]],hf3[tokens[2]],hf4[tokens[3]]
        y.append(hc[tokens[4]])
#        print(id1,id2,id3,id4)        
        x.append([id1,id2,id3,id4])
    fi.close()
    return np.array(x), np.array(y)

x,y = process_data('raw_data.txt')



from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

cls = DecisionTreeClassifier()
cls.fit(x,y)

y_pred = cls.predict(x)

acc = accuracy_score(y, y_pred)
print(acc)


 
from sklearn.naive_bayes import MultinomialNB
cls = MultinomialNB(alpha=0.01)
cls.fit(x, y) 

y_pred = cls.predict(x)
acc = accuracy_score(y, y_pred)
print(acc)







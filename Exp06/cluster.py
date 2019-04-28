# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:03:18 2019

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
  
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN 

 
#Load data X here
 
y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
 
y_pred = DBSCAN(eps = 0.15, min_samples = 5).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()



# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 23:37:37 2018

@author: ultra
"""

import numpy as np

def file2matrix(filename):
    f = open(filename)  # 打开文件
    dataSet = f.readlines()  # 读取文件的全部内容
    numberOfLines = len(dataSet)  # 获得数据集的行数
    data = np.zeros((numberOfLines, 3))  # 创建一个初始值为0，
                                             # 大小为 numberOfLines x 3 的数组
    label = [] # 用于保存没个数据的类别标签
    index = 0
    for line in dataSet: # 处理每一行数据
        line = line.strip() # 去掉行首尾的空白字符,(包括'\n', '\r', '\t', ' ')
        listFromLine = line.split() # 分割每行数据，保存到一个列表中
        data[index, :] = listFromLine[0:3] # 将列表中的特征保存到reurnMat中
        label.append(int(listFromLine[-1])) # 保存分类标签
        index += 1
    label = np.array(label)
    return data, label

# data, label = file2matrix('dating/datingTestSet2.txt')

#
#
# 
#import matplotlib.pyplot as plt  
#
#def showPlots(x, y, labels): 
#    # x:x轴数据，y:轴数据，labels:分类标签数据
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.scatter(x ,y, 15.0*labels, 15*labels)  
#    plt.show()
#
##showPlots(data[:,1], data[:,2], label)
##showPlots(data[:,0], data[:,1], label)
#
#
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 分别求各个特征的最小值
    maxVals = dataSet.max(0) # 分别求各个特征的最大值
    ranges = maxVals - minVals # 各个特征的取值范围
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
# normDataSet, ranges, minVals= autoNorm(data)

#
#
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#
# x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.2)
#
clf = KNeighborsClassifier(n_neighbors=500)

# clf = clf.fit(x_train, y_train)


# y_pred= clf.predict(x_test)

# acc = metrics.accuracy_score(y_test, y_pred)


# print(acc)



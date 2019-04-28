# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:57:16 2019

@author: 123456
"""
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def file2matrix(filename):
    f = open(filename) # 打开文件
    dataSet = f.readlines() # 读取文件的全部内容
    numberOfLines = len(dataSet) # 获得数据集的行数
    data = np.zeros((numberOfLines, 3)) # 创建一个初始值为0，
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

data, label = file2matrix('dating/datingTestSet2.txt')

sims = cosine_similarity(data)

sorted_sims = np.argsort(-sims)  #np.sort(-sims)
print(sorted_sims)


 


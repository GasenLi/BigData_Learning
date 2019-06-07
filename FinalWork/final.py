import pickle
from nltk import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import label_propagation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import BernoulliNB
from skift import FirstColFtClassifier
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize




# 词干还原
def stemming(token):
    stemming=SnowballStemmer("english")
    stemmed=[stemming.stem(each) for each in token]
    return stemmed

def tokenize(text):
    tokenizer=RegexpTokenizer("[^\s]+")  #设置正则表达式规则
    tokens=tokenizer.tokenize(text)
    stems=stemming(tokens)
    return stems


def loadTest(x_train,x_valid, stopWord):
    tfidf2 = TfidfVectorizer(analyzer='word', tokenizer=tokenize, stop_words=stopWord)
    x_train_tfid = tfidf2.fit_transform(x_train)
    x_valid_tfid = tfidf2.transform(x_valid)

    return x_train_tfid.toarray(), x_valid_tfid, tfidf2.get_feature_names()


def loadData(x_train,x_valid):
    tfidf2 = TfidfVectorizer(analyzer='word', stop_words='english')
    x_train_tfid = tfidf2.fit_transform(np.r_[x_train, x_valid])

    return x_train_tfid.toarray(), tfidf2.get_feature_names()


# 模型选择
def chooseModel(modelNum):
    return {
        1: neighbors.KNeighborsClassifier(),  # K近邻 0.549
        2: MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),  # 朴素贝叶斯
        3: DecisionTreeClassifier(),  # 决策树
        4: LogisticRegression(),  # 逻辑回归
        5: RandomForestClassifier(),  # 随机森林
        6: LinearSVC(penalty='l2',loss='squared_hinge',dual=True,tol=0.0001,C=1.0,multi_class='ovr',fit_intercept=True,intercept_scaling=1,class_weight=None,verbose=0,random_state=None,max_iter=1000),  # 线性支持向量分类
        7: NuSVC(),  # 核支持向量分类
        8: svm.SVC(),  # 支持向量分类 0.196
        9: MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=13,verbose=10,learning_rate_init=.1),  # 神经网络 0.566
        10: label_propagation.LabelSpreading(gamma=0.25,max_iter=5),  # 半监督 0.196
        11: LinearRegression(), # 线性回归模型
        12: SGDRegressor(),
        13: SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,tol=0.1, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,random_state=None), # 0.196
        14: BernoulliNB(),
        15: FirstColFtClassifier(),
        16: SGDClassifier(loss='hinge', penalty='l2', alpha=1e-2, n_iter=5, random_state=9)  # 9 12 38
    }.get(modelNum,'error')


#调参
def parameterTuning(n):
    return SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = 1e-2, n_iter=5, random_state=n)

# 执行预测
def predict(x_train_tfid, x_valid_tfid, y_train, model):
    model.fit(x_train_tfid[:5425],y_train)
    predict = model.predict(x_valid_tfid)

    return predict


import re
# 构造核心词
def buildCoreWord(feature):
    coreWords = []
    pattern = re.compile('[0-9]+')
    for i in range(len(feature)):
        j = i+1
        while j < len(feature):
            if feature[i] in feature[j]:
                if len(feature[i]) > 1 and not pattern.findall(feature[i]):
                    coreWords.append(feature[i])
                    break
            j = j+1

    print(" ".join(coreWords))
    print(len(coreWords))
    return coreWords

# 单词拆分
def wordSplit(x, coreWords):
    for index, line in enumerate(x):
        for word in line.split(" "):
            for coreWord in coreWords:
                if coreWord in word:
                    line.replace(word, "" ,1)
                    split = word.split(coreWord)
                    if split[0] == "" and split[1] == "":
                        continue
                    for splitword in split:
                        if splitword != "":
                            line = line + " " + splitword

        x[index] = line

# 读取特征词
def readFeature():
    featurePath = "E:\workSpace\BigData_Learning\FinalWork\\feature.txt"
    featureFile = open(featurePath, 'r', encoding='latin-1')

    feature = []
    for line in featureFile:
        for text in line.split():
            feature.append(text)
    return feature


# 读取核心词文件
def readCoreWordsFile():
    coreWordsPath = "E:\workSpace\BigData_Learning\FinalWork\coreWord.txt"
    coreWordsFile = open(coreWordsPath, 'r', encoding='latin-1')

    coreWords = []
    for line in coreWordsFile:
        for text in line.split():
            coreWords.append(text)
    return coreWords


# 读取停用词
def readStopWordsFile():
    stopWordsPath = "E:\workSpace\BigData_Learning\FinalWork\moreStopWords.txt"
    stopWordsFile = open(stopWordsPath, 'r', encoding='latin-1')

    stopWords = []
    for line in stopWordsFile:
        split = re.split('[\n ]', line)
        for word in split:
            if word != '':
                stopWords.append(word)

    return stopWords

import math
# 复合词拆分
# def cutLongNameFun(s):
'''
    longWords变为 long word：log里面有很多长函数名，比如WbxMeeting_VerifyMeetingIsExist。
    将其拆成小单词wbx meeting verify meeting is exist，更有意义。若有大写则分割。
    '''
# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
# 建立一个成本字典，假设Zipf定律和成本= -math.log（概率）。
words = open("words-by-frequency.txt").read().split() # 有特殊字符的话直接在其中添加
wordcost = dict((k, math.log((i+1)*math.log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)


def infer_spaces(s):
    '''Uses dynamic programming to infer the location of spaces in a string without spaces.
    .使用动态编程来推断不带空格的字符串中空格的位置。'''
    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))


def cutList(list):
    for lineindex,line in enumerate(list):
        split = line.split()
        for index,word in enumerate(split):
            if len(word) >= 12:
                split[index] = infer_spaces(word)

        list[lineindex] = " ".join(split)


## 数字字母分割
def separateNumAndLett(list):
    for lineindex,line in enumerate(list):
        list[lineindex] = " ".join(re.findall(r'[0-9]+|[a-z]+', line))




# 单词还原
from FinalWork import SpellCheck
def wordRestore(list):
    for lineindex,line in enumerate(list):
        split = line.split()
        for index, word in enumerate(split):
            if len(word) <= 3:
                temp = SpellCheck.correct(word)
                if word != temp:
                    print(word+ "  " +temp)
                    split[index] = temp

        list[lineindex] = " ".join(split)


from sklearn.metrics import accuracy_score
if __name__ == "__main__":
    x_train,y_train,x_valid,y_valid = pickle.load(open('E:\workSpace\BigData_Learning\Data\Final\data_student.pkl','rb'))
    # x_test = pickle.load(open('E:\workSpace\BigData_Learning\Data\Final\data_test.pkl','rb'))

    separateNumAndLett(x_train)
    separateNumAndLett(x_valid)

    cutList(x_train)
    cutList(x_valid)

    # wordRestore(x_train)
    # wordRestore(x_valid)


    x_train_tfid, x_valid_tfid, feature= loadTest(x_train, x_valid, readStopWordsFile())
    print(feature)

    predict = predict(x_train_tfid, x_valid_tfid, y_train, chooseModel(16))
    acc = accuracy_score(y_valid, predict)
    print(acc)

    #print(SpellCheck.correct("abstrait"))
    #np.savetxt("final.txt", predict, fmt='%d')



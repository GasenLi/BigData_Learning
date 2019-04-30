import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
def loadData(x_train,x_valid):
    tfidf2 = TfidfVectorizer(analyzer='word', stop_words='english')
    x_train_tfid =  tfidf2.fit_transform(np.r_[x_train, x_valid]).toarray()

    return x_train_tfid, tfidf2.get_feature_names()


def predict(x_train_tfid, y_train):
    # knn = neighbors.KNeighborsClassifier()
    #knn.fit(x_train_tfid[:5425], y_train)
    #knnPredict = knn.predict(x_train_tfid[5425:])

    # mnb = MultinomialNB()
    # mnb.fit(x_train_tfid[:5425],y_train)
    # mnbPredict = mnb.predict(x_train_tfid[5425:])

    cls = DecisionTreeClassifier()
    cls.fit(x_train_tfid[:5425],y_train)
    predict = cls.predict(x_train_tfid[5425:])

    return predict




from sklearn.metrics import accuracy_score
if __name__ == "__main__":
    x_train,y_train,x_valid,y_valid = pickle.load(open('E:\workSpace\BigData_Learning\Data\Final\data_student.pkl','rb'))
    x_train_tfid, feature = loadData(x_train, x_valid)
    predict = predict(x_train_tfid, y_train)

    acc = accuracy_score(y_valid, predict)
    print(acc)

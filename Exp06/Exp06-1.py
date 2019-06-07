import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def loadData():
    dataPath = 'E:\workSpace\BigData_Learning\Data\Exp06\circle.txt'
    dataFile = open(dataPath, 'r')

    X = []
    YTrue = []
    for line in dataFile:
        data = line.split(',')
        couple = []
        couple.append(float(data[0]))
        couple.append(float(data[1]))

        X.append(couple)
        YTrue.append(int(data[2]))

    return np.array(X), np.array(YTrue)

import Exp06.evaluator as EV
if __name__ == "__main__":
    print(plt.get_backend() + "---------")

    X, YTrue = loadData()

    y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()
    acc, nmi, ari = EV.evaluate(YTrue, y_pred)
    print("KMeans --- acc:" + str(acc) + "  nmi:" + str(nmi))

    y_pred = DBSCAN(eps = 0.15, min_samples = 5).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()
    acc, nmi, ari = EV.evaluate(YTrue, y_pred)
    print("DBSCAN --- acc:" + str(acc) + "  nmi:" + str(nmi))
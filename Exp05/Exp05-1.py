import numpy as np


def process_data(filename):
    hf1 = {'Sunny':0, 'Overcast':1, 'Rain':2}
    hf2 = {'Hot':0, 'Mild':1, 'Cool':2}
    hf3 = {'High':0, 'Normal':1,}
    hf4 = {'Light':0, 'Strong':1}
    hc = {'Yes':0, 'No':1}
    x = []
    y = []

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

    outputFile = 'processed_data.txt'
    oF = open(outputFile, 'w')
    i = 0;
    for value in y:
        oF.write(str(x[i][0]) + " ")
        oF.write(str(x[i][1]) + " ")
        oF.write(str(x[i][2]) + " ")
        oF.write(str(x[i][3]) + " ")
        oF.write(str(value))
        oF.write('\r\n')
        i = i+1

    oF.close()
    return np.array(x), np.array(y)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
if __name__ == "__main__":
    x,y = process_data('E:\workSpace\BigData_Learning\Data\Exp05\\raw_data.txt')
    cls = DecisionTreeClassifier()
    cls.fit(x,y)

    y_pred = cls.predict(x)

    print(y_pred)
    acc = accuracy_score(y, y_pred)
    print(acc)
from Exp04 import dating
import numpy

def showmax(lt):
    index1 = 0  #记录出现次数最多的元素下标
    max = 0  #记录最大的元素出现次数

    for i in range(len(lt)):
        flag = 0 #记录每一个元素出现的次数

    for j in range(i+1,len(lt)): #遍历i之后的元素下标
        if lt[j] == lt[i]:
            flag += 1  #每当发现与自己相同的元素，flag+1

        if flag > max:  #如果此时元素出现的次数大于最大值，记录此时元素的下标
            max = flag
            index1 = i

    return lt[index1]

def classification(data,label,rate,maximumDistance):
    predictions = []

    prePosition = int(len(data)*rate)
    while prePosition < len(data):
        samPosition = 0
        prediction = []

        while samPosition < len(data)*rate:

            distance = (data[samPosition][0]-data[prePosition][0]) ** 2 \
                       + (data[samPosition][1]-data[prePosition][2]) ** 2 \
                       + (data[samPosition][2]-data[prePosition][2])  ** 2

            if distance < maximumDistance:
                prediction.append(label[samPosition])

            samPosition += 1

        if len(prediction) > 0:
            predictions.append(showmax(prediction))
        else:
            predictions.append(0)

        prePosition += 1

    return predictions

def getCorrectRate(label, predictions, rate):
    labelPosition = int(len(label)*rate)
    prePosition = 0
    correctNum = 0

    while prePosition < len(predictions):
        if predictions[prePosition] == label[labelPosition]:
            correctNum += 1

        prePosition += 1
        labelPosition += 1

    return correctNum/len(predictions)


def processData(data):
    dataPosition = 0
    while dataPosition < len(data):
        data[dataPosition][0] = data[dataPosition][0]/1000
        data[dataPosition][2] = data[dataPosition][2]*10

        dataPosition += 1


if __name__ == "__main__":
    rate = 0.3
    maximumDistance = 300

    data, label = dating.file2matrix('E:\workplace\BigData\Data\dating\datingTestSet2.txt')
    processData(data)
    predictions = classification(data, label, rate, maximumDistance)
    correctRate = getCorrectRate(label, predictions, rate)

    print(predictions)
    print(correctRate)



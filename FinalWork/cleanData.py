import re

dataFile = open("E:\workSpace\BigData_Learning\Data\Final\\2.txt", 'r')
outputFile = open("E:\workSpace\BigData_Learning\Data\Final\\CorrectWord.txt", 'w', encoding='UTF-8')


for line in dataFile:
    outputFile.write(re.sub('[	0-9 \n.	]', '', line))
    outputFile.write('\n')

outputFile.close()
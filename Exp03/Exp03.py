import os


def printVector():
    file = open("vec_tf.txt",'w')
    
    for fileName in vectors:
        file.write(fileName)
        file.write(": [")
        for rate in vectors[fileName]:
            file.write(repr(rate)+" ")
        file.write("]\n\r")
    
    file.close()
        

def buildVectors():
    diction.sort
    
    for fileName in allFileDict:
        vector = []
        fileDict = allFileDict[fileName]

        for word in diction:
            if word in fileDict:
                vector.append(fileDict[word])
            else:
                vector.append(0)
        
        vectors[fileName] = vector
    

def buildFileDict(file):
    fileDict = {}

    for line in file:
        for text in line.split():
            word = ''.join(list(filter(str.isalpha,text)))
            if word in fileDict:
                fileDict[word] += 1
            else:
                fileDict[word] = 1

    return fileDict 

    
def buildDIction(fileDict):
    for key in fileDict:
        if key not in diction:
            diction.append(key)
            

diction = [] 
allFileDict = {}
vectors = {}
DATA_PATH = './data'
sourceDir = os.listdir(DATA_PATH)
for fileName in sourceDir:
    file = open(os.path.join(DATA_PATH, fileName),'r')
    fileDict = buildFileDict(file)
    
    buildDIction(fileDict)
    allFileDict[fileName] = fileDict

buildVectors()
printVector()


    
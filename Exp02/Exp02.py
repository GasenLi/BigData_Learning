import os

DATA_PATH = 'E:/workplace/BigData/Data/20newsgroups'
OUTPUT_PATH = 'E:/workplace/BigData/实验二/inversed_index.txt'
dirs1 = os.listdir(DATA_PATH)

for dirs2 in dirs1:
    dirs2 = os.path.join(DATA_PATH,dirs2)
    fileNames = os.listdir(dirs2)
    index = {}

    for fileName in fileNames:
        file = open(os.path.join(dirs2, fileName),'r',encoding='latin-1')
        dict = {}
        
        for line in file:
            for text in line.split():
                word = ''.join(list(filter(str.isalpha,text)))
                if word in dict:
                    dict[word] += 1
                else:
                    dict[word] = 1
        
        for key in dict:
            if key not in index:
                 index[key] = {}
          
            index[key][fileName] = dict[key]

outputFile = open(OUTPUT_PATH, 'w', encoding='latin-1')
for word in index:
    outputFile.write(word + '  ')
    for key in index[word]:
        outputFile.write(key + ':' + str(index[word][key]) + ' ')
    outputFile.write('\n\r')

outputFile.close()
        

    
    
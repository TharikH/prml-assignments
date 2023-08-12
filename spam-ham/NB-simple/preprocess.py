import numpy as np
import os
from nltk.stem import PorterStemmer
import nltk
import enchant
import pickle

def preprocess(content,U,totalCount):
    d={}
    words=content.split()
    ps = PorterStemmer()
    engDict=enchant.Dict("en_US")
    for word in words:
        if (not word.isalnum()) and (word != '$'):
            continue
        if word.isnumeric():
            value="NUMBER"
        else:
            value=word.lower()
            if engDict.check(value):
                value=ps.stem(value)
            else:
                value="UNKNOWN"
        
        if(d.get(value) == None):
            if U.get(value) == None:
                U[value]=totalCount
                totalCount+=1
            d[value]=1
            
    
    
    return d

def readData(PATH,spam,U,totalCount):
    dirList=os.listdir(PATH)
    size=len(dirList)
#     print(size)
    X=[]
    y=[spam for i in range(size)]
    c=0
    for file in dirList:
        with open(PATH+'/'+file, 'r',errors='ignore') as f:
#             print(file)
            content=f.read()
            contentDict=preprocess(content,U,totalCount)
#         print(contentDict)
        X.append(contentDict)
#         if c==10:
#             break
#         c+=1
    return X,y

def makeDataStructure(spamX,spamY,U):
    m=len(spamX)
    d=len(U)
    X=np.zeros([d,m])
    for i in range(m):
        for word in spamX[i]:
            if U.get(word) == None:
                continue
            X[U[word]][i]=spamX[i][word]

    return X

def storeData(XSpam,XHam,U):
    spamfile=open('spam-new', 'wb')
    hamfile=open('ham-new', 'wb')
    Ufile= open('u-new', 'wb')
    
    
    pickle.dump(XSpam,spamfile)
    pickle.dump(XHam,hamfile)
    pickle.dump(U, Ufile)
    
    spamfile.close()
    hamfile.close()
    Ufile.close()
       

U={}   
totalCount=0
SPAM_PATH="Dataset/enron1/spam"
HAM_PATH="Dataset/enron1/ham"
spamX,spamY=readData(SPAM_PATH,1,U,totalCount)
hamX,hamY=readData(HAM_PATH,0,U,totalCount)
# print(U)

# print(X)
# listOfWords=list(U)
XSpam=makeDataStructure(spamX,spamY,U)
XHam=makeDataStructure(hamX,hamY,U)


storeData(XSpam,XHam,U)


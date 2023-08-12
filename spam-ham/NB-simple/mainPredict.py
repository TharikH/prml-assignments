import numpy as np
import os
from nltk.stem import PorterStemmer
import nltk
import enchant
import pickle

def probPart(x,mean):
    d=x.shape[0]
#     print(mean.shape)
    prob=1
    count=d**2
    for i in range(d):
        if x[i] != 0 :
            if mean[i] != 0:
                prob*=mean[i]
            else:
                prob*=1/count
        else:
            if mean[i] != 1:
                prob*=(1-mean[i])
            else:
                prob*=1/count
                
    return prob
            
    
def naiveBayesPrediction(x,p,mean):
    spamProbability=probPart(x,mean[:,1])*p
    hamProbability=probPart(x,mean[:,0])*(1-p)
    if(spamProbability >= hamProbability):
        return 1
    else:
        return 0
    
def loadReprsentation():
    pfile = open('p-new', 'rb')
    meanfile = open('mean-new', 'rb')
    Ufile= open('u-new', 'rb')
    
    p=pickle.load(pfile)
    mean=pickle.load(meanfile)
    U=pickle.load(Ufile)
    
    pfile.close()
    meanfile.close()
    Ufile.close()
    
    return p,mean,U

def testPreprocess(content,U):
    words=content.split()
    ps = PorterStemmer()
    engDict=enchant.Dict("en_US")
    d=len(U)
    xTest=np.zeros([d,1])
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
        
        if(U.get(value) == None):
            continue
            
        xTest[U[value]][0]+=1
                
    return xTest

def prediction(file,U,p,mean):
    content=file.read()
    xTest=testPreprocess(content,U)
#     print(xTest.shape)
#     print(mean.shape)
    yPredicted=naiveBayesPrediction(xTest,p,mean)
    print(yPredicted)

# with open("Dataset/enron1/spam/0006.2003-12-18.GP.spam.txt") as f:
#     p,mean,U=loadReprsentation()
#     prediction(f,U,p,mean)
    
#testing folder
TEST_PATH_SPAM="Dataset/enron6/spam"
TEST_PATH_HAM="Dataset/enron6/ham"
p,mean,U=loadReprsentation()

# print(mean)
def test(PATH,p,mean,U,spam):
    
    dirList=os.listdir(PATH)
    size=len(dirList)
    count=0
    c=0
    for file in dirList:
        with open(PATH+'/'+file, 'r',errors='ignore') as f:
                content=f.read()
                xTest=testPreprocess(content,U)
                yPredicted=naiveBayesPrediction(xTest,p,mean)
                count+=yPredicted
                
#         if c==100:
#             break
#         c+=1
#     size=c
#     if spam == 1:
#         print(count/size)
#     else:
#         print(1 - count/size)
    return count,size
                

def predictOnTestData(PATH):
    p,mean,U=loadReprsentation()
    dirList=os.listdir(PATH)
    size=len(dirList)
    yPredictedDict={}
    for file in dirList:
        with open(PATH+'/'+file, 'r',errors='ignore') as f:
                content=f.read()
                xTest=testPreprocess(content,U)
                yPredicted=naiveBayesPrediction(xTest,p,mean)
                yPredictedDict[file]=yPredicted
    
    print(yPredictedDict)
    
PATH="test/"
predictOnTestData(PATH)


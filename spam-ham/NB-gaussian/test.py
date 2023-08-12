import numpy as np
import os
from nltk.stem import PorterStemmer
import nltk
import enchant
import pickle

def gaussianPart(x,mean,cov):
    
    d=cov.shape[0]
    print(x.shape)
    print(mean.shape)
    mean=mean.reshape(d,1)
    sub=x-mean
    print(sub.shape)
    sub=sub.reshape(d,1)
    expTerm=-(sub@np.linalg.pinv(cov)@(sub.transpose()))
#     expTerm=-(sub@(sub.transpose()))
    return np.exp(expTerm)

def naiveBayesPrediction(x,p,mean,cov):
    spamProbability=gaussianPart(x,mean[:,1],cov)*p
    hamProbability=gaussianPart(x,mean[:,0],cov)*(1-p)
    if(spamProbability >= hamProbability):
        return 1
    else:
        return 0
    
def loadReprsentation():
    pfile = open('p', 'rb')
    meanfile = open('mean', 'rb')
    covfile = open('cov', 'rb')
    Ufile= open('u', 'rb')
    
    p=pickle.load(pfile)
    mean=pickle.load(meanfile)
    cov=pickle.load(covfile)
    U=pickle.load(Ufile)
    
    pfile.close()
    meanfile.close()
    covfile.close()
    Ufile.close()
    
    return p,mean,cov,U

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

def test(PATH,p,mean,cov,U,spam):
    
    dirList=os.listdir(PATH)
    size=len(dirList)
    count=0
    for file in dirList:
        with open(PATH+'/'+file, 'r',errors='ignore') as f:
                content=f.read()
                xTest=testPreprocess(content,U)
                yPredicted=naiveBayesPrediction(xTest,p,mean,cov)
                count+=yPredicted
    if spam == 1:
        print(count/size)
    else:
        print(1 - count/size)



p,mean,cov=naiveBayesTraining(XSpam,XHam)
cov=np.dot(np.identity(cov.shape[0]),cov)

#testing folder
TEST_PATH_SPAM="Dataset/enron6/spam"
TEST_PATH_HAM="Dataset/enron6/ham"
p,mean,U=loadReprsentation()

test(TEST_PATH_SPAM,p,mean,cov,U,1)
test(TEST_PATH_HAM,p,mean,cov,U,1)  
import numpy as np
import os
from nltk.stem import PorterStemmer
import nltk
import enchant
import pickle


def loadData():
    spamfile=open('spam', 'rb')
    hamfile=open('ham', 'rb')
    Ufile= open('u', 'rb')
    
    U=pickle.load(Ufile)
    XSpam=pickle.load(spamfile)
    XHam=pickle.load(hamfile)
    
    spamfile.close()
    hamfile.close()
    Ufile.close()
    
    return U,XSpam,XHam

def storeRepresentation(p,mean):
    pfile = open('p', 'wb')
    meanfile = open('mean', 'wb')
    
    pickle.dump(p, pfile)
    pickle.dump(mean, meanfile)
    
    pfile.close()
    meanfile.close()
    
def loadReprsentation():
    pfile = open('p-f', 'rb')
    meanfile = open('mean-f', 'rb')
    Ufile= open('u', 'rb')
    
    p=pickle.load(pfile)
    mean=pickle.load(meanfile)
    U=pickle.load(Ufile)
    
    pfile.close()
    meanfile.close()
    Ufile.close()
    
    return p,mean,U

def naiveBayesTraining(spamX,hamX):
    spamSize=np.sum(np.sum(spamX))
    hamSize=np.sum(np.sum(hamX))
    size=hamX.shape[1]+spamX.shape[1]
    mean=np.zeros([spamX.shape[0],2])
    p=spamSize/(spamSize+hamSize)
    mean[:,1]=np.sum(spamX,axis=1)/spamSize
    mean[:,0]=np.sum(hamX,axis=1)/hamSize

    
    return p,mean


U,XSpam,XHam=loadData()

# d=XSpam.shape[0]
# ones=np.ones([d,1])


# laplace smoothing
# XSpam=np.concatenate((XSpam,ones),axis=1)
# XHam=np.concatenate((XHam,ones),axis=1)

# training
p,mean=naiveBayesTraining(XSpam,XHam)
# print(mean.shape)

storeRepresentation(p,mean)



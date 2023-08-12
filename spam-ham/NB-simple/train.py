import numpy as np
import os
from nltk.stem import PorterStemmer
import nltk
import enchant
import pickle

def loadData():
    spamfile=open('spam-new', 'rb')
    hamfile=open('ham-new', 'rb')
    Ufile= open('u-new', 'rb')
    
    U=pickle.load(Ufile)
    XSpam=pickle.load(spamfile)
    XHam=pickle.load(hamfile)
    
    spamfile.close()
    hamfile.close()
    Ufile.close()
    
    return U,XSpam,XHam

def storeRepresentation(p,mean):
    pfile = open('p-new', 'wb')
    meanfile = open('mean-new', 'wb')
    
    pickle.dump(p, pfile)
    pickle.dump(mean, meanfile)
    
    pfile.close()
    meanfile.close()
    
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

def naiveBayesTraining(spamX,hamX):
    spamSize=spamX.shape[1]
    hamSize=hamX.shape[1]
    size=spamSize+hamSize
    mean=np.zeros([spamX.shape[0],2])
    p=spamSize/size
    
    mean[:,1]=np.sum(spamX,axis=1)/spamSize
    mean[:,0]=np.sum(hamX,axis=1)/hamSize

    
    return p,mean


U,XSpam,XHam=loadData()

# laplace smoothing
# XSpam=np.concatenate((XSpam,ones),axis=1)
# XHam=np.concatenate((XHam,ones),axis=1)

# training
p,mean=naiveBayesTraining(XSpam,XHam)
# print(mean.shape)

storeRepresentation(p,mean)



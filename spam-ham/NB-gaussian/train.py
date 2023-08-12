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


def storeRepresentation(p,mean,cov,U):
    pfile = open('p', 'ab')
    meanfile = open('mean', 'ab')
    covfile = open('cov', 'ab')
    
    pickle.dump(p, pfile)
    pickle.dump(mean, meanfile)
    pickle.dump(cov, covfile)
    
    pfile.close()
    meanfile.close()
    covfile.close()

def naiveBayesTraining(spamX,hamX):
    spamSize=spamX.shape[1]
    hamSize=hamX.shape[1]
    size=hamX.shape[1]+spamX.shape[1]
    mean=np.zeros([spamX.shape[0],2])
    data=np.concatenate((spamX,hamX),axis=1)
    p=spamSize/size
    mean[:,0]=np.sum(spamX,axis=1)/spamSize
    mean[:,1]=np.sum(hamX,axis=1)/hamSize
    cov=(data@data.transpose())/size
    
    
    return p,mean,cov



U,XSpam,XHam=loadData()
p,mean,cov=naiveBayesTraining(XSpam,XHam)
storeRepresentation(p,mean,cov,U)

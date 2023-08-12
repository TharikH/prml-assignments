import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

def loadData():
    data = pd.read_csv("A2Q1.csv",header=None)
    return data
def changeDataStructure(data):
    data_numpy=data.to_numpy()
    data_numpy=data_numpy.transpose()
    return data_numpy

# kmeans part
LABEL_COLOR_LIST=["b","g","r","c","m","y","k","w"]
def findError(x,mean):
    sub=x-mean
    error=np.dot(sub,sub)
    return error

#find mean nearest to a point
def getIndicatorOfX(x,means,curr_indicator):
    index=curr_indicator
    k=means.shape[1]
    n=means.shape[0]
    curr_mean=means[:,curr_indicator]
    small=findError(x,curr_mean)
    for i in range(k):
        mean=means[:,i]
        val=findError(x,mean)
        if(small > val):
            small=val
            index=i

    return index

#re-assignemnet step
def assignMean(data,indicator,means):
    m=data.shape[1]
    for i in range(m):
        temp_ind=indicator[i]
        indicator[i]=getIndicatorOfX(data[:,i],means,temp_ind)
    return indicator

#computing mean step
def computeMean(data,means,indicator):
    m=data.shape[1]
    n=data.shape[0]
    k=means.shape[1]
    err_sum=0
    for i in range(k):
        count=0
        mean=np.zeros(n)
        curr_mean=means[:,i]
        for j in range(m):
            if indicator[j] == i:
                x=data[:,j]
                mean=mean+x
                count=count+1
                err_sum+=findError(x,curr_mean)

        if count !=0:
            means[:,i]=mean/count
    
    return means,err_sum

def isEqual(arr1,arr2):
    n=len(arr1)
    for i in range(n):
        if arr1[i] != arr2[i]:
            return False
    return True

def calculateError(data,indicator,means):
    m=data.shape[1]
    n=data.shape[0]
    s=0
    for i in range(m):
        s+=findError(data[:,i],means[:,indicator[i]])    
    return s

def plotIteration(iteration,error,k,randinit):
    plt.title("Iteration vs Error with k = "+str(k) + " with random initialization - "+str(randinit))
    plt.xlabel("Iteration",fontsize=15)
    plt.ylabel("Error",fontsize=15)
    plt.plot(iteration,error)
    plt.show()

def Kmeans(data,k,randinit):
    n=data.shape[0]
    m=data.shape[1]
    means=np.zeros([k,n])
    indicator=[0]*m
    trans_data=data.transpose()
    
    #randomly initialize means
    i=0
    random.seed(i+randinit)#seeds are used so as to fix random initialization

    while(i < k):
        val=random.choice(trans_data)
        means[i]=val
        i+=1
    

    means=means.transpose()#change to n x k matrix
    
    count=0
    iteration=[]
    error=[]
    iteration.append(count)
    error.append(calculateError(data,indicator,means))
    print(error[0])
    count+=1
    
    prev_indicator=copy.deepcopy(indicator)#to store curr assignments so as o compare whether next assignment converged or not
    indicator=assignMean(data,indicator,means)
#     print(prev_indicator)
#     print(indicator)
#     print(means)
    #run the loop till kmeans converge
    while(not isEqual(prev_indicator,indicator)):
        
        prev_indicator=copy.deepcopy(indicator)
        means,err_sum=computeMean(data,means,indicator)
        
        iteration.append(count)
        error.append(err_sum)
        count+=1
        
        indicator=assignMean(data,indicator,means)
    return means,indicator,iteration,error

# k=4
# i=0
# # for i in range(0,1):
# print("Random initialization " + str(i))
# means,indicator,iteration,error=Kmeans(data,k,i)
# #     print(iteration)
# plotIteration(iteration,error,k,i)
# print(error[len(error) - 1])

def plot(x,y,xlabel="X",ylabel="Y",title=""):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def calculateProbability(x,mean,cov,pi):
    sub=x-mean
    d=mean.shape[0]
    det=np.linalg.det(cov)
    if(det < 1e-10):
        cov=cov+np.identity(d) * 0.1

    pseudo_determinent=np.linalg.det(cov)
    invCov=np.linalg.pinv(cov)
#     det=1/np.linalg.det(invCov)
#     eig_values,_ = np.linalg.eig(cov)
#     pseudo_determinent = np.product(np.real(eig_values[eig_values > 1e-2]))
    
        
    powerTerm=-(sub.transpose() @ invCov @ sub) / 2
    coeff=1/(((2 * np.pi)**(d/2) * ((pseudo_determinent) ** 0.5)))
    lambdaValue=coeff * np.exp(powerTerm) * pi
    return lambdaValue
    
def logLikelyHood(data,piList,mean,covarianceList):
    
    d,m=data.shape
    k=len(piList)
    totalLog=0
    for i in range(m):
        total=0;
        for j in range(k):
            total+=calculateProbability(data[:,None,i],mean[:,None,j],covarianceList[j],piList[j])
        totalLog+=np.log(total)
    return totalLog

def expectation(piList,mean,covarianceList,data):
    d,m=data.shape
    k=len(piList)
    
    lambdaMatrix=np.zeros([m,k])
    
    for i in range(m):
        total=0;
        for j in range(k):
            lambdaMatrix[i][j]=calculateProbability(data[:,None,i],mean[:,None,j],covarianceList[j],piList[j])
            total+=lambdaMatrix[i][j]
        lambdaMatrix[i,:]/=total
    
    return lambdaMatrix
    

def maximization(lambdaMatrix,data):
    m,k=lambdaMatrix.shape
    d=data.shape[0]
    piList=[0 for i in range(k)]
    mean=np.zeros([d,k])
    covarianceList=[]
    for i in range(k):
        piList[i]=np.mean(lambdaMatrix[:,i])
        lambdaX=np.zeros([d,1])
        lambdaSum=0
        for j in range(m):
            lambdaX+=(lambdaMatrix[j][i]* data[:,None,j])
            lambdaSum+=lambdaMatrix[j][i]
        
        mean[:,None,i]=lambdaX/lambdaSum
        cov=np.zeros([d,d])
        for j in range(m):
            sub=data[:,None,j] - mean[:,None,i]
            cov+=lambdaMatrix[j][i] * (sub @ sub.transpose())
        cov/=lambdaSum
        covarianceList.append(cov)
        
                      
    return piList,mean,covarianceList
    
def gaussianEM(data,means,indicator):
    d,k=means.shape
    m=data.shape[1]
    collectionList=[[] for i in range(k)]
    covarianceList=[]
    piList=[0 for i in range(k)]
    for i in range(m):
        collectionList[indicator[i]].append(data[:,i])
        piList[indicator[i]]+=1

    for i in range(k):
        collectionList[i]=np.array(collectionList[i]).transpose()
        t=collectionList[i].shape[1]
        covarianceList.append((collectionList[i] @ collectionList[i].transpose()) / t)
        piList[i]/=m
#     print(check_symmetric(covarianceList[3]))
#     covarianceList=[np.identity(d) for i in range(k)]
    count=1
    logList=[]
    iteration=[]
    prevError=logLikelyHood(data,piList,means,covarianceList)
    print(prevError)
    while(count<40):  
        iteration.append(count)
        logList.append(prevError[0][0])
        lambdaMatrix=expectation(piList,means,covarianceList,data)
        piList,means,covarianceList=maximization(lambdaMatrix,data)
    #     if(diffOfParameter(prevTheta,newTheta)):
        currError=logLikelyHood(data,piList,means,covarianceList)
    #     if(abs(prevError - currError) <= 0.001):
    #         break

#         print(prevError)
    #     print(means)
#         print(means)
    #     print(covarianceList)
#         print(sum(piList))
        prevError=currError
        count+=1
    return iteration,logList,means
#     plot(iteration,logList,"Iteration","Loglikelyhood")

data=loadData()
data=changeDataStructure(data)

k=4
totalList=[]
for i in range(0,100):
    print("Random initialization " + str(i))
    means,indicator,iteration,error=Kmeans(data,k,i)
    iteration,loglikely,means=gaussianEM(data,means,indicator)
    totalList.append(loglikely)
    
                     
plot(iteration,np.mean(totalList,axis=0),"Iteration","Loglikelihood","gaussian-random-100")   

k=4
i=1
means,indicator,iteration,error=Kmeans(data,k,i)
print("k-means error :",error[len(error) - 1])
iteration,loglikely,means=gaussianEM(data,means,indicator)
indicator=assignMean(data,indicator,means)
means,err_sum=computeMean(data,means,indicator)
print("GMM error after making it into cluster : "err_sum)

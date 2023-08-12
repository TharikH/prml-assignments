import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def loadData():
    data = pd.read_csv("Dataset.csv",header=None)
    data.columns=["X","Y"]
    return data

def changeDataStructure(data):
    data_numpy=data.to_numpy()
    data_numpy=data_numpy.transpose()
    return data_numpy

def dataCenter(data):
    mean=data.mean(axis=1,keepdims=True)
    data=data - mean
    return data

def findCovariance(data):
    cov=data@data.transpose()
    n=cov.shape[0]
    cov=cov/n
    return cov

def findEigen(covariance_matrix):
    eigen_value,eigen_vector= np.linalg.eig(covariance_matrix)
    return eigen_value,eigen_vector

def findTopKEigenVec(eigen_vector,eigen_value,k):
    n=eigen_value.shape[0]
    highest_eigen_val_index = eigen_value.argsort()[::-1]
    w=[] #top k eigen vectors are stored
    for i in range(k):
        w.append(np.array(eigen_vector[:,highest_eigen_val_index[i]]))
        w[i]=w[i].reshape(eigen_vector.shape[1],1)
    return w

def createProxy(w,data):
    new_data=(data.transpose() @ w) @ w.transpose()
    return new_data

def plotOn_ith_Eigenvec(data,w,i):
    new_data=createProxy(w[i],data)
    plt.figure(figsize=(12, 12))
    plt.title("Dataset",fontsize=20)
    plt.xlabel("X Data",fontsize=15)
    plt.ylabel("Y Data",fontsize=15)
    plt.scatter(new_data.transpose()[0],new_data.transpose()[1],color='k',label="Projection on to first eigen")
    plt.scatter(data[0],data[1],label="Dataset")
    scale=1
    ax = plt.axes()
    ax.arrow(0,0,scale*w[1][0][0],scale*w[1][1][0] , head_width=0.1, head_length=0.5,color='r',label="Second eigen component")
    ax.arrow(0,0,scale*w[0][0][0],scale*w[0][1][0] , head_width=0.1, head_length=0.5,color='y',label="First eigen component")
    plt.legend(loc="upper left",fontsize=10)
    plt.show()

data=loadData()
print("mean of data :")
print(data.mean())

data=loadData()
data=changeDataStructure(data)
data=dataCenter(data)
plt.figure(figsize=(10, 10))
plt.scatter(data[0],data[1],label="Data")

plt.title("Dataset",fontsize=20)
plt.xlabel("X Data",fontsize=15)
plt.ylabel("Y Data",fontsize=15)
plt.legend(loc="upper left",fontsize=15)

plt.show()

k=2
covariance_matrix=findCovariance(data)
eigen_value,eigen_vector=findEigen(covariance_matrix)
w=findTopKEigenVec(eigen_vector,eigen_value,k)

print("top k eigen vectors:")
print(w)
print("eigen values without sorting")
print(eigen_value)

eigen_value.sort()
eigen_value=eigen_value[::-1]
print("variance by PC1 : ",eigen_value[0])
print("variance by PC2 : ",eigen_value[1])
s=sum(eigen_value)
variance_of_PC1 = eigen_value[0]/s
variance_of_PC2 = eigen_value[1]/s


print("percent variance by PC1 : ",variance_of_PC1*100)
print("percent variance by PC1 : ",variance_of_PC2*100)


plotOn_ith_Eigenvec(data,w,0)
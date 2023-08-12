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

def polynomial_kernel(x,y,p):
    value=(x.transpose() @ y) + 1
    value=value**p;
    return value;

def gaussian_kernel(x,y,sigma):
    sub_val=x-y
    power_term=-(sub_val.transpose() @ sub_val)/(2*(sigma**2))
    value=np.exp(power_term)
    return value

def computePolynomilaKernelMatrix(data,polynomial):
    size=data.shape[1]
    K=np.zeros([size,size])
    for i in range(size):
        for j in range(size):
            K[i][j]=polynomial_kernel(data[:,i],data[:,j],polynomial)
    
    return K

def computeGaussianKernelMatrix(data,sigma):
    size=data.shape[1]
    K=np.zeros([size,size])
    for i in range(size):
        for j in range(size):
            K[i][j]=gaussian_kernel(data[:,i],data[:,j],sigma)
    
    return K
    
def centerKernelMatrix(K):
    n=K.shape[0];
    In = np.full((n,n), 1/n)
    K=K - In @ K - K @ In + In@K@In
    return K

def normalizeEigenVecor(eigenValue,beta,k): 
    
    sorted_eigen=np.sort(eigenValue)[::-1]
    for i in range(k):
        beta[i]=beta[i]/(sorted_eigen[i] ** 0.5)
    return beta

def KPCA(K,title):
    K=centerKernelMatrix(K)
    K_eigenvalue,K_eigenvector=findEigen(K)
    beta=findTopKEigenVec(K_eigenvector,K_eigenvalue,2)
    alpha=normalizeEigenVecor(K_eigenvalue,beta,2)
    alpha[0].shape
    n=alpha[0].shape[0]
    
    new_X = K @ alpha[0]
    new_Y = K @ alpha[1]
    
    plt.figure(figsize=(10, 10))
    plt.title(title,fontsize=20)
    plt.scatter(new_X,new_Y,label="Data")
    plt.xlabel('new X', fontsize=15)
    plt.ylabel('new Y', fontsize=15)
    plt.legend(loc="upper left",fontsize=10)
    plt.show()


data=loadData()
# print(data.head(10))

print("mean of data :")
print(data.mean())

# data=loadData()
data=changeDataStructure(data)
# data=dataCenter(data)
plt.figure(figsize=(10, 10))
plt.scatter(data[0],data[1],label="Data")

plt.title("Dataset",fontsize=20)
plt.xlabel("X Data",fontsize=15)
plt.ylabel("Y Data",fontsize=15)
plt.legend(loc="upper left",fontsize=15)
plt.show()


polynomials=[1,2,3]
sigmas=np.arange(0.1,1.1,0.1)

for polynomial in polynomials:
    K=computePolynomilaKernelMatrix(data,polynomial)
    title="Polynomial Kernel of p = "+str(polynomial)
    KPCA(K,title)


for sigma in sigmas:
    K=computeGaussianKernelMatrix(data,sigma)
    title="Gaussian Kernel of sigma = "+str(sigma)
    KPCA(K,title)

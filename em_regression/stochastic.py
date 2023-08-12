import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def errorFunction(X,w,y):
    errorVector=X.transpose()@w - y
    error = errorVector.transpose() @ errorVector
    return error;


def converge(w1,w2):
    if np.linalg.norm(w1-w2)**2 <= 0.1 :
        return True
    return False

def derivative(X,y,w):
    return 2*(((X @ X.transpose()) @ w) - (X@y))

def computeNewW(X,y,w,step):
    der=derivative(X,y,w)
#     newW = w - step*(der)/np.linalg.norm(der)
    newW=w - step*der
    return newW;
def plot(x,y,xlabel="X",ylabel="Y",title=""):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    
def diffWithMl(w):
    diff= w_ml-w
#     return diff.transpose() @ diff
    return np.linalg.norm(diff)

def stochasticGradientDecent(X,y,step,batchSize):
    t=0
    iterations=[]
    errors=[]
    diffMl=[]
    count=0
    np.random.seed(2)
    w=np.random.rand(100,1)
    s=w
    count+=1
    prevError = errorFunction(X,w,y);
    w_new=w
    prevW=w
    while(True):
        iterations.append(t)
        errors.append(prevError[0][0])
        diffMl.append(diffWithMl(w_new))
        
        indexes=np.random.randint(X.shape[1], size=batchSize)
#         print(indexes)
        X_batch=X[:,indexes]
        y_batch=y[indexes,:]
        
#         print(X_batch,y_batch)
        w_new=computeNewW(X_batch,y_batch,w_new,step)
        s=s+w_new
        count+=1
        currError=errorFunction(X,w_new,y);
        if(currError[0][0] == np.inf or abs(prevError - currError) <= 0.00001):
#             print("hello")
            break
#         if(converge(w_new,prevW)):
#             break
        prewW=w_new
        prevError=currError
        t+=1

    
    plot(iterations,errors,"iterations","Squared Error","iteration vs error")
    plot(iterations,diffMl,"iterations","| Wᵗ - Wₘₗ |","iteration vs | Wᵗ - Wₘₗ |")
    w_avg=s/count
    
    return w_avg;


trainData=pd.read_csv("A2Q2Data_train.csv",header=None)
testData=pd.read_csv("A2Q2Data_test.csv",header=None)
X=trainData.iloc[:,:-1]
X_test=testData.iloc[:,:-1]
y=trainData.iloc[:,-1].to_frame()
y_test=testData.iloc[:,-1].to_frame()

X=X.to_numpy()
y=y.to_numpy()
X_test=X_test.to_numpy()
X_test=X_test.transpose()
X=X.transpose()
y_test=y_test.to_numpy()
w_ml = np.linalg.inv(X @ X.transpose()) @ X @y


batchSize=100
step=0.000039
# step=0.01
w=stochasticGradientDecent(X,y,step,batchSize)
print(errorFunction(X,w,y))
print(errorFunction(X_test,w,y_test))
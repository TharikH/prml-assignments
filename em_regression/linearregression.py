import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def errorFunction(X,w,y):
    errorVector=X.transpose()@w - y
    error = errorVector.transpose() @ errorVector
    return error

def converge(w1,w2):
    if np.linalg.norm(w1-w2)**2 <= 0.0000001 :
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
    
# commented line conains various trial and errors of different methods
def gradientDecent(X,y,step):
    t=0
    iterations=[]
    errors=[]
    diffMl=[]
    np.random.seed(2)
    w=np.random.rand(100,1)
#     w=np.zeros([100,1])
    prevError = errorFunction(X,w,y);
    prevW=w
    while(True):
        w=computeNewW(X,y,w,step)
        currError=errorFunction(X,w,y);
        if(currError[0][0] == np.inf):
            break
        iterations.append(t)
        errors.append(prevError[0][0])
        diffMl.append(diffWithMl(w))
        if(abs(prevError - currError) <= 0.00001):
            break
#         if(converge(w,prevW)):
#             break
        prevW=w
        prevError=currError
        t+=1
#         step=(1/t)*0.0005

    plot(iterations,errors,"iterations","Squared Error","iteration vs error")
    plot(iterations,diffMl,"iterations","| Wᵗ - Wₘₗ |","itertaion vs | Wᵗ - Wₘₗ |")
    return w









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

# analytical solution

w_ml = np.linalg.inv(X @ X.transpose()) @ X @y
print("analytical solution is : ")
print(errorFunction(X,w_ml,y))
print(errorFunction(X_test,w_ml,y_test))

#gradient decent
print("gradient solution is : ")
step=0.0000039
# step=1/60000
w=gradientDecent(X,y,step)    
print(errorFunction(X,w,y))
print(errorFunction(X_test,w,y_test))




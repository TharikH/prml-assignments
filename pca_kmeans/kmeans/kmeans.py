import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from matplotlib.lines import Line2D


LABEL_COLOR_LIST=["b","g","r","c","m","y","k","w"]

def loadData():
    data = pd.read_csv("Dataset.csv",header=None)
    data.columns=["X","Y"]
    return data

def changeDataStructure(data):
    data_numpy=data.to_numpy()
    data_numpy=data_numpy.transpose()
    return data_numpy

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

#check prev indicators are same or not. If same we can stop kmeans iteration
def isEqual(arr1,arr2):
    n=len(arr1)
    for i in range(n):
        if arr1[i] != arr2[i]:
            return False
    return True

def plotKmeans(data,indicator,means,k,randinit):
    color_label=[LABEL_COLOR_LIST[i] for i in indicator]
    
    plt.figure(figsize=(10, 10))
    plt.scatter(data[0],data[1],c=color_label,label="Dataset")
    plt.scatter(means[0],means[1],c=LABEL_COLOR_LIST[0:k],s=1000,label="Means")
    

    plt.title("K-means with k = "+str(k)+ " with random initialization - "+str(randinit),fontsize=15)
    plt.xlabel("X Data",fontsize=15)
    plt.ylabel("Y Data",fontsize=15)
    

    line1 = Line2D([], [], color="white", marker='o', markerfacecolor="white",markeredgecolor="black",markersize=5)
    line2 = Line2D([], [], color="white", marker='o', markerfacecolor="white",markeredgecolor="black",markersize=10)
    plt.legend((line1, line2), ('Datapoints', 'Means'), numpoints=1, loc="upper left",fontsize=15)

#     plt.legend(loc="upper left",fontsize=20)

    plt.show()

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
    plt.plot(iteration,error,c=LABEL_COLOR_LIST[k])
    plt.show()

def Kmeans(data,k,randinit):
    n=data.shape[0]
    m=data.shape[1]
    means=np.zeros([k,n])
    indicator=[0]*m
    trans_data=data.transpose()
    
    #randomly initialize means
    for i in range(k):
        random.seed(i+randinit)#seeds are used so as to fix random initialization
        val=random.choice(trans_data)
        if val not in means:
            means[i]=val
        else:
            i=i-1
    means=means.transpose()#change to n x k matrix
    
    count=0
    iteration=[]
    error=[]
    iteration.append(count)
    error.append(calculateError(data,indicator,means))
    count+=1
    
    prev_indicator=copy.deepcopy(indicator)#to store curr assignments so as o compare whether next assignment converged or not
#     plotKmeans(data,indicator,means,k)
    indicator=assignMean(data,indicator,means)
#     plotKmeans(data,indicator,means,k)

    #run the loop till kmeans converge
    while(not isEqual(prev_indicator,indicator)):
        
        prev_indicator=copy.deepcopy(indicator)
        means,err_sum=computeMean(data,means,indicator)
        
#         plotKmeans(data,indicator,means,k)
        iteration.append(count)
        error.append(err_sum)
        count+=1
        
        indicator=assignMean(data,indicator,means)
#     plotKmeans(data,indicator,means,k,randinit)
    return means,indicator,iteration,error

def findIndicatorOfVoronoi(x,y,means):
    indicator=[]
    size=len(x);
    
    for i in range(size):
        cord=np.array([x[i],y[i]])
        indicator.append(getIndicatorOfX(cord,means,0));
    
    return indicator
def voronoiRegions(means):
    x=[]
    y=[]
    
    #create 
    for i in np.arange(-10,10,0.05):
        for j in np.arange(-10,10,0.05):
            x.append(i)
            y.append(j)
    
    indicator=findIndicatorOfVoronoi(x,y,means)
    LABEL_COLOR_LIST=["b","g","r","c","m","y","k","w"]
    color_label=[LABEL_COLOR_LIST[i] for i in indicator]
    
    plt.figure(figsize=(10, 10))
    
    plt.xlabel("X Ranges",fontsize=15)
    plt.ylabel("Y Ranges",fontsize=15)
    plt.title("Voronoi region with k = "+str(means.shape[1]),fontsize=15)
    
    plt.scatter(x,y,c=color_label)
    plt.scatter(means[0],means[1],c="black",s=100,label="Means")
    
    
    plt.legend(fontsize=15,loc='upper left', bbox_to_anchor=(0.05, 0.96),
          ncol=3, fancybox=True, shadow=True, borderpad=0.4)
    plt.show()




data=loadData()
# print(data.head(10))


data=loadData()
data=changeDataStructure(data)

# plot dataset
plt.figure(figsize=(10, 10))
plt.scatter(data[0],data[1],label="Data")
plt.title("Dataset",fontsize=20)
plt.xlabel("X Data",fontsize=15)
plt.ylabel("Y Data",fontsize=15)
plt.legend(loc="upper left",fontsize=15)
plt.show()

#####
# Question 2.1
####

k=4
for i in range(0,5):
    print("Random initialization " + str(i))
    means,indicator,iteration,error=Kmeans(data,k,i)
    plotKmeans(data,indicator,means,k,i)
    plotIteration(iteration,error,k,i)


####
# Question 2.2
###
K_value=[2,3,4,5]

for k in K_value:
    means,indicator,_,_=Kmeans(data,k,0)
    plotKmeans(data,indicator,means,k,0)
    voronoiRegions(means)


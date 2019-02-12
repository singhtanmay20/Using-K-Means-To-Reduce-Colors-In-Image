UBIT = 'tanmaypr';
from copy import deepcopy
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))
import pandas as pd
from matplotlib import pyplot as plt
import cv2

def dist(a, b,axis):# function to calculate euclidean distance
    return np.linalg.norm(a - b, axis=axis)

def myKMean(n,z):
    width, height, depth = z.shape
    reshaped_z = np.reshape(
    z, (width * height, depth))
    a=0
    labels=[]
    label=[0 for j in range(len(reshaped_z))]
    for i in range(n):
        labels.append(a)
        a+=1
    print(labels)
    x=np.array
    x=reshaped_z
    x = (x-x.min())/(x.max()-x.min())
    print(x)
    print(reshaped_z.shape)
    f1=x[:,:1]
    f2=x[:,1:2]
    f3=x[:,2:]
    
    number_of_clusters=n
    #center_of_clusters=np.array
    center_of_clusters=np.random.randint(n, size=(n,3))
    center_of_clusters=x[:n,:]
    c1_x=center_of_clusters[:,:1]
    c2_x=center_of_clusters[:,1:2]
    c3_x=center_of_clusters[:,2:]
    print(center_of_clusters)
    
    #print(distance_from_centroid)
    
    c_old=np.zeros(center_of_clusters.shape)
    error = dist(center_of_clusters, c_old,None)
    print(error)
    
    distance_from_centroid=[]
    while error!=0:
        clusters=[]
        for i in range(len(x)):
            distance_from_centroid=[]
            for j in range(number_of_clusters):
                #print(dist(x[i],center_of_clusters[j],axis=None))
                distance_from_centroid.append(dist(x[i],center_of_clusters[j],axis=None))
            cluster=distance_from_centroid.index(min(distance_from_centroid))
            clusters.append(cluster)
       
        c_old=deepcopy(center_of_clusters)
        x *= 255.0/x.max()
        for i in range(number_of_clusters):
            points=[]
            for j in range(len(x)):
                if clusters[j]==i:
                    points.append(x[j,:])
                    label[j]=labels[i]  
            #print(points)
            center_of_clusters[i] = np.mean(points, axis=0)
        
        #print(center_of_clusters)
        error = dist(center_of_clusters, c_old, None)
    return center_of_clusters,clusters,label,x
    


# In[4]:


img = cv2.imread('/home/tanmay/Documents/cvip/proj2/data/baboon.jpg')
width, height, depth = img.shape
Z = np.float32(img)
K = 20
print(Z.shape)
#plt.scatter(Z[:,0:1],Z[:,1:2],facecolors='none',edgecolors='b',marker="^") 
center_of_clusters,clusters,label,x=myKMean(K,Z)
center_of_clusters = np.uint8(center_of_clusters)
label=np.asarray(label)
print(label.shape)
#print(label)
#print(center_of_clusters)
res=center_of_clusters[label]
print(center_of_clusters[0])
print(res.shape)
res2=res.reshape((img.shape)).astype('uint8')
res2=np.asarray(res2)
#print(res2)
cv2.imwrite("/home/tanmay/Documents/cvip/proj2/task3_baboon_20.jpg",res2)





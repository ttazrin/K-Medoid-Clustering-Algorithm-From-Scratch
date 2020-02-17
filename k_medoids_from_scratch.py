# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:32:30 2019

@author: Tazrin
"""

import pandas as pd
import numpy as np
import sklearn.metrics
from statistics import mean

data = pd.read_csv("data.csv")
k = 3   #Change this value for changing number of clusters
#calculate dissimilarity matrix
distanceMat = sklearn.metrics.pairwise_distances(data.iloc[:, :], Y=data.iloc[:, :], metric='euclidean')
sumDist = sum(distanceMat[:,:])
#randomly choose center
centers = data.sample(n=k)
centerIdx = []
for i in centers.itertuples():
    centerIdx.append(i[0])
    

#clustering
for iter in range(100):
    cluster = []
    #assigning point to closest cluster
    for i in data.itertuples():
        minDist = np.argmin(distanceMat[i[0],centerIdx])
        cluster.append(minDist)
    #calculate new center
    newCenter = []
    for i in range(k):
        eachCluster = [[sumDist[j],j] for j in range(len(cluster)) if cluster[j] == i]
        m = eachCluster[np.argmin([i[0] for i in eachCluster])]
        if m[1] not in newCenter:
            newCenter.append(m[1])

    if newCenter == centerIdx:
        print(iter)
        break
    else:
        centerIdx = newCenter

newData = pd.read_csv("data.csv")
newData['cluster'] = cluster
x = 'cluster_' + str(k) + '.csv'
newData.to_csv(x)

#silhouette width
silWidth = []
for oneCluster in range(k):
    eachClusterSW = [j for j in range(len(cluster)) if cluster[j] == oneCluster]
    SW = []
    #calculate silhouette width for each point in a cluster
    for i in eachClusterSW:
        #a
        Alist = distanceMat[70,eachClusterSW]
        a = sum(Alist)/(len(Alist)-1)
        #b
        avgDist = []
        for l in range(k):
            if l!=oneCluster:
                otherClusterSW = [distanceMat[i,j] for j in range(len(cluster)) if cluster[j] == l]
                avgDist.append(mean(otherClusterSW))
        b = min(avgDist)
        sw = (b-a)/max(a,b)
        SW.append(sw)
    silWidth.append(mean(SW))

print('Silhouette width for each cluster',silWidth)
print('Average silhouette width', mean(silWidth))

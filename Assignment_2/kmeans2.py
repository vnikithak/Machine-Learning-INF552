import numpy as np
import pandas as pd
from collections import defaultdict
class KMeans:
    def __init__(self,k=3,maxIterations=300):
        self.k=k
        self.maxIterations=maxIterations
    def calcDistance(self,a,b):
        return np.linalg.norm(a-b)
    def cluster(self,dat):
        self.centroids = []
        for i in range(self.k):
            self.centroids.append(dat[np.random.randint(0,np.size(dat,0))])
        for i in range(self.maxIterations):
            self.clusters = defaultdict(list)
            for row in dat:
                #find distance to each centroids
                euclidean_distance=[]
                for c in self.centroids:
                    euclidean_distance.append(self.calcDistance(row,c))
                #find minimum distances
                idx=0
                minDistance=euclidean_distance[0]
                for j,dist in enumerate(euclidean_distance):
                    if dist<minDistance:
                        idx=j
                        minDistance=dist
                self.clusters[idx].append(row)
            #recompute the centroids
            for i in range(self.k):
                self.centroids[i]=np.average(self.clusters[i],axis=0)
        for x in range(self.k):
                print(self.centroids[x])



def main():
    k=3
    train_dat = np.genfromtxt('clusters.csv', delimiter=',')
    km=KMeans()
    km.cluster(train_dat)

if __name__=="__main__":
    main()

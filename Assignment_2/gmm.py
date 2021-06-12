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
        return self.clusters


class GMM():
    def __init__(self,num_clusters=3,maxIterations=300):
        self.num_clusters=num_clusters
        self.maxIterations=maxIterations
        self.means=[]
        self.sigma=[]
        self.pi=[]
    def cluster(self,dat):
        #initialize the means,covariance, amplitude
        k=KMeans()
        clusters=k.cluster(dat)
        membership=None
        self.pi=np.ones(self.num_clusters)/self.num_clusters
        for c in range(self.num_clusters):
            self.means.append(np.mean(clusters[c],axis=0))
            self.sigma.append(np.cov(np.array(clusters[c]).T))
        for i in range(self.maxIterations):
            membership1=self.estep(dat)
            if membership is not None and membership1 is not None:
                if (np.abs(membership-membership1)==0).all():
                    break
            membership=membership1
            self.mstep(dat,membership)

    def mstep(self,dat,membership):
        #calculate means,covariance and amplitude
        self.pi=np.sum(membership,axis=0)/len(dat)
        self.means=[]
        self.sigma=[]
        num_points=len(dat)
        d=len(dat[0])
        for i in range(self.num_clusters):
            self.means.append(np.sum(np.multiply(dat,membership[:,i].reshape(num_points,1)),axis=0)/np.sum(membership[:,i]))
            cov=np.zeros((d,d))
            for j in range(num_points):
                temp=dat[j]-self.means[i]
                temp=np.dot(temp.T.reshape(d,1),temp.reshape(1,d))
                temp=temp*membership[j][i]
                cov=np.add(cov,temp)
            cov=cov/np.sum(membership[:,i])
            self.sigma.append(cov)
    def estep(self,dat):
        n=len(dat)
        d=np.size(dat,1)
        r=np.zeros((n,self.num_clusters))
        for c in range(self.num_clusters):
            mean=self.means[c]
            cov=self.sigma[c]
            frac=1/((2*np.pi)**(d/2))
            det=np.linalg.det(cov)
            if det==0:
                det+=0.0000001
                covinv=np.multiply(np.linalg.pinv(np.multiply(cov.T,cov)),cov.T)
            else:
                covinv=np.linalg.inv(cov)
            frac=frac/np.sqrt(det)
            for i in range(n):
                power=(dat[i,:]-mean).T
                power=np.dot(-0.5*power,covinv)
                power=np.dot(power,dat[i,:]-mean)
                r[i][c]=frac*np.exp(power)
        for i in range(n):
            denom=np.sum(self.pi*r[i])
            for c in range(self.num_clusters):
                r[i][c]=self.pi[c]*r[i][c]/denom
        return r







def main():
    k=3
    train_dat = np.genfromtxt('clusters.csv', delimiter=',')
    gm=GMM()
    gm.cluster(train_dat)
    for i in range(3):
        print("Mean of cluster"+str(i+1)+":")
        print(gm.means[i])
        print("Covariance of cluster"+str(i+1)+":")
        print(gm.sigma[i])
        print("Amplitude of cluster"+str(i+1)+":")
        print(gm.pi[i])

if __name__=="__main__":
    main()

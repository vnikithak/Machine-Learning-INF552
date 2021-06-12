"""
Group members:

Naureen Firdous
Nikitha Krishna Vemulapalli
Vijayantika Inkulla

"""

import numpy as np
import pandas as pd

class PCA:
    def __init__(self,components=2):
        self.components=components

    def find_components(self,dat):
        mean=np.mean(dat,axis=0)
        dat=dat-mean
        covariance=(np.dot(dat.T,dat))/(len(dat)-1)

        #compute eigen values and vectors

        evalues,evectors=np.linalg.eig(covariance)
        val=np.argsort(-evalues)[:2]
        self.evectors_trunc=evectors[...,val]

        #calculate the new points using truncated version of U

        self.new_vectors=np.ones((dat.shape[0],self.components))
        for i in range(dat.shape[0]):
            self.new_vectors[i]=np.dot(self.evectors_trunc.T,dat[i])
        return self.evectors_trunc.T,self.new_vectors


def main():
    input_data=np.loadtxt("pca_data.txt",dtype='float',delimiter="\t")
    pca=PCA()
    principal_comps,tranformed_points=pca.find_components(input_data)
    print(principal_comps)

if __name__=="__main__":
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
def main():
    k=3
    train_dat = np.genfromtxt('clusters.csv', delimiter=',')
    print(train_dat)
    km=KMeans(n_clusters=3)
    km.fit(train_dat)
    print(km.cluster_centers_)

if __name__=="__main__":
    main()

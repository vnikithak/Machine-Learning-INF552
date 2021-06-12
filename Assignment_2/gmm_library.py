from sklearn.mixture import GaussianMixture
import numpy as np
def main():
    k=3
    train_dat = np.genfromtxt('clusters.csv', delimiter=',')
    gm=GaussianMixture(n_components=3)
    gm.fit(train_dat)
    print(gm.means_)
    print(gm.covariances_)
    print(gm.weights_)
if __name__=="__main__":
    main()

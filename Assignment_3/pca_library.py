"""
Group members:

Naureen Firdous
Nikitha Krishna Vemulapalli
Vijayantika Inkulla

"""

import numpy as np
from sklearn.decomposition import PCA

input_data=np.loadtxt("pca_data.txt",dtype='float',delimiter="\t")
pca= PCA(n_components=2)
pca.fit_transform(input_data)

print(pca.components_)

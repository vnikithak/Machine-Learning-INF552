"""
Group members:
Vijayantika Inkulla
Naureen Firdous
Nikitha Krishna Vemulapalli
"""
import numpy as np
from sklearn.linear_model import Perceptron

def main():
    input_data=np.genfromtxt("classification.txt",delimiter=",")
    X=input_data[:,:3]
    Y=input_data[:,3]
    p=Perceptron(max_iter=7000)
    p.fit(X,Y)
    print("W0(threshold):",p.intercept_)
    print("Weights:\n",p.coef_)
    print("Accuracy:",p.score(X,Y)*100)

if __name__=="__main__":
    main()

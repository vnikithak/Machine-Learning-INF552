"""
Group members:
Vijayantika Inkulla
Naureen Firdous
Nikitha Krishna Vemulapalli
"""

import numpy as np
import random
import matplotlib.pyplot as plt
class Perceptron:
    def __init__(self,dimensions=3):
        self.dimensions=dimensions
    def calculateWeights(self,X,Y):
        x0=np.ones((len(X),1))
        X=np.concatenate((x0,X),axis=1)
        learning_rate=0.01
        k=random.randint(0,len(X)-1)
        weights=X[k]
        misclassifications_list=[]
        iter=0
        for iter in range(7000):
            #print(iter)
            for i in range(len(X)):
                xi=X[i]
                s=np.dot(weights.T,xi)
                yi=Y[i]
                if s<0 and yi==1:
                    weights=weights+learning_rate*xi
                    endLoop=False
                    break
                elif s>=0 and yi==-1:
                    weights=weights-learning_rate*xi
                    endLoop=False
                    break
            misclassifications=0
            for i in range(len(X)):
                s=np.dot(weights.T,X[i])
                if s<0 and yi==1:
                    misclassifications+=1
                elif s>=0 and yi==-1:
                    misclassifications+=1
            accuracy=((len(X)-misclassifications)/len(X))*100
            misclassifications_list.append(misclassifications)
        return weights,misclassifications_list,accuracy
def main():
    input_data=np.genfromtxt("classification.txt",delimiter=",")
    d=3
    X=input_data[:,:3]
    Y=input_data[:,4]
    p=Perceptron()
    weights,misclassifications_list,accuracy=p.calculateWeights(X,Y)
    print("Weights:\n",weights)
    print("Accuracy:",accuracy)
    plt.plot(np.arange(0,7000),misclassifications_list)
    plt.show()
if __name__=="__main__":
    main()

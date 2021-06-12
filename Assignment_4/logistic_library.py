"""
Group members:
Vijayantika Inkulla
Naureen Firdous
Nikitha Krishna Vemulapalli
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    input_data=np.genfromtxt("classification.txt",delimiter=",")
    X=input_data[:,:3]
    Y=input_data[:,4]
    l = LogisticRegression()
    l.fit(X, Y)
    print("W0(threshold):",l.intercept_)
    print("Weights:\n",l.coef_)
    YPred = l.predict(X)
    correct = np.where(Y == YPred)[0].shape[0]
    tot = YPred.shape[0]
    print("Accuracy:",(correct/tot)*100)

if __name__=="__main__":
    main()

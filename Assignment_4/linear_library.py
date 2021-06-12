"""
Group members:
Vijayantika Inkulla
Naureen Firdous
Nikitha Krishna Vemulapalli
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    input_data=np.genfromtxt("linear_regression.txt",delimiter=",")
    X=input_data[:,:-1]
    Y=input_data[:,-1]
    l=LinearRegression()
    l.fit(X,Y)
    print("W0(threshold):",l.intercept_)
    print("Weights:\n",l.coef_)

if __name__=="__main__":
    main()

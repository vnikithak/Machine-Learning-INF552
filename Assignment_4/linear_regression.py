"""
Group members:
Vijayantika Inkulla
Naureen Firdous
Nikitha Krishna Vemulapalli
"""
import numpy as np

class LinearRegression:

	def __init__(self):
		self.weights=None

	def calculateWeights(self,D,Y):
		x0=np.ones((len(D),1))
		D=np.concatenate((x0,D),axis=1)
		A=np.linalg.inv(np.dot(D.T,D))
		B=np.dot(D.T,Y)
		self.weights=np.dot(A,B)
		return self.weights

def main():
	input_data=np.genfromtxt("classification.txt",delimiter=",")
	D=input_data[:,:2]
	Y=input_data[:,2]
	l=LinearRegression()
	weights=l.calculateWeights(D,Y)
	print("Weights:\n",weights)

if __name__=="__main__":
	main()

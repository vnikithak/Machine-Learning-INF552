"""
Group members:
Vijayantika Inkulla
Naureen Firdous
Nikitha Krishna Vemulapalli
"""
import numpy as np
class LogisticRegression:
	def __init__(self):
		self.wt=None
		self.wt_D=None
		self.prob=None

	def calculateWeights(self,D,Y):
		x0=np.ones((len(D),1))
		D=np.concatenate((x0,D),axis=1)
		epochs=7000
		gradient=np.zeros(4)
		self.wt=np.random.rand(D.shape[1])
		eta=0.01
		for i in range(epochs):
			for j in range(len(Y)):
				denom=1+np.exp(np.dot(np.dot(D[j],self.wt.T),Y[j]))
				gradient=gradient+(np.dot(Y[j],D[j])/denom)
			gradient=-(gradient/len(Y))
			self.wt=self.wt-(eta*gradient)
			self.wt_D=np.dot(D,self.wt.T)
		self.prob=np.exp(self.wt_D)/(1+np.exp(self.wt_D))
		self.prob[self.prob>0.5]=1
		self.prob[self.prob<0.5]=-1
		return (self.wt, self.wt_D, self.prob)

	def calculateAccuracy(self, Y):
		count=0
		for i in range(len(Y)):
			if self.prob[i]==Y[i]:
				count+=1
		accuracy=(count/len(Y))*100
		return accuracy

def main():
	input_data=np.genfromtxt("classification.txt",delimiter=",")
	D=input_data[:,0:3]
	Y=input_data[:,4]
	l=LogisticRegression()
	weights,weights_X,probabilities=l.calculateWeights(D,Y)
	print("Weights:\n",weights)
	print("Probabilities:\n",probabilities)
	accuracy=l.calculateAccuracy(Y)
	print("Accuracy:",accuracy)

	
if __name__ == "__main__":
	main()

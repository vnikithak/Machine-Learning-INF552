import numpy as np
import cvxopt
import matplotlib.pyplot as plt

class LinearSVM:
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y
		self.rows, self.cols = X.shape

	def fit(self):
		Q = np.zeros((self.rows, self.rows))
		for i in range(self.rows):
			for j in range(self.rows):
				Q[i,j] = np.dot(self.X[i].T,self.X[j])
		Q=np.outer(self.Y, self.Y)*Q

		#parameters for cvxopt.solvers.qp() to solve QPP
		P = cvxopt.matrix(Q)
		q = cvxopt.matrix(np.ones(self.rows)*-1)
		G = cvxopt.matrix(np.diag(np.ones(self.rows)*-1))
		h = cvxopt.matrix(np.zeros(self.rows))
		A = cvxopt.matrix(self.Y, (1, self.rows))
		b = cvxopt.matrix(0.0)
		#solve QPP
		alpha= np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x']).reshape(1, self.rows)[0]
		self.sv_index = np.where(alpha>0.00001)[0]
		self.alpha = alpha[self.sv_index]
		self.sv = self.X[self.sv_index]
		self.sv_y = self.Y[self.sv_index]
		print(len(self.alpha), "support vectors out of", self.rows, "points")
		self.weights = np.zeros(self.cols)
		for i in range(len(self.alpha)):
			self.weights += self.alpha[i] * self.sv_y[i] * self.sv[i]
		self.b = self.sv_y[0] - np.dot(self.weights, self.sv[0])

def main():
	
	input_data = np.loadtxt('linesep.txt', dtype = 'float', delimiter =',')
	X = input_data[:,:2]
	Y = input_data[:,2]
	svm = LinearSVM(X, Y)
	svm.fit()
	print("Value of b:")
	print(svm.b)
	print("Weights:")
	print(svm.weights)
	print("Support vectors")
	print(svm.sv)
	plt.scatter(X[:,0],X[:,1], c=Y, cmap = 'bwr', alpha = 1, s=50, edgecolors = 'k')
	plt.scatter(svm.sv[:,0], svm.sv[:, 1], facecolors = 'none', s = 100, edgecolors = 'k')
	left = -(svm.weights[0] * (-1) + svm.b)/svm.weights[1]
	right = -(svm.weights[0] * 1 + svm.b)/svm.weights[1]
	plt.plot([-1,1], [left,right])
	plt.show()

if __name__ == "__main__":
	main()


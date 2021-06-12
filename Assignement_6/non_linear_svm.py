import numpy as np
import matplotlib.pyplot as plt
import cvxopt

class NonLinearSVM:

	def calc_z(self, x, y):
		#using polynomial kernel
		return [1, x**2, y**2, np.sqrt(2)*x,  np.sqrt(2)*y, np.sqrt(2)*x*y]

	def kernel(self,X1,X2):
		#for two points X1 and X2 calculate Z1-transpose.Z2
		return np.dot(self.calc_z(X1[0],X1[1]),self.calc_z(X2[0],X2[1]))

	def fit(self, X, Y):

		N = X.shape[0] #number of data points
		Q = np.zeros((N,N)) #Q matrix where Q=summation(yi*yj*zi.T*zj)
		for i in range(N):
			for j in range(N):
				Q[i,j] = self.kernel(X[i],X[j])
		Q=np.outer(Y,Y)*Q
        #parameters for solving quadratic programming using cvxopt.solver
		P = cvxopt.matrix(Q)
		q = cvxopt.matrix(np.ones(N) * -1)
		G = cvxopt.matrix(np.diag(np.ones(N) * -1))
		h = cvxopt.matrix(np.zeros(N))
		A = cvxopt.matrix(Y,(1,N))
		b = cvxopt.matrix(0.0)
		#solve using quadratic progarmming for lagrange variables
		alpha= np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x']).reshape(1,N)[0]
		sv_index = np.where(alpha>0.00001)[0]
		#print(sv_index)
		self.alpha = alpha[sv_index]
		self.sv=np.ones((len(sv_index),6))
		for i,v in enumerate(sv_index):
			self.sv[i] = self.calc_z(X[v][0],X[v][1])
		self.sv_y = Y[sv_index]
		print("%d support vectors out of %d points" % (len(self.alpha), N))
		self.weights = np.zeros(6)
		for i in range(len(self.alpha)):
			self.weights += self.alpha[i] * self.sv_y[i] * self.sv[i]
		self.b = self.sv_y[0] - np.dot(self.weights, self.sv[0])
def main():
	data=np.loadtxt('nonlinsep.txt',dtype='float',delimiter=',')
	X=data[:,:2]
	Y=data[:,2]
	plt.scatter(data[:,0],data[:,1],c=Y)
	plt.show()
	svm=NonLinearSVM()
	svm.fit(X,Y)
	print("Value of b:")
	print(svm.b)
	print("Weights:")
	print(svm.weights)
	print("Support vectors:")
	print(svm.sv)

if __name__=="__main__":
    main()

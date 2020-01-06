import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AdalineGD:
	def __init__(self, epoch = 100, learn_rate = 0.01, random_state = 1):
		# since updates to weights are real numbers it needs more iterations to converge as compared to Perceptron
		self.epoch = epoch
		self.learn_rate = learn_rate
		self.random_state = random_state


	def fit(self, X, y):
		'''

		cost function
			J(w) = 1/2 * sum(y - @(z)) ** 2
				here, z = x * w
		to find the minimum of gradient
			diff(J(w)) = -(sum(y-@(z)))*x
		we move in the opposite direction of this with learning rate
			step = - learn_rate * diff(J(w))
			therefore,
				step = learn_rate * sum(y - @(z)) * x

		'''
		rgen = np.random.RandomState(self.random_state)
		# initialise weights with mean 0 and sd 1
		self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1] + 1)
		self.cost_ = []

		for i in range(self.epoch):
			# x * w
			net_input = self.net_input(X)
			# same thing
			output = self.activation(net_input)
			errors = (y - output)
			self.w_[1:] += self.learn_rate * X.T.dot(errors)
			self.w_[0] += self.learn_rate * errors.sum()
			# j(w) = 1/2* sum(y-@(z))^2
			cost = (errors**2).sum()/2.0
			self.cost_.append(cost)
		return self


	def net_input(self, X):
		return (np.dot(X, self.w_[1:]) + self.w_[0])


	def activation(self, X):
		'''
			this functions is there to show how information flows in more complex algorithms
			we can also ommit this function for adaline
		'''
		return X


	def predict(self, X):
		return np.where(self.activation(self.net_input(X))>=0.0, 1, -1)

def main():
	# A = AdalineGD()
	# test example
	#  I have tried it with my own example to see how it works, and it does the job pretty well
	# X = resident, 18+, male, married
	X = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[1,1,0,0],[1,0,1,0],[1,1,0,0]])
	# y = can vote
	y = np.array([-1, -1, -1, -1, 1, -1, 1])
	
	'''
	# train
	A.fit(X, y)
	# predict
	X = np.array([[1,1,1,1],[1,1,1,0],[1,0,1,1],[0,1,1,1]])
	print(A.predict(X))
	'''

	# Standardization
	# x = (x - mean) / sd
	# althought it is already standardised
	X_std = X.copy()
	X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:,0].std()
	X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:,1].std()
	X_std[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:,2].std()
	X_std[:, 3] = (X[:, 3] - X[:, 3].mean()) / X[:,3].std()

	# lets visualise the convergence
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,4))

	ada1 = AdalineGD(epoch=50, learn_rate = 0.01).fit(X_std,y)
	ax[0].plot(range(1, len(ada1.cost_)+1),
					np.log10(ada1.cost_), marker = 'o')
	ax[0].set_xlabel('Epochs')
	ax[0].set_ylabel('log(Sum-Squared-Error)')
	ax[0].set_title('Adaline learn_rate : 0.01')

	ada2 = AdalineGD(epoch=50, learn_rate = 0.0001).fit(X_std,y)
	ax[1].plot(range(1, len(ada1.cost_)+1),
					np.log10(ada2.cost_), marker = 'o')
	ax[1].set_xlabel('Epochs')
	ax[1].set_ylabel('log(Sum-Squared-Error)')
	ax[1].set_title('Adaline  learn_rate : 0.0001')
	plt.show()


if __name__ == '__main__':
	main()
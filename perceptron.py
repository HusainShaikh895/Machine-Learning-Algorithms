'''
	Simple Perceptron with save() and load() functionality
	This code is my version of perceptron from 'Python Machine Learning(3rd edition) by Sebastian Raschka'

'''
__author__ = 'Husain Shaikh'

import numpy as np

class Perceptron:
	def __init__(self, epoch = 50, learn_rate = 0.01, random_seed = 1):
		# epoch is number of iterations-if data is not linearly seperable it might run forever otherwise
		self.epoch = epoch
		self.learn_rate = learn_rate
		self.random_seed = random_seed

	def fit(self, X, y, path = 'p.csv'):
		rgen = np.random.RandomState(self.random_seed)
		self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
		# Draw random samples from a normal (Gaussian) distribution.
		# loc : float or array_like of floats
		# Mean (“centre”) of the distribution.
		# scale : float or array_like of floats
		# Standard deviation (spread or “width”) of the distribution.
		# size : int or tuple of ints, optional
		self.errors_ = []
		for _ in range(self.epoch):
			errors = 0
			for xi, target in zip(X,y):
				update = self.learn_rate * (target - self.predict(xi))
				self.w_[0] += update * 1
				self.w_[1:] += update * xi
				errors += int(update != 0.0)
			self.errors_.append(errors)
		self.save(path)
		return self

	def net_input(self, X):
		''' calculate the z = w0 + w1x1 + w2x2 + ... + wnxn '''
		return (np.dot(X, self.w_[1:]) + self.w_[0])
				
	def predict(self, X, path = 'p.csv'):
		''' activation function (1 if >= 0.0 else -1) '''
		self.load(path)
		return np.where(self.net_input(X) >= 0.0, 1, -1)

	def save(self, path):
		w = np.asarray([self.w_])
		np.savetxt(path, w, delimiter=",")

	def load(self, path):
		with open(path, 'r') as fin:
			for line in fin:
				self.w_ = np.asarray(list(map(float,line.split(','))))

def main():
	P = Perceptron()
	# X = resident, 18+, male, married
	X = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[1,1,0,0],[1,0,1,0],[1,1,0,0]])
	# y = can vote
	y = np.array([-1, -1, -1, -1, 1, -1, 1])
	# fit and save
	P.fit(X, y)
	# predict
	X = np.array([[1,1,1,1],[1,1,1,0],[1,0,1,1],[0,1,1,1]])
	# load and predict
	print(P.predict(X))

if __name__ == '__main__':
	main()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class LogisticRegressionGD:
	def __init__(self, epoch = 100, learn_rate = 0.01, random_state = 1):
		''' since updates to weights are real numbers it needs more iterations to converge as compared to Perceptron'''
		self.epoch = epoch
		self.learn_rate = learn_rate
		self.random_state = random_state

	def fit(self, X, y):
		'''
		cost function
			J(w) = sum( -ylog(@(z)) - (1-y)log(1-(@(z)))   )
				here, z = x * w
				and @(z) is sigmoided values of z
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
			print(errors)
			self.w_[1:] += self.learn_rate * X.T.dot(errors)
			self.w_[0] += self.learn_rate * errors.sum()
			# j(w) = sum( -ylog(@(z)) - (1-y)log(1-(@(z)))   )
			cost = (-y.dot(np.log(output)) - 
					((1-y).dot(np.log(1-output))))
			self.cost_.append(cost)
		return self

	def net_input(self, X):
		'''
			calculate x*w
		'''
		return (np.dot(X, self.w_[1:]) + self.w_[0])

	def activation(self, z):
		'''
			Sigmoid Activation in logistic Regression
		'''
		return 1.0/ (1.0 + np.exp((-z)))

	def predict(self, X):
		return np.where(self.net_input(X)>=0.0, 1, 0)
		'''
			equivalent to 
				return np.where(self.activation(self.net_input(X))>=0.5, 1, 0))
		'''
def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.01):
	# setupe marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'green', 'blue', 'pink', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	# plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	a = np.arange(x1_min, x1_max, resolution)
	b = np.arange(x2_min, x2_max, resolution)
	xx1, xx2 = np.meshgrid(a,b)
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap=cmap)

	for idx, c1 in enumerate(np.unique(y)):
		plt.scatter(x = X[ y == c1, 0], y = X[y == c1, 1],
					alpha = 0.8, c = colors[idx], 
					marker = markers[idx], label = c1,
					edgecolor = 'black')
	# highlight test examples
	if(test_idx):
		X_test, y_test = X[test_idx, :], y[test_idx]
		plt.scatter(X_test[:, 0], X_test[:, 1],
					c = '', edgecolor='black', alpha=1.0,
					linewidth=1, marker='o',
					s = 100, label = 'test set')



def main():
	iris = datasets.load_iris()
	# sepal length, width petal length, width
	#  we will use petal length and width
	X = iris.data[:100, [2,3]]
	y = iris.target[:100]
	# print('Class labels: ', np.unique(y))

	# train_test_split() shuffles automatically before splitting
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)
	
	'''
	# stratify = y , makes sure that each array have the same proportion of classes as the original dataset
	# to check if that's true
	print('Labels counts in y:', np.bincount(y))
	print('Labels counts in y_train:', np.bincount(y_train))
	print('Labels counts in y_test:', np.bincount(y_test))
	# output :
	# Labels counts in y: [50 50 50]
	# Labels counts in y_train: [35 35 35]
	# Labels counts in y_test: [15 15 15]
	'''

	# standardise the features (feature scaling)
	sc = StandardScaler()
	sc.fit(X_train)
	X_train_std = sc.transform(X_train)
	X_test_std = sc.transform(X_test)
	# making the model
	lrgd = LogisticRegressionGD(learn_rate = 0.05, epoch = 1000, random_state = 1)
	# training
	lrgd.fit(X_train_std, y_train)
	# plotting the results
	plot_decision_regions(X = X_train_std, y = y_train, classifier = lrgd)
	plt.xlabel('Petal Length')
	plt.ylabel('petal width')
	plt.legend(loc='upper left')
	plt.show()


if __name__ == '__main__':
	main()
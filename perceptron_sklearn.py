from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.01):
	# setupe marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'green', 'blue', 'gray', 'cyan')
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
	X = iris.data[:, [2,3]]
	y = iris.target
	# print('Class labels: ', np.unique(y))

	# shuffles automatically before splitting
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


	# predict and check for misclassifications
	ppn = Perceptron(eta0=0.1, random_state = 1)
	ppn.fit(X_train_std, y_train)
	y_pred = ppn.predict(X_test_std)
	print('Miscalssified ', (y_test != y_pred).sum(),'out of', len(y_test),'examples')
	'''
		it gave us 1, which is out of 45 test examples
		hence error rate is 1/45 = 0.022
		or 2.2 %
		Many Ml practitioners report ACCURACY instead of error rate
		accuracy = 1 - error = 1 - 0.022 = 97.8%

			metrics module of sklearn also provides readily available performance metrics
	'''

	# to calculate the accuracy
	print('Accuracy Score: %.3f' %accuracy_score(y_test, y_pred))

	# to directly calculate the score(without having to store the predictions) we also have,
	# score which combines predict and accuracy_score automatically
	print('Accuracy: %.3f' %ppn.score(X_test_std, y_test))



	# plotting decision surface with scatterplot
	X_combined_std = np.vstack((X_train_std, X_test_std))
	y_combined = np.hstack((y_train, y_test))
	plot_decision_regions(X = X_combined_std, y = y_combined, classifier=ppn, test_idx = range(105,150))
	plt.xlabel('Petal length[std]')
	plt.ylabel('Petal width[std]')
	plt.legend()
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()











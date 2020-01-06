from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from logisticRegression import plot_decision_regions


def main():
	iris = datasets.load_iris()
	X = iris.data[:, [2,3]]
	y = iris.target[:]
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=1, stratify=y)
	X_combined = np.vstack([X_train, X_test])
	y_combined = np.hstack([y_train, y_test])

	'''
	minkowski metric is a generalisation of the euclidean and manhattan distances
	it becomes euclidean when p = 2, and manhattan at p = 1

	regularisation is not applicable to DTCs and KNN we can use feature selection and dimentionality reduction to avoid overfitting(curse of dimensionality)

	'''
	knn = KNeighborsClassifier(n_neighbors = 5, p=2, metric = 'minkowski')
	knn.fit(X_train, y_train)
	plot_decision_regions(X_combined, y_combined, classifier = knn, test_idx = range(105, 150))
	plt.xlabel('Petal Length')
	plt.ylabel('Petal Width')
	plt.title('KNN classifier')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()
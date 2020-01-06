from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from logisticRegression import plot_decision_regions
import numpy as np


def main():
	iris = datasets.load_iris()
	X = iris.data[:, [2,3]]
	y = iris.target[:]
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1, stratify = y)
	X_combined = np.vstack([X_train, X_test])
	y_combined = np.hstack([y_train, y_test])
	# n_estimators is number of DCTs
	# n_jobs defines how many cores of our computer should be used simultaneously
	forest = RandomForestClassifier(criterion='gini', n_estimators=25,random_state = 1, n_jobs= 1)
	forest.fit(X_train, y_train)
	print('Score : %.3f' %forest.score(X_test, y_test))
	plot_decision_regions(X_combined, y_combined, forest, test_idx = range(105, 150))
	plt.xlabel('Petal Length')
	plt.ylabel('Petal Width')
	plt.title('RandomForestClassifier')
	plt.legend(loc='best')
	plt.show()


if __name__ == '__main__':
	main()
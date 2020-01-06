from logisticRegression import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



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


	lr = LogisticRegression(C = 100.0, random_state = 1, solver='lbfgs', multi_class='ovr')
	lr.fit(X_train_std, y_train)
	y_pred = lr.predict(X_test_std)
	print('Accuracy Score : %.3f' %lr.score(X_test_std, y_test))
	# plotting decision surface with scatterplot
	X_combined_std = np.vstack((X_train_std, X_test_std))
	y_combined = np.hstack((y_train, y_test))
	plot_decision_regions(X = X_combined_std, y = y_combined, classifier=lr, test_idx = range(105,150))
	plt.xlabel('Petal length[std]')
	plt.ylabel('Petal width[std]')
	plt.legend()
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()
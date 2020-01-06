from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from logisticRegression import plot_decision_regions
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def main():
	iris = datasets.load_iris()
	# sepal length, width petal length, width
	#  we will use petal length and width
	X = iris.data[:, [2,3]]
	y = iris.target[:]
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
	X_combined_std = np.vstack([X_train_std, X_test_std])
	y_combined_std = np.hstack([y_train, y_test])


	svm = SVC(kernel = 'linear', C = 1.0, random_state = 1)
	svm.fit(X_train_std, y_train)
	print('Score : %.3f' %svm.score(X_test_std, y_test))
	plot_decision_regions(X_combined_std, y_combined_std, classifier = svm, test_idx = range(105,150))
	plt.xlabel('Petal Length(std)')
	plt.ylabel('Petal Width(std)')
	plt.title('Support Vector Classifier')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
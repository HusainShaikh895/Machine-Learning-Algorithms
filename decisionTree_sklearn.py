from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from logisticRegression import plot_decision_regions
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz


def main():
	iris = datasets.load_iris()
	X = iris.data[:, [2,3]]
	y = iris.target[:]
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1, stratify = y)
	tree_model = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=3, random_state = 1)
	tree_model.fit(X_train, y_train)
	X_combined = np.vstack([X_train, X_test])
	y_combined = np.hstack([y_train, y_test])
	print('Score : %.3f' %tree_model.score(X_test, y_test))
	plot_decision_regions(X_combined, y_combined, classifier = tree_model,test_idx = range(105, 150))
	plt.xlabel('Petal Length')
	plt.ylabel('Petal Width')
	plt.legend()
	plt.title('DecisionTreeClassifier')
	plt.show()
	# tree.plot_tree(tree_model)
	# plt.show()
	dot_data = export_graphviz(tree_model, filled=True, rounded=True, class_names = ['setosa', 'Versicolor', 'Virginica'], feature_names = ['petal Length', 'Petal width'], out_file = None)
	graph = graph_from_dot_data(dot_data)
	graph.write_png('tree.png')



if __name__ == '__main__':
	main()
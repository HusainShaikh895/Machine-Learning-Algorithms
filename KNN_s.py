# Husain Shaikh
# Google ML YouTube
# KNN from scratch

from scipy.spatial import distance

def euclid(a, b):
	# finds distance between two points
	return distance.euclidean(a, b)

class KNN():

	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train
	
	def predict(self, x_test):
		predictions = []
		for row in x_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions

	def closest(self, row):
		best_distance = euclid(row, self.x_train[0])
		best_index = 0
		for i in range(1, len(self.x_train)):
			dist = euclid(row, x_train[i])
			if(dist<best_distance):
				best_index = i
				best_distance = dist
		return self.y_train[best_index]





from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
# Features
y = iris.target
# Output / Label

#sklearn.cross_validation is outdated
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.7)

# Divide x into (x_train, x_test) and similarly for y (y_train, y_test)


#note : it is neighbOrs and not neighbOUrs
from sklearn.neighbors import KNeighborsClassifier
classifier = KNN()

# Train : provide training data
classifier.fit(x_train, y_train)

# Predict : Provide never before seen test data
predictions = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

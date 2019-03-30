# Husain Shaikh
# Google ML YouTube
# using libraries and pre-defined functions


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
classifier = KNeighborsClassifier()

# Train : provide training data
classifier.fit(x_train, y_train)

# Predict : Provide never before seen test data
predictions = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

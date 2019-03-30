#Simple classifier for weight

from sklearn import tree


features=[[60,76,87,98],[50,45,65,76],[55,78,78,98],[61,34,76,23],[70,87,88,78],[90,76,87,76],[35,56,76,87],[62,66,67,87]]
label=[1,0,1,0,1,1,0,1]


pr=tree.DecisionTreeClassifier()
pr=pr.fit(features,label)
test=[[42,62,82,45]]
output=pr.predict(test)
print(output)

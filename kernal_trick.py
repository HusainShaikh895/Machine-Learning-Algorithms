'''
	Non-linear data which is transformed (not really) to higher dimension until it is linearly seperable
	is called kernal trick
'''


import numpy as np
import matplotlib.pyplot as plt
from logisticRegression import plot_decision_regions
from sklearn.svm import SVC

def main():
	# creating data of 200 examples of xor logic
	np.random.seed(1)
	X_xor = np.random.randn(200,2)
	y_xor = np.logical_xor(X_xor[:, 0]>0, X_xor[:,1]>0)
	y_xor = np.where(y_xor, 1, -1)
	# visualising how it looks
	plt.scatter(X_xor[y_xor == 1, 0],X_xor[y_xor == 1, 1],c='b', marker='x',label='1')
	plt.scatter(X_xor[y_xor == -1, 0],X_xor[y_xor == -1, 1],c='r', marker='s',label='-1')
	#  as it is clearly not linearly seperable we will need the kernel trick
	# C is the strictness of soft margin 
	#  gamma is the fitting parameter
	# higher gamma means overfit and vice-versa
	svm = SVC(kernel='rbf', C=10.0, gamma=0.5, random_state=1)
	svm.fit(X_xor, y_xor)
	plot_decision_regions(X_xor, y_xor, classifier = svm)
	plt.show()




if __name__ == '__main__':
	main()

#to remove warnings if any
import warnings
warnings.filterwarnings(action="ignore", module="sklearn", message="internal gelsd")

import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

model = LinearRegression()

x = [[i] for i in range(200)]
y = [[2*i] for i in range(200)]

print(len(x),"\n",len(y))

model.fit(x, y)

print(model.predict([[1500],[1000],[1800]])) 


plt.scatter(x, y, color="red", label="dataset")
plt.plot(x, model.predict(x), color="blue", label="Linear Regression Model")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression")
plt.show()

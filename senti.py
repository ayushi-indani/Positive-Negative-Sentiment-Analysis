import numpy as np
import pandas as pd

df = pd.read_csv('data2.csv')

X = np.array(df)
X = np.delete(X, 852, 1)

y = np.array(df)
y = y[:, -1]
y = y[:, np.newaxis]
m, n = X.shape

w = np.random.rand(1, n)
b = np.random.rand(1, 1)

print ("Gradient Descent Started")
while True:
	z = np.dot(X, w.T) + b
	a = (1/(1+np.exp(-z)))

	dz = np.subtract(a, y)
	dw = np.multiply(dz, X)
	dw = np.sum(dw, axis = 0, keepdims= True) / m
	alpha = 0.01
	up_w = w-alpha*dw
	s=np.less(abs(up_w-w),0.001)
	if np.all(s):
		print ("Cost is Minimum")
		break

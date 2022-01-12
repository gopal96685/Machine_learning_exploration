# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6.0, 4.5)

# Preprocessing Input data
data = pd.read_csv('data.csv')
X = data[['a']].values
Y = data[['b']].values 
plt.scatter(X, Y)
plt.show()

print(data.dtypes)
c = 0
m = 0
learning = .0001
iteration = 50

n = len(X)

for i in range(iteration):
	Y_pred = m*X + c
	d_m = (-2/n) * sum(X * (Y - Y_pred) )
	d_c = (-2/n) * sum( Y - Y_pred) 
	m = m - learning*d_m
	c = c - learning*d_c

Y_pred = m*X + c
plt.scatter(X, Y)
plt.scatter(X, Y_pred)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue') # predicted
plt.show()

































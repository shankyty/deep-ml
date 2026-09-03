import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X = np.array(X)
	y = np.array(y)
	X_T = X.T
	X_2_1 = np.linalg.inv(np.dot(X_T,X))
	
	return np.dot(np.dot(X_2_1,X_T),y)
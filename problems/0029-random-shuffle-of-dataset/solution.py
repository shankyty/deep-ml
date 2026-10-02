import numpy as np

def shuffle_data(X, y, seed=None):
	# Your code here
	if seed is not None:
		np.random.seed(seed)
	n = len(X)
	shuffled_X = X.copy()
	shuffled_y = y.copy()
	for i in range(n - 1, 0, -1):
		j = np.random.randint(0, i + 1)
		temp_X = shuffled_X[i].copy()
		shuffled_X[i] = shuffled_X[j]
		shuffled_X[j] = temp_X
		temp_Y = shuffled_y[i].copy()
		shuffled_y[i] = shuffled_y[j]
		shuffled_y[j] = temp_Y
	
	return shuffled_X, shuffled_y
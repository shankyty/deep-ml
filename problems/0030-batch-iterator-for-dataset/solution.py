import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	n_samples = X.shape[0]
	batches = []
	for i in range(0,n_samples, batch_size):
		X_batch = X[i:i + batch_size]
		if y is not None :
			y_batch = y[i:i + batch_size]
			batches.append([X_batch,y_batch])
		else :
			batches.append([X_batch])
	return batches
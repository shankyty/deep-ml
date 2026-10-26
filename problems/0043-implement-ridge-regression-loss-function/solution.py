import numpy as np

def ridge_loss(
	X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	l2 = alpha * np.dot(w,w)
	y_pred = np.dot(X, w)
	mse = np.mean((y_true - y_pred) ** 2)
	return mse + l2

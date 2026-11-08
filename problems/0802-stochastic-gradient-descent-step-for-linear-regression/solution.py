import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    # X shape is (n_samples, n_features)
    # y shape is (n_samples,)
    # weights shape is (n_features, )
    n_samples  = X.shape[0] 
    w = weights.astype(float).copy()
    for i in range(n_iter):
        idx = i % n_samples
        xi = X[idx]
        yi = y[idx]

        y_pred = xi.dot(w)
        gradient = 2 * (y_pred - yi) * xi
        w -= learning_rate * gradient
    return w.tolist()


import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    X_b = X[batch_indices]
    y_b = y[batch_indices]
    m = len(batch_indices)

    y_hat = np.dot(X_b, weights) + bias
    e = y_hat - y_b

    dw = (2.0 / m) * np.dot(X_b.T, e)
    db = (2.0 / m) * np.sum(e)

    new_weights = weights - lr * dw
    new_bias = bias - lr * db
    return np.append(new_weights, new_bias)

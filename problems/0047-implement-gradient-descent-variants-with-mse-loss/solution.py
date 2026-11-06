import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1,1)
    w = np.asarray(weights, dtype=float).reshape(-1,1)
    m,n = X.shape
    for epoch in range(n_epochs):
        if method == 'batch':
            predictions = X @ w
            errors = predictions - y
            grad = (2.0)/m * (X.T @ errors)
            w = w - learning_rate * grad

        elif method == 'stochastic':
            
            for idx in range(m):
                xi = X[idx: idx + 1, :]
                yi = y[idx: idx + 1, :]
                prediction = xi @ w
                errors = prediction - yi
                grad = 2.0 * (xi.T @ errors)
                w = w - learning_rate * grad
        elif method == 'mini_batch':
           
            for i in range(0,m,batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                predictions = X_batch @ w
                k = X_batch.shape[0]
                errors = predictions - y_batch
                grad = (2.0/k) * X_batch.T @ errors
                w = w - learning_rate * grad
    return w.flatten()

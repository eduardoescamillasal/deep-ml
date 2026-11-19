import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    # Your code here
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    w = np.asarray(initial_weights, dtype=float)
    b = float(initial_bias)
    n_samples = len(y)
    mse_values: list[float] = []
    for _ in range(epochs):
        z = x @ w + b
        y_hat = _sigmoid(z)

        mse = float(np.mean((y_hat - y)** 2))
        mse_values.append(round(mse, 4))

        dz = (2.0 / n_samples) * (y_hat - y) * y_hat * (1.0 - y_hat)
        dw = x.T @ dz
        db = float(np.sum(dz))

        w = w - learning_rate * dw
        b = b - learning_rate * db
    
    updated_weights = [round(float(val), 4) for val in w]
    updated_bias = round(float(b), 4)
    return updated_weights, updated_bias, mse_values
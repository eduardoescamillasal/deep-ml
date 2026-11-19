import numpy as np

def _sigmoid(z: float) -> float:
    z_clamped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.

    Args:
        a_prev: activations from previous layer, shape (K,)
        W: weight matrix, shape (J, K)
        b: bias vector, shape (J,)
        y: target vector, shape (J,)

    Returns:
        dict with keys 'dW' (J x K nested list), 'db' (list of length J),
        and 'da_prev' (list of length K).
    """
    # Your code here
    a_prev_arr = np.asarray(a_prev, dtype=float)
    w_arr = np.asarray(W, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    z = w_arr @ a_prev_arr + b_arr
    a = _sigmoid(z)
    dC_z = 2.0 * (a - y_arr) * a * (1 - a)
    dC_W = dC_z[:, None] * a_prev_arr[None, :]
    dC_b = dC_z
    dC_a_prev = w_arr.T @ dC_z

    return {
        "dW": dC_W.tolist(),
        "db": dC_b.tolist(),
        "da_prev": dC_a_prev.tolist()
    }

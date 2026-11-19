import numpy as np
def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Your code here
    sigmoid = float(np.exp(-x) / (1 + np.exp(-x))**2)
    tanh = float(1 - np.tanh(x)**2)
    relu = 1.0 if x > 0 else 0.0

    return {
        "sigmoid": sigmoid,
        "tanh": tanh,
        "relu": relu
    }
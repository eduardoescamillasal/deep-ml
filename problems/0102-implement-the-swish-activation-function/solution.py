import numpy as np
def swish(x: float) -> float:
    """
    Implements the Swish activation function.

    Args:
        x: Input value

    Returns:
        The Swish activation value
    """
    # Your code here
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    return x * sigmoid(x)
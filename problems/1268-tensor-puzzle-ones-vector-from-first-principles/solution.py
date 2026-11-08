import numpy as np

def ones(n: int) -> np.ndarray:
    """Return a length-n float vector of ones without calling np.ones."""
    # Your code here
    zeros = np.zeros(n)
    for i in range(n):
        zeros[i] = 1.
    return zeros.astype(float)

import numpy as np

def diag(A: np.ndarray) -> np.ndarray:
    """Return the main diagonal of square matrix A."""
    # Your code here
    m,n = A.shape
    i = np.arange(min(m,n))
    return A[i,i]

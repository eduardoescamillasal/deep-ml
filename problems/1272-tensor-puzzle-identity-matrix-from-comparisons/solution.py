import numpy as np

def eye(n: int) -> np.ndarray:
    """n x n identity matrix without np.eye."""
    # Your code here
    i = np.arange(n)
    mask = i[:, None] == i[None, :]
    return mask.astype(float)

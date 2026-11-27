import numpy as np

def row_normalize(counts: list[list[float]]) -> list[list[float]]:
    """Convert a count matrix into a row-stochastic probability matrix."""
    C = np.asarray(counts, dtype=float)
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = C / row_sums
    
    return probs.tolist()
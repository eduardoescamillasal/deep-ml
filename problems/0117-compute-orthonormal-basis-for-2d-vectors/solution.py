import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    if not vectors:
        return []
    basis: list[np.ndarray] = []
    for vec in vectors:
        v = np.asarray(vec, dtype=float)
        w = v

        for u in basis:
            w = w - np.dot(w, u) * u
        
        norm = np.linalg.norm(w)
        if norm > tol:
            basis.append(w / norm)

    
    
    return basis
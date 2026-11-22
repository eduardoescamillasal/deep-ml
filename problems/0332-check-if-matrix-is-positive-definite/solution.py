import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """

    matrix =np.asarray(matrix, dtype=float)
    is_symmetric = np.allclose(matrix.T, matrix)
    eigenvalues = np.linalg.eigvals(matrix)
    real_eigenvalues = np.real(eigenvalues)
    sorted_eigenvalues = [round(float(val), 4) for val in np.sort(real_eigenvalues)]

    is_positive_definite = bool(is_symmetric and np.all(real_eigenvalues > 1e-10))
    return {
        "is_positive_definite": is_positive_definite,
        "eigenvalues": sorted_eigenvalues
    }
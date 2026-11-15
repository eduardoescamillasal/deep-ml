from typing import Callable

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Hessian matrix of function f at the given point using finite differences.
    
    Args:
        f: A scalar function that takes a list of floats and returns a float
        point: The point at which to compute the Hessian (list of coordinates)
        h: Step size for finite differences (default: 1e-5)
        
    Returns:
        The Hessian matrix as a list of lists (n x n where n = len(point))
    """
    # Your code here
    n = len(point)
    H = [[0.0]*n for _ in range(n)]
    f_val = f(point)
    for i in range(n):
        for j in range(n):
            if i == j:
                forward_pnt = list(point)
                forward_pnt[i] += h
                backward_pnt = list(point)
                backward_pnt[i] -=h
                H[i][j] = (f(forward_pnt) - 2 * f_val + f(backward_pnt))/(h**2)
            else:
                p1 = list(point)
                p1[i] += h
                p1[j] += h

                p2 = list(point)
                p2[i] += h
                p2[j] -= h

                p3 = list(point)
                p3[i] -= h
                p3[j] += h

                p4 = list(point)
                p4[i] -= h
                p4[j] -= h

                H[i][j] = (f(p1) - f(p2) - f(p3) + f(p4))/(4*h**2)
    
    return H
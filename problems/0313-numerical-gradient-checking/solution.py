import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    x_arr = np.asarray(x, dtype=float)
    grad_analytical = np.asarray(analytical_grad, dtype=float)
    grad_num = np.zeros_like(x_arr, dtype=float)

    for idx in np.ndindex(x_arr.shape):
        x_plus = x_arr.copy()
        x_minus = x_arr.copy()
        x_plus[idx] += epsilon
        x_minus[idx] -= epsilon
        grad_num[idx] = (float(f(x_plus)) - float(f(x_minus))) / (2.0 * epsilon)
        

    # 2. Compute relative error using L2 norms
    # ||grad_num - grad_analytical||_2
    diff_norm = float(np.linalg.norm(grad_num - grad_analytical))
    #  ||grad_num||_2 + ||grad_analytical||_2
    denom = np.linalg.norm(grad_num) + np.linalg.norm(grad_analytical)
    
    if denom == 0.0:
        relative_error = 0.0
    else:
        relative_error = diff_norm / denom
    return grad_num, relative_error
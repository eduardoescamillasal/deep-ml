import numpy as np

def woodbury_solve(sigma_diag, X, B):

    sig_inv = 1.0 / np.asarray(sigma_diag)
    

    sig_inv_X = sig_inv[:, None] * X
    

    mid_term = np.eye(X.shape[1]) + X.T @ sig_inv_X
    mid_term_inv = np.linalg.inv(mid_term)
    

    sig_inv_B = sig_inv[:, None] * B if B.ndim > 1 else sig_inv * B
    
 
    return sig_inv_B - sig_inv_X @ (mid_term_inv @ (X.T @ sig_inv_B))




def sherman_morrison(A_inv, u, v):
    u = np.asarray(u).reshape(-1,1)
    v = np.asarray(v).reshape(-1,1)
    denominator = 1 + (v.T @ A_inv @ u).item()
    if(np.isclose(denominator, 0.0, atol=1e-12)):
        raise ValueError("Rank-1 update is singular")
    numerator = (A_inv @ u) @ (v.T @ A_inv)
    return A_inv - numerator / denominator

def logdet_rank_one_update(logdet_A, A_inv, u, v):
    u = np.asarray(u).reshape(-1,1)
    v = np.asarray(v).reshape(-1,1)
    factor = 1.0 + (v.T @ A_inv @ u).item()
    if factor <= 0.0:
        raise ValueError("Updated determinant is not positive.")
    return logdet_A + np.log(abs(factor))

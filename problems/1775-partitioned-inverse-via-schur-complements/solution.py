import numpy as np

def partitioned_inverse(M, k):
     E = M[:k, :k]
     F = M[:k, k:]
     G = M[k:, :k]
     H = M[k:, k:]

     H_inv = np.linalg.inv(H)
     M_H = E - F @ H_inv @ G
     M_H_inv = np.linalg.inv(M_H)

     top_left = M_H_inv
     top_right = -1.0 * M_H_inv @ F @ H_inv
     bottom_left = -1.0 * H_inv @ G @ M_H_inv
     bottom_right = H_inv + H_inv @ G @ M_H_inv @ F @ H_inv

     top_row = np.hstack([top_left, top_right])
     bottom_row = np.hstack([bottom_left, bottom_right])
     return np.vstack([top_row, bottom_row])

def block_slogdet(M, k):
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]

    H_inv = np.linalg.inv(H)
    M_H = E - F @ H_inv @ G
    sign_M_H, log_M_H = np.linalg.slogdet(M_H)
    sign_H, log_H = np.linalg.slogdet(H)

    return (sign_M_H * sign_H, log_M_H + log_H)

def bordered_inverse(A_inv, b, c):
    b = np.asarray(b).reshape(-1,1)

    v= A_inv @ b 
    s = c - (b.T @ v).item()
    if abs(s) < 1e-12:
        raise ValueError("")

    top_left = A_inv + (1.0/s) * (v @ v.T) 
    top_right = -(1.0/s) * v
    bottom_left = -(1.0/s) * v.T
    bottom_right = np.array([[1.0 / s]])

    top_row = np.hstack([top_left, top_right])
    bottom_row = np.hstack([bottom_left, bottom_right])
    return np.vstack([top_row, bottom_row])


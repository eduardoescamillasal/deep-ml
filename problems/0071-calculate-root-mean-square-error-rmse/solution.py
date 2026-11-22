
import numpy as np

def rmse(y_true, y_pred):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if y_pred.shape != y_true.shape or y_pred.size == 0 or y_true.size == 0:
        raise ValueError("Shapes differ")
    rmse_res  = np.sqrt(np.mean((y_pred - y_true)**2))
    return round(rmse_res,3)

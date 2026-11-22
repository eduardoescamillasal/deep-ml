
import numpy as np

def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_mean = np.mean(y_true)

    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_mean)**2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res/ss_tot

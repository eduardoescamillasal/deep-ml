import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
    scores = np.asarray(scores)
    scores = scores - np.max(scores) 
    exp_vals = np.exp(scores)
    return (exp_vals / np.sum(exp_vals)).tolist()
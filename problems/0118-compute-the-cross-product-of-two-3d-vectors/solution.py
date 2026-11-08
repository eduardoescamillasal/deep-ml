import numpy as np

def cross_product(a, b):
    # Your code here
    a = np.asarray(a)
    b = np.asarray(b)

    return a[[1,2,0]] * b[[2,0,1]] - b[[1,2,0]] * a[[2,0,1]]
import numpy as np
def rotation_layer(X, angle):
    X = np.asarray(X)
    R =[
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ]
    R = np.asarray(R)
    return X @ R.T
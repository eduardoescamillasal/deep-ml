import numpy as np

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def digit_network_forward(image, weights, biases):
    """Run an image through a fully connected sigmoid network.

    Returns (output activations rounded to 4 decimals, predicted digit).
    """
    a = np.asarray(image, dtype=float).ravel()

    for w, b in zip(weights, biases):
        w_arr = np.asarray(w, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        z = w_arr @ a + b_arr
        a = _sigmoid(z)

    digit = int(np.argmax(a))
    rounded_activations = [round(float(val), 4) for val in a]
    return (rounded_activations, digit)
    

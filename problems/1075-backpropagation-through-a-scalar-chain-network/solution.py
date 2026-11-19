import numpy as np

def _sigmoid(x: float) -> float:
    x_clamped = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + np.exp(-x))

def backprop_chain(a0: float, y: float, w: list, b: list) -> dict:
    """
    Compute gradients of C = (a_L - y)^2 with respect to each weight and bias
    in a chain network where every layer has a single sigmoid neuron.

    Args:
        a0: input activation (scalar)
        y: target value (scalar)
        w: list of L weights
        b: list of L biases

    Returns:
        Dictionary with keys 'dW' and 'dB', each a list of L floats rounded to 6 places.
    """
    # forward pass
    num_layers = len(w)
    activations = [float(a0)]
    for idx in range(num_layers):
        z = float(w[idx]) * activations[-1] + float(b[idx])
        activations.append(_sigmoid(z))
    
    # backward pass
    d_a = 2.0 * (activations[-1] - float(y))
    dw_list = [0.0] * num_layers
    db_list = [0.0] * num_layers

    for idx in range(num_layers - 1, -1, -1):
        a_prev = activations[idx]
        a_curr = activations[idx + 1]
        w_curr = float(w[idx])

        d_z = d_a * a_curr * (1 - a_curr)
        dw_list[idx] = d_z * a_prev
        db_list[idx] = d_z 

        d_a = d_z * w_curr

    return {
        'dW': [round(val, 6) for val in dw_list],
        'dB': [round(val, 6) for val in db_list]
    }

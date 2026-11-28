import numpy as np
def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute the Huber Loss between true and predicted values.

    Args:
        y_true (float | list[float]): Ground truth values
        y_pred (float | list[float]): Predicted values
        delta (float): Transition threshold between MSE and MAE behavior

    Returns:
        float: Average Huber loss
    """
    # Your code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    deviation = np.abs(y_true - y_pred)
    result = np.where(
        deviation <= delta,
        0.5 * deviation**2,
        delta * (deviation - 0.5 * delta)
        )
    
    return float(np.mean(result))
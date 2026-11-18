import numpy as np

def forward_kinematics(l1, l2, theta1, theta2):
    # Return (position, jacobian)
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)

    x2 = l2 * np.cos(theta1 + theta2)
    y2 = l2 * np.sin(theta1 + theta2)

    jacobian = np.array([
        [-y1 - y2, -y2],
        [x1 + x2, x2]
    ])
    
    position = np.array(
        [x1 + x2, y1 + y2]
    )

    return position, jacobian
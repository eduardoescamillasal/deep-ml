import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	grad_arr = np.array(gradient)
	mag = np.linalg.norm(grad_arr)
	if mag < 1e-9: 
		return dict(magnitude = mag, direction = [0,0], descent_directinon=[0,0])
	direction = 1.0 / mag * grad_arr

	return dict(magnitude = mag, direction = direction, descent_direction = -1*direction)
import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	# Your code here
	current_val = float(x)
	deriv_accum = 1.0
	for func in reversed(functions):
		if func == 'square':
			local_deriv = 2 * current_val
			current_val = current_val ** 2
		elif func == 'sin':
			local_deriv = np.cos(current_val)
			current_val = np.sin(current_val)
		elif func == 'exp':
			local_deriv = np.exp(current_val)
			current_val = np.exp(current_val)
		elif func == 'log':
			local_deriv = 1.0 / current_val
			current_val = np.log(current_val)
		else:
			raise ValueError(f"Unknown function label: {func}")
		
		deriv_accum *= local_deriv

	return float(deriv_accum)
from math import sqrt
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	[[a,b],[c,d]] = matrix
	trace = a + d
	det = a*d - b*c
	sol1 = trace /2 + sqrt(trace ** 2 - 4 * det) / 2
	sol2 = trace / 2 - sqrt(trace ** 2 - 4 * det) /2
	return [sol1, sol2]
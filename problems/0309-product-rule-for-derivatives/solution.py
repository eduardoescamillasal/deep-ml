import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    # Your code here
    def poly_add(a_coeffs, b_coeffs):
        a_coeffs, b_coeffs = normalize_size(a_coeffs, b_coeffs)
        return [a + b for a, b in zip(a_coeffs, b_coeffs)]

    def normalize_size(a_coeffs, b_coeffs):
        if len(a_coeffs) > len(b_coeffs):
            b_coeffs = b_coeffs + [0] *(len(a_coeffs) - len(b_coeffs))
        else:
            a_coeffs = a_coeffs + [0] *(len(b_coeffs) - len(a_coeffs))
        return a_coeffs, b_coeffs

    def poly_derivative(coeffs: list) -> list:
        a = [i*coeff for i, coeff in enumerate(coeffs)]
        return a[1:]

    def poly_mult(a_coeffs: list, b_coeffs: list) -> list:
        res = [0] * (len(a_coeffs) + len(b_coeffs) - 1)
        for i, a in enumerate(a_coeffs):
            for j, b in enumerate(b_coeffs):
                res[i + j] += a * b
        return res
    
    f_prime = poly_derivative(f_coeffs)
    g_prime = poly_derivative(g_coeffs)
    g_f_prime = poly_mult(g_coeffs, f_prime)
    f_g_prime = poly_mult(f_coeffs, g_prime)
    result = poly_add(f_g_prime, g_f_prime)
    return [0.0] if result == [] else result
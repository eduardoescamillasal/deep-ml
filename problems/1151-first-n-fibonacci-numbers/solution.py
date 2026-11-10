def first_n_fibonacci(n):
    # Return a list of the first n Fibonacci numbers
    if n == 0:
        return []
    if n == 1:
        return [0,1]
    a,b = 0,1
    res = [0,1]
    for _ in range(n - 2):
        a,b = b, a+b
        res.append(b)
    return res
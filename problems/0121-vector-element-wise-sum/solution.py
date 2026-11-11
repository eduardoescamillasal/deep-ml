import numpy as np
def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
    # Return the element-wise sum of vectors 'a' and 'b'.
    # If vectors have different lengths, return -1.
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    if(arr_a.size != arr_b.size):
        return -1
    return (arr_a + arr_b).tolist()

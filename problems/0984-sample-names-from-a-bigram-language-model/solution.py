import numpy as np

def sample_name(P, itos, seed, max_len=100):
    # Your code here
    P = np.asarray(P, dtype=np.float64)
    V = P.shape[0]
    rng = np.random.default_rng(seed)

    out = []
    ix = 0
    for _ in range(max_len):
        row = P[ix]
        total = row.sum()

        if total == 0:
            probs = np.zeros(V, dtype=np.float64)
            probs[0] = 1.0
        else:
            probs = row / total
        
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0

        r = rng.random()
        ix = int(np.searchsorted(cdf, r, side='left'))
        ix = min(max(ix, 0), V - 1)

        if ix == 0:
            break
        
        out.append(itos[ix])
    return ''.join(out)

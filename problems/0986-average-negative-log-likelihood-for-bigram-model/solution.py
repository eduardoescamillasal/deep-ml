import numpy as np

def bigram_avg_nll(P, words, stoi):
    """Average negative log-likelihood of words under a bigram model."""
    log_likelihood = 0.0
    n = 0
    if len(words) == 0:
        return 0.0
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = np.log(prob)
            log_likelihood += logprob
            n += 1
    return -log_likelihood / n 

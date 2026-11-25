import numpy as np

def bigram_counts(words):
    chars = sorted(set("".join(words)))
    stoi = {".":0}
    for i,ch in enumerate(chars, start=1):
        stoi[ch] = i 
    
    V = len(stoi)
    N = np.zeros((V, V), dtype=int)

    for w in words:
        seq = ["."] + list(w) + ["."]
        for a, b in zip(seq, seq[1:]):
            N[stoi[a], stoi[b]] += 1
    return N.tolist()
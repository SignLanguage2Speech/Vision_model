import numpy as np

def edit_distance(r, h):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    
    Implementation of this algorithm:
    https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_full_matrix

    Attributes:
        r -> reference sequence.
        h -> hypothesis sequence.

    Builds a (|r|+1) x (|h|+1) matrix M, where M_{i,j} is the edit distance between r and h up until this point.
    M_{0,0} is initialized at 0.
    """
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape(
        (len(r) + 1, len(h) + 1)
    )
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1]
                insert = d[i][j - 1]
                delete = d[i - 1][j]
                d[i][j] = min(substitute, insert, delete)
    return d

def edit_distance_scalar(r,h):
    return edit_distance(r,h)[-1,-1]
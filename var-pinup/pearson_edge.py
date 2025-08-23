import numpy as np
from itertools import combinations

def zscore_cols(X, eps=1e-8):
    m = X.mean(0, keepdims=True); s = X.std(0, keepdims=True)+eps
    return (X-m)/s

def pearson_edges_unwrapped(x, eps=1e-8):
    """
    Compute UNWRAPPED, moment-to-moment Pearson edge time series
    for ALL unordered node pairs in an N-node time series.

    Edges E_ij(t) = z_i(t) * z_j(t), after global per-node z-score.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must have shape (T, N)")

    T, N = x.shape

    z = zscore_cols(x, eps)

    pairs = list(combinations(range(N), 2))
    edges = np.empty((T, len(pairs)), float)
    for k, (i, j) in enumerate(pairs):
        edges[:, k] = z[:, i] * z[:, j]
    return edges, pairs
import numpy as np

def movavg_cols(X, W: int):
    """Centered moving average (same length) applied column-wise."""
    if W <= 1: 
        return X.copy()
    c = np.ones(W, dtype=float) / W
    Y = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        Y[:, j] = np.convolve(X[:, j], c, mode="same")
    return Y


def r2_from_single(y, z):
    """R^2 of linear reconstruction of z from single predictor y (with intercept)."""
    y = (y - y.mean()) / (y.std() + 1e-8)
    zc = z - z.mean()
    r = np.corrcoef(y, zc)[0, 1]
    return r * r


def r2_from_multi(Y, z):
    """R^2 of linear reconstruction of z from multiple predictors Y (with intercept)."""
    zc = z - z.mean()
    X = np.column_stack([np.ones(len(zc)), (Y - Y.mean(0)) / (Y.std(0) + 1e-8)])
    beta, *_ = np.linalg.lstsq(X, zc, rcond=None)
    zhat = X @ beta
    sse = np.sum((zc - zhat) ** 2)
    sst = np.sum(zc ** 2)
    return 1.0 - sse / sst, zhat
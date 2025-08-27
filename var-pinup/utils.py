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




### visualisation 
# --- helper: place a legend that adapts to N lines
def add_smart_legend(ax, max_rows_right=12, min_font=7):
    """
    If there are <= max_rows_right items: put legend on the right (single column).
    Otherwise: put a multi-column legend at the bottom.
    Returns the legend artist.
    """
    import math
    fig = ax.figure
    handles, labels = ax.get_legend_handles_labels()
    n = len(labels)

    if n <= max_rows_right:
        # right-side, single column
        fs = max(min_font, 10 - 0.2 * max(0, n - 6))
        leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0., frameon=False, fontsize=fs)
        # make room on the right
        fig.subplots_adjust(right=0.78)
    else:
        # bottom legend with multiple columns
        cols = min(n, math.ceil(n / max_rows_right))
        fs = max(min_font, 9 - 0.1 * max(0, cols - 4))
        leg = fig.legend(handles, labels, loc='lower center',
                         ncol=cols, frameon=False, fontsize=fs)
        # make room at the bottom
        fig.subplots_adjust(bottom=0.22)
    return leg

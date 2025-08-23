import numpy as np
import matplotlib.pyplot as plt

def stabilise_A(A: np.ndarray, max_radius: float = 0.98) -> np.ndarray:
    """Rescale A so its spectral radius <= max_radius (keeps VAR(1) stable)."""
    eigvals = np.linalg.eigvals(A)
    radius = np.max(np.abs(eigvals))
    if radius > max_radius and radius > 0:
        A = (max_radius / radius) * A
    return A

def tvp_var1(
    A0: np.ndarray, #initial transition matrix
    z: np.ndarray, #underlying TVP eg. z = 0.6*np.sin(2*np.pi*np.arange(T)/2000)
    gains: dict, #which entries in A contain the TVP (and dial up knob)
    Sigma: np.ndarray | float = 0.05, #noise
    x0: np.ndarray | None = None, #IC
    seed: int | None = 123, 
    stabilise_each_step: bool = True, #after every transition matrix update, restabilise the matrix
    phi = lambda u: u,
    visualize: bool = True,
):
    """
    General TVP-VAR(1) simulator with plotting.

    x_t = A(t) x_{t-1} + eps_t,   eps_t ~ N(0, Sigma)

    Parameters
    ----------
    A0 : (N, N) array
        Initial transition matrix (before TVP modulation).
    z : (T,) array
        Time-varying parameter (TVP). Length defines T.
    gains : dict[(i, j) -> float | callable]
        Which A entries are modulated and how:
          - float g:      A_ij(t) = A0_ij + g * phi(z[t])
          - callable f:   A_ij(t) = f(A0_ij, phi(z[t]), t)  # returns new scalar
    Sigma : (N, N) array or float
        Noise covariance (if float, uses Sigma * I).
    x0 : (N,) array or None
        Initial condition (random small vector if None).
    seed : int or None
        RNG seed.
    stabilise_each_step : bool
        If True, rescale A(t) each step to keep spectral radius <= 0.98.
    phi : callable
        Nonlinearity on z before modulation (default identity).
    visualize : bool
        If True, plots z(t), all modulated coefficients, and all node series.

    Returns
    -------
    x : (T, N) array
    z : (T,) array
    coef_paths : dict[(i, j) -> (T,) array]
    A_t : (T, N, N) array
    figs : list[matplotlib.figure.Figure]  # empty if visualize=False
    """
    rng = np.random.default_rng(seed)
    A0 = np.asarray(A0, dtype=float)
    N = A0.shape[0]
    assert A0.shape == (N, N), "A0 must be square."
    z = np.asarray(z, dtype=float).reshape(-1)
    T = len(z)

    # Noise covariance
    if np.isscalar(Sigma):
        Sigma = float(Sigma) * np.eye(N)
    else:
        Sigma = np.asarray(Sigma, dtype=float)
    L = np.linalg.cholesky(Sigma)

    # Allocate
    x = np.zeros((T, N))
    A_t = np.zeros((T, N, N))
    coef_paths = {k: np.zeros(T) for k in gains.keys()}

    # Initial condition
    if x0 is None:
        x[0] = 0.1 * rng.standard_normal(N)
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (N,), "x0 must be shape (N,)"
        x[0] = x0

    # t = 0
    A = A0.copy()
    z0 = phi(z[0])
    for (i, j), rule in gains.items():
        base = A0[i, j]
        A[i, j] = rule(base, z0, 0) if callable(rule) else base + float(rule) * z0
        coef_paths[(i, j)][0] = A[i, j]
    if stabilise_each_step:
        A = stabilise_A(A)
    A_t[0] = A

    # Iterate
    for t in range(1, T):
        A = A0.copy()
        zt = phi(z[t])

        for (i, j), rule in gains.items():
            base = A0[i, j]
            A[i, j] = rule(base, zt, t) if callable(rule) else base + float(rule) * zt
            coef_paths[(i, j)][t] = A[i, j]

        if stabilise_each_step:
            A = stabilise_A(A)

        A_t[t] = A
        eps = L @ rng.standard_normal(N)
        x[t] = A @ x[t - 1] + eps

    figs = []
    if visualize:
        # 1) TVP
        fig1 = plt.figure()
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(T), z)
        ttl = "Latent TVP z(t)" #if suptitle is None else f"{suptitle} — TVP"
        plt.title(ttl)
        plt.xlabel("time"); plt.ylabel("z(t)")
        figs.append(fig1)

        # # 2) Modulated coefficients
        # if len(coef_paths) > 0:
        #     fig2 = plt.figure()
        #     for (i, j), path in coef_paths.items():
        #         plt.plot(np.arange(T), path, label=f"A[{i},{j}](t)")
        #     ttl = "Modulated coefficients" if suptitle is None else f"{suptitle} — A_ij(t)"
        #     plt.title(ttl)
        #     plt.xlabel("time"); plt.ylabel("value")
        #     plt.legend()
        #     figs.append(fig2)

        # 3) Node time series
        fig3 = plt.figure()
        plt.figure(figsize=(10, 4))
        for n in range(N):
            plt.plot(np.arange(T), x[:, n], label=f"node {n}")
        ttl = "Node time series"
        plt.title(ttl)
        plt.xlabel("time"); plt.ylabel("amplitude")
        plt.legend()
        figs.append(fig3)

        plt.show()

    return x, z, coef_paths, A_t, figs
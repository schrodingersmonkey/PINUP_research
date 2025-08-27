from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, List
import numpy as np

@dataclass
class ExperimentConfig:
    exp_name: str = "5 nodes"

    #first, all the VAR sim parameters. 
    N: int = 5
    T: int = 5000
    Sigma: float = 0.01
    seed: int = 123
 
    #transition matrix
    background_mode: str = "uniform"    #options are zeros , constant and uniform
    self_memory: float = 0.35         #diagonal entries
    background_coupling: float = 0.0
    bg_mu: float = 0.05               # mean of uniform off-diagonals
    bg_width: float = 0.03            # half-width; samples in [mu - width, mu + width]
    force_symmetric: bool = True
    max_radius_offdiag: float | None = 0.98  # if set, rescale OFF-DIAGONALS to keep rho ≤ this

    #tvp
    tvp_amp: float = 0.6
    tvp_period: int = 2000
    tvp_phi: Callable[[float], float] = staticmethod(lambda u: u)  # nonlinearity on z before modulation

    # --- Which entries are modulated by the TVP ---
    # symmetric link by default (i,j) and (j,i)
    sym_links: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 1), (1, 2),]) #default val is only one link
    gain: float = 0.9
    directed: bool = False  
    #alternative
    link_gains: Dict[Tuple[int, int], float] = field(default_factory=dict) #overrides global gain 

    # --- Edge processing / SFA ---
    smoothing_W: int = 200
    n_sfa_components: int = 2 


# helpers
def _spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))

def _rescale_offdiagonals_to_radius(A: np.ndarray, target_rho: float, diag_val: float) -> np.ndarray:
    """Rescale only off-diagonal entries to keep spectral radius ≤ target_rho; restore diag to diag_val."""
    if target_rho is None:
        return A
    A = A.copy()
    # temporarily zero diag to isolate off-diagonal contribution
    d = np.diag(A).copy()
    np.fill_diagonal(A, 0.0)
    rho = _spectral_radius(A + np.diag([diag_val]*A.shape[0]))
    if rho > target_rho and rho > 0:
        scale = target_rho / rho
        A *= scale
    # restore diagonal
    np.fill_diagonal(A, diag_val)
    return A

def build_A0(cfg: ExperimentConfig) -> np.ndarray:
    """Construct base A0 with chosen background. Symmetric, diagonal = self_memory."""
    rng = np.random.default_rng(cfg.seed)
    N = cfg.N
    A0 = np.zeros((N, N), dtype=float)

    # set diagonal (self-memory) first
    np.fill_diagonal(A0, cfg.self_memory)

    if cfg.background_mode == "zeros":
        # keep off-diagonals at 0
        pass

    elif cfg.background_mode == "constant":
        # fill all off-diagonals with a constant
        for i in range(N):
            for j in range(N):
                if i != j:
                    A0[i, j] = cfg.background_coupling
        if cfg.force_symmetric:
            A0 = (A0 + A0.T) / 2.0
        np.fill_diagonal(A0, cfg.self_memory)

    elif cfg.background_mode == "uniform":
        # sample upper-tri off-diagonals ~ U[mu - width, mu + width], mirror to lower
        low = cfg.bg_mu - cfg.bg_width
        high = cfg.bg_mu + cfg.bg_width
        iu = np.triu_indices(N, k=1)
        A0[iu] = rng.uniform(low, high, size=len(iu[0]))
        if cfg.force_symmetric:
            A0 = A0 + A0.T
            np.fill_diagonal(A0, cfg.self_memory)
        else:
            # if not forcing symmetry, also sample lower triangle
            il = np.tril_indices(N, k=1)
            A0[il] = rng.uniform(low, high, size=len(il[0]))
            np.fill_diagonal(A0, cfg.self_memory)

    else:
        raise ValueError(f"Unknown background_mode: {cfg.background_mode}")

    # optional: keep spectral radius in check by scaling ONLY off-diagonals
    if cfg.max_radius_offdiag is not None:
        A0 = _rescale_offdiagonals_to_radius(A0, cfg.max_radius_offdiag, cfg.self_memory)

    return A0


def build_gains(cfg: ExperimentConfig) -> Dict[Tuple[int, int], float]:
    """
    Construct dictionary of modulated entries.
    Priority:
    - If cfg.link_gains is non-empty, use it directly.
    - Otherwise, use cfg.sym_links with global cfg.gain.
    """
    gains: Dict[Tuple[int, int], float] = {}
    if cfg.link_gains:
        for (i, j), g in cfg.link_gains.items():
            gains[(i, j)] = g
            if not cfg.directed:
                gains[(j, i)] = g
    else:
        for (i, j) in cfg.sym_links:
            gains[(i, j)] = cfg.gain
            if not cfg.directed:
                gains[(j, i)] = cfg.gain
    return gains


def make_tvp(cfg: ExperimentConfig) -> np.ndarray:
    """Generate TVP trajectory given config."""
    t: np.ndarray = np.arange(cfg.T)
    z: np.ndarray = cfg.tvp_amp * np.sin(2 * np.pi * t / cfg.tvp_period)
    return z
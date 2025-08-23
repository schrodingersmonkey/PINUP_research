from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, List
import numpy as np

@dataclass
class ExperimentConfig:
    #first, all the VAR sim parameters. 
    N: int = 3
    T: int = 5000
    Sigma: float = 0.01
    seed: int = 123

    #transition matrix
    self_memory: float = 0.35 #ie, the strength of the diagonal entries. 
    background_coupling: float = 0.0 

    #tvp
    tvp_amp: float = 0.6
    tvp_period: int = 2000
    tvp_phi: Callable[[float], float] = staticmethod(lambda u: u)  # nonlinearity on z before modulation

    # --- Which entries are modulated by the TVP ---
    # symmetric link by default (i,j) and (j,i)
    sym_link: Tuple[int, int] = (0, 1)
    gain: float = 0.9
    directed: bool = False  # if True, only (i,j) is modulated

    # --- Edge processing / SFA ---
    smoothing_W: int = 50
    n_sfa_components: int = 2


def make_tvp(cfg: ExperimentConfig) -> np.ndarray:
    t = np.arange(cfg.T)
    z = cfg.tvp_amp * np.sin(2 * np.pi * t / cfg.tvp_period)
    return z

def build_A0(cfg: ExperimentConfig) -> np.ndarray:
    A0 = np.zeros((cfg.N, cfg.N), dtype=float)
    np.fill_diagonal(A0, cfg.self_memory)
    if cfg.background_coupling != 0.0:
        for i in range(cfg.N):
            for j in range(cfg.N):
                if i != j:
                    A0[i, j] = cfg.background_coupling
    return A0

def build_gains(cfg: ExperimentConfig) -> Dict[Tuple[int,int], float]:
    i, j = cfg.sym_link
    g = {(i, j): cfg.gain}
    if not cfg.directed:
        g[(j, i)] = cfg.gain
    return g


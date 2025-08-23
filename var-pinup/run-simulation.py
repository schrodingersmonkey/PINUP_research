from pearson_edge import pearson_edges_unwrapped, zscore_cols
from sfa import run_sfa_on_edges
from var_simulate import tvp_var1
from utils import movavg_cols, r2_from_single, r2_from_multi
from config import ExperimentConfig, make_tvp, build_A0, build_gains

import matplotlib.pyplot as plt
import numpy as np

def run_once(cfg: ExperimentConfig, visualise: bool = True):
    # Build pieces from config
    A0 = build_A0(cfg)
    z  = make_tvp(cfg)
    gains = build_gains(cfg)

    # Simulate
    x, z, coef_paths, A_t, _ = tvp_var1(
        A0=A0, z=z, gains=gains, Sigma=cfg.Sigma, seed=cfg.seed,
        stabilise_each_step=True, phi=cfg.tvp_phi, visualize=False
    )

    # Nodes -> edges
    E_raw, pairs = pearson_edges_unwrapped(x)

    # Smooth edges and re-standardize
    E_sm = movavg_cols(E_raw, cfg.smoothing_W)
    E_sm = zscore_cols(E_sm)

    # Smooth TVP for fair comparison/plots
    z_sm = np.convolve(z, np.ones(cfg.smoothing_W)/cfg.smoothing_W, mode="same")

    # SFA on smoothed edges
    Y, deltas, _ = run_sfa_on_edges(E_sm, n_components=cfg.n_sfa_components)
    y1 = Y[:, 0]

    # Metrics
    y1z = (y1 - y1.mean()) / (y1.std() + 1e-8)
    corr1 = np.corrcoef(y1z, z_sm - z_sm.mean())[0, 1]
    R2_1 = r2_from_single(y1, z_sm)
    k = min(3, Y.shape[1])
    R2_k, zhat_k = r2_from_multi(Y[:, :k], z_sm)

    summary = {
        "pairs": pairs,
        "corr_sfa1_vs_zsm": float(corr1),
        "R2_single": float(R2_1),
        "R2_topk": float(R2_k),
        "topk": k,
        "deltas": deltas.tolist(),
        "config": cfg.__dict__,
    }

    if visualise:
        # (A) edges (smoothed) + smoothed TVP
        plt.figure(figsize=(10, 3.2))
        for idx, p in enumerate(pairs):
            plt.plot(E_sm[:, idx], lw=0.8, label=f"edge {p}")
        plt.plot(z_sm, lw=1.5, label="z(t) (smoothed)")
        plt.title("Smoothed instantaneous edges vs smoothed TVP")
        plt.xlabel("time"); plt.legend(ncol=len(pairs)); plt.tight_layout(); plt.show()

        # (B) SFA #1 vs TVP (both z-scored)
        zz = (z_sm - z_sm.mean()) / (z_sm.std() + 1e-8)
        plt.figure(figsize=(10, 3.2))
        plt.plot(y1z, label="SFA #1 (on smoothed edges)")
        plt.plot(zz, lw=1.5, label="z(t) (smoothed, z-scored)")
        plt.title(f"Recovered driver vs TVP  |  R²(single)={R2_1:.2f},  R²(top-{k})={R2_k:.2f}")
        plt.xlabel("time"); plt.legend(); plt.tight_layout(); plt.show()

        # (C) Reconstructed TVP from SFA #1
        r = np.corrcoef(y1z, z_sm - z_sm.mean())[0, 1]
        zhat_1 = r * y1z + z_sm.mean()
        plt.figure(figsize=(10, 3.2))
        plt.plot(z_sm, label="z(t) (smoothed)")
        plt.plot(zhat_1, label=f"recon from SFA1 (R²={R2_1:.2f})")
        plt.title("TVP reconstruction from SFA #1")
        plt.xlabel("time"); plt.legend(); plt.tight_layout(); plt.show()

    return summary

if __name__ == "__main__":
    cfg = ExperimentConfig()
    summary = run_once(cfg, visualise=True)
    print("/nSummary,", summary)
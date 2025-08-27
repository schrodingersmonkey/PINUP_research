# run_example.py (add imports)
import os, json, time, uuid
from dataclasses import asdict
import numpy as np
import matplotlib.pyplot as plt
from pearson_edge import pearson_edges_unwrapped, zscore_cols
from sfa import run_sfa_on_edges
from var_simulate import tvp_var1
from utils import movavg_cols, r2_from_single, r2_from_multi, add_smart_legend
from config import ExperimentConfig, make_tvp, build_A0, build_gains
import argparse



def _mk_run_dir(out_root: str, run_name: str | None = None) -> str:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = run_name or f"run_{ts}_{uuid.uuid4().hex[:6]}"
    out_dir = os.path.join(out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _to_jsonable(obj):
    # make config/metrics JSON-serializable
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if callable(obj):
        return getattr(obj, "__name__", str(obj))
    return obj

def _dump_json(path, data: dict):
    data = {k: _to_jsonable(v) for k, v in data.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def run_once(
    cfg: ExperimentConfig,
    visualize: bool = True,
    save_assets: bool = True,
    out_root: str = "./outputs",
    run_name: str | None = None,
):
    # --- build from config ---
    A0 = build_A0(cfg)
    z  = make_tvp(cfg)
    gains = build_gains(cfg)

    # --- simulate ---
    x, z, coef_paths, A_t, _ = tvp_var1(
        A0=A0, z=z, gains=gains, Sigma=cfg.Sigma, seed=cfg.seed,
        stabilise_each_step=True, phi=cfg.tvp_phi, visualize=False
    )

    # --- nodes -> edges, smoothing, SFA ---
    E_raw, pairs = pearson_edges_unwrapped(x)
    E_sm = movavg_cols(E_raw, cfg.smoothing_W)
    E_sm = zscore_cols(E_sm)
    z_sm = np.convolve(z, np.ones(cfg.smoothing_W)/cfg.smoothing_W, mode="same") 
    #we need to smooth here too so both the tvp and edges have the same temporal resolution

    Y, deltas, _ = run_sfa_on_edges(E_sm, n_components=cfg.n_sfa_components)
    y1 = Y[:, 0]

    # --- metrics ---
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
        "deltas": deltas,
    }

    edge_corrs = [np.corrcoef(y1z, E_sm[:, k])[0,1] for k in range(E_sm.shape[1])]
    print("corr between sfa 1 and edge pairs",list(zip(pairs, edge_corrs)))
# You'll likely see ~±1 for (0,1) and smaller for the others.

# 2) correlation of each edge with the (smoothed) TVP
    edge_vs_tvp = [np.corrcoef(E_sm[:, k], z_sm - z_sm.mean())[0,1] for k in range(E_sm.shape[1])]
    print("edge vs z_sm:", list(zip(pairs, edge_vs_tvp)))

    cfg_dict = asdict(cfg)              # << use asdict, not __dict__
    cfg_dict["tvp_phi"] = _to_jsonable(cfg.tvp_phi)
    # --- output directory & saving ---
    out_dir = None
    if save_assets:
        out_dir = _mk_run_dir(out_root, run_name)
        # save config and metrics
        
        _dump_json(os.path.join(out_dir, "config.json"), cfg_dict)
        _dump_json(os.path.join(out_dir, "metrics.json"), summary)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    for idx, p in enumerate(pairs):
        ax.plot(E_sm[:, idx], lw=0.8, label=f"edge {p}")
    ax.plot(z_sm, lw=1.5, label="z(t) (smoothed)")
    ax.set_title("Smoothed instantaneous edges vs smoothed TVP")
    ax.set_xlabel("time")
    add_smart_legend(ax)  # <-- adaptive legend
    if save_assets: _savefig(os.path.join(out_dir, "edges_vs_tvp.png"))
    if visualize: plt.show()

    # (B) SFA #1 vs z_sm (both z-scored)
    zz = (z_sm - z_sm.mean()) / (z_sm.std() + 1e-8)
    plt.figure(figsize=(10, 3.2))
    plt.plot(y1z, label="SFA #1 (on smoothed edges)")
    plt.plot(zz, lw=1.5, label="z(t) (smoothed, z-scored)")
    plt.title(f"Recovered driver vs TVP  |  R²(single)={R2_1:.2f},  R²(top-{k})={R2_k:.2f}")
    plt.xlabel("time"); plt.legend()
    if save_assets:
        _savefig(os.path.join(out_dir, "sfa1_vs_tvp.png"))
    if visualize:
        plt.show()


    # C NODES (smoothed & z-scored) + z_sm
    x_sm = movavg_cols(x, cfg.smoothing_W)
    x_sm = zscore_cols(x_sm)

    # correlation of each node with the (smoothed) TVP
    node_vs_tvp = [
        np.corrcoef(x_sm[:, i], z_sm - z_sm.mean())[0, 1]
        for i in range(x_sm.shape[1])
    ]
    print("node vs z_sm:", list(enumerate(node_vs_tvp)))

    fig, ax = plt.subplots(figsize=(10, 3.2))
    for i in range(x_sm.shape[1]):
        ax.plot(x_sm[:, i], lw=0.8, label=f"node {i}")
    ax.plot(z_sm, lw=1.5, label="z(t) (smoothed)")
    ax.set_title("Smoothed node signals vs smoothed TVP")
    ax.set_xlabel("time")
    add_smart_legend(ax)
    if save_assets: _savefig(os.path.join(out_dir, "nodes_vs_tvp.png"))
    if visualize: plt.show()

    # optionally also save raw arrays for later analysis
    if save_assets:
        np.savez_compressed(
            os.path.join(out_dir, "arrays.npz"),
            x=x, z=z, z_sm=z_sm, E_raw=E_raw, E_sm=E_sm, Y=Y, A0=A0, A_t=A_t
        )

    # include where we saved things
    summary["out_dir"] = out_dir
    summary["config"] = cfg_dict
    return summary


if __name__ == "__main__":

    # Build base config (your existing pattern)
    cfg = ExperimentConfig()

    summary = run_once(cfg, visualize=True, save_assets=True, out_root="results", run_name=cfg.exp_name)
    print("Saved to:", summary["out_dir"])

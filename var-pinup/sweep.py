#!/usr/bin/env python3
import os, json, argparse, traceback
from dataclasses import replace, asdict
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ExperimentConfig
from run_sim import run_once

DEFAULT_SWEEPS = {
    "Sigma":              [0.0, 0.005, 0.01, 0.02, 0.05, 0.1],
    "tvp_amp":            [0.2, 0.4, 0.6, 0.8, 1.0],
    "tvp_period":         [300, 600, 1000, 1500, 2000],
    "self_memory":        [0.1, 0.2, 0.35, 0.5, 0.7], 
    "background_coupling":[0.0, 0.05, 0.1, 0.2],
}

def parse_values(values_str: str) -> List[float]:
    vals = []
    for tok in values_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except ValueError:
            vals.append(float(tok))
    return vals

def ensure_outdir(p: str):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", type=str, default="Sigma",
                    help="ExperimentConfig field to sweep (e.g., tvp_period, Sigma, self_memory, background_coupling, tvp_amp).")
    ap.add_argument("--values", type=str, default="",
                    help="Comma-separated values. If empty, uses DEFAULT_SWEEPS[param].")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out-dir", type=str, default="sweeps")
    ap.add_argument("--seed", type=int, default=None,
                    help="Base seed; repeats use seed+r.")
    ap.add_argument("--save-json", action="store_true")
    args = ap.parse_args()

    base_cfg = ExperimentConfig()
    if args.seed is not None:
        base_cfg = replace(base_cfg, seed=args.seed)

    if args.values:
        values = parse_values(args.values)
    else:
        if args.param not in DEFAULT_SWEEPS:
            raise ValueError(f"No default sweep values for '{args.param}'. Provide --values.")
        values = DEFAULT_SWEEPS[args.param]

    ensure_outdir(args.out_dir)

    
    csv_path = os.path.join(args.out_dir, f"sweep_{args.param}.csv")
    log_path = os.path.join(args.out_dir, f"sweep_{args.param}.log")

    # incremental write
    cols = ["param","value","repeat","R2_single","R2_topk","seed"]
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=cols).to_csv(csv_path, index=False)

    with open(log_path, "a") as logf:
        for v in values:
            for r in range(args.repeats):
                try:
                    cfg = base_cfg
                    if args.seed is not None:
                        cfg = replace(cfg, seed=args.seed + r)
                    cfg = replace(cfg, **{args.param: v})

                    print(f"[RUN] {args.param}={v}  repeat {r+1}/{args.repeats}")
                    summary = run_once(cfg, visualize=False, save_assets=False)

                    row = {
                        "param": args.param,
                        "value": v,
                        "repeat": r,
                        "R2_single": summary["R2_single"],
                        "R2_topk": summary["R2_topk"],
                        "seed": cfg.seed,
                    }
                    # append immediately
                    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)

                except Exception as e:
                    msg = f"[ERR] {args.param}={v} repeat {r}: {e}\n{traceback.format_exc()}\n"
                    print(msg.strip())
                    logf.write(msg)
                    logf.flush()

    # load aggregated
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[WARN] No successful runs. Check {log_path}.")
        return

    agg = df.groupby(["param","value"]).agg(
        R2_single_mean=("R2_single","mean"),
        R2_single_std=("R2_single","std"),
        R2_topk_mean=("R2_topk","mean"),
        R2_topk_std=("R2_topk","std"),
        n=("R2_single","count"),
    ).reset_index()

    agg_csv = os.path.join(args.out_dir, f"sweep_{args.param}_agg.csv")
    agg.to_csv(agg_csv, index=False)

    sub = agg.sort_values("value")
    x = sub["value"].values

    plt.figure(figsize=(8,4.5))
    plt.plot(x, sub["R2_single_mean"].values, marker="o", label="R² (SFA #1)")
    if args.repeats > 1:
        y = sub["R2_single_mean"].values
        yerr = sub["R2_single_std"].values
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.15, label="±1σ (SFA #1)")
    plt.plot(x, sub["R2_topk_mean"].values, marker="s", linestyle="--", label="R² (top‑k)")
    if args.repeats > 1:
        yk = sub["R2_topk_mean"].values
        ykerr = sub["R2_topk_std"].values
        plt.fill_between(x, yk - ykerr, yk + yerr, alpha=0.10, label="±1σ (top‑k)")

    plt.title(f"Reconstruction quality vs {args.param}")
    plt.xlabel(args.param); plt.ylabel("R²")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

    png_path = os.path.join(args.out_dir, f"sweep_{args.param}.png")
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"[OK] Raw CSV  → {csv_path}")
    print(f"[OK] Agg CSV  → {agg_csv}")
    print(f"[OK] Plot     → {png_path}")

if __name__ == "__main__":
    main()

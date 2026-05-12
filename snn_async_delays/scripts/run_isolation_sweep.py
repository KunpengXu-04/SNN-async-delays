"""
Experiment A: Isolation Breakdown Sweep.

Sweeps tau_m (Part 1) and gap_len (Part 2) to study cross-slot interference.
Imports run_single() from run_step2 — no training logic is duplicated.

Usage:
    python -m scripts.run_isolation_sweep \\
        --config configs/step2_isolation_sweep.yaml \\
        --device auto --runs_dir runs
"""

import argparse
import copy
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from scripts.run_step2 import run_single


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_run_name(op: str, cname: str, h: int, K: int,
                   tau_m: float, gap: int, seed: int) -> str:
    return (f"step2_isolation_{op}_{cname}_h{h}_K{K}"
            f"_taum{int(tau_m)}_gap{gap}_seed{seed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/step2_isolation_sweep.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runs_dir", default="runs")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    cfg = load_cfg(args.config)
    iso = cfg["isolation_sweep"]
    conditions = iso["conditions"]
    h    = cfg["hidden_size"]
    K    = iso["K"]
    op   = iso["op"]
    seed = cfg["seed"]

    all_results = []

    # ── Part 1: tau_m sweep ───────────────────────────────────────────────
    gap = iso["tau_m_sweep"]["gap_len"]
    print(f"\n{'='*60}")
    print(f"Part 1: tau_m sweep  (gap_len={gap} fixed, K={K})")
    print(f"{'='*60}")
    for tau_m in iso["tau_m_sweep"]["tau_m_values"]:
        for cond in conditions:
            cname    = cond["name"]
            run_name = build_run_name(op, cname, h, K, tau_m, gap, seed)
            run_cfg  = copy.deepcopy(cfg)
            run_cfg["lif_tau_m"] = float(tau_m)
            run_cfg["gap_len"]   = int(gap)
            run_cfg["op_name"]   = op
            res = run_single(run_cfg, K=K, condition=cond, device=device,
                             base_runs_dir=args.runs_dir,
                             run_name_override=run_name)
            res["tau_m"]   = tau_m
            res["gap_len"] = gap
            res["part"]    = "tau_m_sweep"
            all_results.append(res)

    # ── Part 2: gap_len sweep ─────────────────────────────────────────────
    tau_m = float(iso["gap_len_sweep"]["tau_m"])
    print(f"\n{'='*60}")
    print(f"Part 2: gap_len sweep  (tau_m={tau_m} fixed, K={K})")
    print(f"{'='*60}")
    for gap in iso["gap_len_sweep"]["gap_len_values"]:
        for cond in conditions:
            cname    = cond["name"]
            run_name = build_run_name(op, cname, h, K, tau_m, gap, seed)
            run_cfg  = copy.deepcopy(cfg)
            run_cfg["lif_tau_m"] = tau_m
            run_cfg["gap_len"]   = int(gap)
            run_cfg["op_name"]   = op
            res = run_single(run_cfg, K=K, condition=cond, device=device,
                             base_runs_dir=args.runs_dir,
                             run_name_override=run_name)
            res["tau_m"]   = tau_m
            res["gap_len"] = gap
            res["part"]    = "gap_len_sweep"
            all_results.append(res)

    # ── Save summary ──────────────────────────────────────────────────────
    out_json = os.path.join(args.runs_dir, "step2_isolation_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    cols = ["part", "tau_m", "gap_len", "condition", "K",
            "accuracy", "throughput_K_per_spk", "ops_per_neuron_per_ms",
            "mean_hidden_spikes"]
    out_csv = os.path.join(args.runs_dir, "step2_isolation_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    print(f"\nSummary saved to: {out_json}")
    print(f"Summary saved to: {out_csv}")

    # ── Print result table ────────────────────────────────────────────────
    print(f"\n{'part':12s}  {'tau_m':5s}  {'gap':3s}  {'condition':25s}  acc")
    print("-" * 65)
    for r in all_results:
        print(f"{r['part']:12s}  {r['tau_m']:5}  {r['gap_len']:3}  "
              f"{r['condition']:25s}  {r['accuracy']:.4f}")


if __name__ == "__main__":
    main()

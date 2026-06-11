"""Backfill diagnostic plots for Step 3 Plan D runs whose plots/ directory
is empty (due to the op="mixed" trace-extraction bug, fixed in utils/viz.py).

Usage:
    python -m scripts.backfill_step3_plots [--device cpu] [runs_dir ...]

If no runs_dir is given, scans the default set of Step 3 sweep directories.
"""
import argparse
import json
import os

import torch

from snn.model import SNNSimultaneousModel
from utils.viz import save_run_diagnostic_plots

DEFAULT_RUNS_DIRS = [
    "runs/step3_planD",
    "runs/step3_planD_4ops",
    "runs/step3_planD_4ops_16k",
    "runs/step3_planD_4ops_16k_h100",
]


def backfill_one(run_dir: str, device: str, force: bool) -> str:
    eval_path = os.path.join(run_dir, "eval_results.json")
    cfg_path = os.path.join(run_dir, "config.json")
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    plot_dir = os.path.join(run_dir, "plots")
    npz_path = os.path.join(plot_dir, "diagnostic_data.npz")

    if not (os.path.exists(eval_path) and os.path.exists(cfg_path)
            and os.path.exists(ckpt_path)):
        return "skip (incomplete run)"

    if os.path.exists(npz_path) and not force:
        return "skip (already has diagnostic_data.npz)"

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    with open(eval_path, encoding="utf-8") as f:
        eval_results = json.load(f)

    K = cfg["K"]
    model = SNNSimultaneousModel(
        n_queries=K, n_hidden=cfg["hidden_sizes"][0],
        win_len=cfg["win_len"], read_len=cfg["read_len"], d_max=cfg["d_max"],
        train_mode=cfg["train_mode"],
        delay_param_type=cfg["delay_param_type"], delay_step=cfg["delay_step"],
        fixed_delay_value=cfg["fixed_delay_value"],
        lif_tau_m=cfg["lif_tau_m"], lif_threshold=cfg["lif_threshold"],
        lif_reset=cfg["lif_reset"], lif_refractory=cfg["lif_refractory"],
        dt=cfg["dt"], surrogate_beta=cfg["surrogate_beta"],
        n_input_channels=cfg["n_input"], readout_type=cfg["readout_type"],
        num_hidden_layers=cfg["num_hidden_layers"], hidden_sizes=cfg["hidden_sizes"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    log_rows = []
    log_csv = os.path.join(run_dir, "train_log.csv")
    if os.path.exists(log_csv):
        import csv
        with open(log_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                log_rows.append({k: (int(v) if k == "epoch" else float(v))
                                  for k, v in row.items()})

    save_run_diagnostic_plots(
        model=model, cfg={**cfg, "model_name": cfg.get("name", "model")},
        log_rows=log_rows, eval_results=eval_results, run_dir=run_dir,
        K=K, op="mixed", device=device, seed=999,
    )

    if os.path.exists(npz_path):
        return "OK"
    return "FAILED (see warnings)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dirs", nargs="*", default=DEFAULT_RUNS_DIRS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true",
                         help="Regenerate even if diagnostic_data.npz exists")
    args = parser.parse_args()

    for runs_dir in args.runs_dirs:
        if not os.path.isdir(runs_dir):
            print(f"{runs_dir}: not found, skipping")
            continue
        for name in sorted(os.listdir(runs_dir)):
            run_dir = os.path.join(runs_dir, name)
            if not os.path.isdir(run_dir) or name == "plots":
                continue
            status = backfill_one(run_dir, args.device, args.force)
            print(f"{run_dir}: {status}")


if __name__ == "__main__":
    main()

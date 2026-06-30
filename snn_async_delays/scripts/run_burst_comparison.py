"""
Burst vs Rate Encoding Comparison Runner.

Sweeps conditions × K × seeds for both Plan D (sequential) and Plan C (simultaneous).
Each condition carries its own encoding_mode and burst params.

Usage:
    # Plan D (sequential, default)
    python -m scripts.run_burst_comparison \\
        --config configs/burst_comparison_sequential.yaml

    # Plan C (simultaneous)
    python -m scripts.run_burst_comparison \\
        --config configs/burst_comparison_simultaneous.yaml --plan simultaneous

    # GPU
    python -m scripts.run_burst_comparison \\
        --config configs/burst_comparison_sequential.yaml --device cuda

    # Single quick run for testing
    python -m scripts.run_burst_comparison \\
        --config configs/burst_comparison_sequential.yaml \\
        --condition burst_wad --K 2 --seed 42 --epochs 5
"""

import argparse
import copy
import csv
import json
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader

from data.boolean_dataset import MultiQueryDataset
from data.encoding import encode_sequential_trial, encode_simultaneous_trial
from snn.model import SNNSimultaneousModel
from train.trainer import SimultaneousTrainer
from train.eval import evaluate_simultaneous, max_K_at_threshold, save_eval_results
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.viz import (
    plot_training_curves, plot_delay_distribution, plot_delay_histogram,
    save_run_diagnostic_plots,
)


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_encode_fn(base_encode, cond: dict, cfg: dict):
    """Build encode_fn with burst params merged from condition > config."""
    def _get(key, default):
        return cond.get(key, cfg.get(key, default))
    return partial(
        base_encode,
        encoding_mode=_get("encoding_mode", "rate"),
        burst_n_spikes_on=_get("burst_n_spikes_on", 2),
        burst_n_spikes_off=_get("burst_n_spikes_off", 1),
        burst_phase_on=_get("burst_phase_on", 0.2),
        burst_phase_off=_get("burst_phase_off", 0.8),
        burst_jitter_ms=_get("burst_jitter_ms", 0),
    )


def run_single_sequential(cfg: dict, K: int, condition: dict, seed: int,
                           device: str, base_runs_dir: str) -> dict:
    """Single Plan D run: sequential shared-channel injection."""
    run_cfg = copy.deepcopy(cfg)
    run_cfg["seed"] = seed
    set_seed(seed)
    logger = setup_logger("burst_seq")

    op       = run_cfg["op_name"]
    h        = run_cfg["n_hidden"]
    cname    = condition["name"]
    sub_win  = run_cfg.get("sub_win", 10)
    win_len  = K * sub_win
    d_max    = win_len
    run_cfg.update({"win_len": win_len, "d_max": d_max})

    readout_type = run_cfg.get("readout_type", "linear")
    rt_suffix = f"_rt{readout_type}" if readout_type != "linear" else ""
    run_name = f"burst_seq_{op}_{cname}_h{h}_K{K}_sw{sub_win}_seed{seed}{rt_suffix}"
    run_dir  = os.path.join(base_runs_dir, run_name)

    eval_path = os.path.join(run_dir, "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            results = json.load(f)
        logger.info(f"=== {run_name} [SKIPPED, acc={results.get('accuracy', '?'):.4f}] ===")
        return results

    logger.info(f"=== {run_name} (K={K}, sub_win={sub_win}, enc={condition.get('encoding_mode','rate')}) ===")

    _encode_fn = _make_encode_fn(encode_sequential_trial, condition, run_cfg)

    def make_loader(split):
        seed_off = {"train": 0, "val": 1, "test": 2}[split]
        ds = MultiQueryDataset(
            K=K, n_samples=run_cfg[f"n_{split}"], same_op=True, op_name=op,
            ops_list=run_cfg["ops_list"], seed=seed + seed_off,
        )
        return DataLoader(ds, batch_size=run_cfg["batch_size"], shuffle=(split == "train"))

    train_loader = make_loader("train")
    val_loader   = make_loader("val")
    test_loader  = make_loader("test")

    model = SNNSimultaneousModel(
        n_queries=K,
        n_hidden=h,
        win_len=win_len,
        read_len=run_cfg["read_len"],
        d_max=d_max,
        train_mode=condition["train_mode"],
        delay_param_type=condition.get("delay_param_type", run_cfg.get("delay_param_type", "sigmoid")),
        delay_step=condition.get("delay_step", run_cfg.get("delay_step", 1.0)),
        fixed_delay_value=condition.get("fixed_delay_value", run_cfg.get("fixed_delay_value", None)),
        lif_tau_m=run_cfg["lif_tau_m"],
        lif_threshold=run_cfg["lif_threshold"],
        lif_reset=run_cfg["lif_reset"],
        lif_refractory=run_cfg["lif_refractory"],
        dt=run_cfg["dt"],
        surrogate_beta=run_cfg["surrogate_beta"],
        n_input_channels=2,
        readout_type=readout_type,
    )

    trainer = SimultaneousTrainer(model, run_cfg, run_dir, device, encode_fn=_encode_fn)
    trainer.save_config({
        **run_cfg, **condition,
        "K": K, "n_input": 2, "sub_win": sub_win,
        "readout_type": readout_type, "experiment": "burst_comparison_seqD",
    })
    log_rows = trainer.fit(train_loader, val_loader, run_cfg["epochs"])

    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device))
    results = evaluate_simultaneous(model, test_loader, run_cfg, device, encode_fn=_encode_fn)
    results.update({
        "op": op, "condition": cname, "train_mode": condition["train_mode"],
        "hidden_size": h, "K": K, "sub_win": sub_win, "seed": seed,
        "encoding_mode": condition.get("encoding_mode", "rate"),
        "experiment": "burst_comparison_seqD",
    })
    save_eval_results(results, eval_path)

    plot_dir = os.path.join(run_dir, "plots")
    plot_training_curves(log_rows, os.path.join(plot_dir, "training_curves.png"))
    d_ih = model.get_delays()["ih"].detach().cpu().numpy()
    plot_delay_distribution(d_ih, run_name, os.path.join(plot_dir, "delays_ih.png"))
    plot_delay_histogram(d_ih, os.path.join(plot_dir, "delays_ih_hist.png"), title=run_name)

    # Enhanced diagnostic plots: diagnostic_data.npz + enhanced_raster.png + enhanced_flow.png
    save_run_diagnostic_plots(model, run_cfg, log_rows, results, run_dir, K, op, device)

    logger.info(f"K={K} enc={condition.get('encoding_mode','rate')}  "
                f"acc={results['accuracy']:.4f}  K/spk={results['throughput_K_per_spk']:.4f}")
    return results


def run_single_simultaneous(cfg: dict, K: int, condition: dict, seed: int,
                             device: str, base_runs_dir: str) -> dict:
    """Single Plan C run: simultaneous dedicated-channel injection."""
    run_cfg = copy.deepcopy(cfg)
    run_cfg["seed"] = seed
    set_seed(seed)
    logger = setup_logger("burst_simul")

    op    = run_cfg["op_name"]
    h     = run_cfg["n_hidden"]
    cname = condition["name"]
    run_name = f"burst_simul_{op}_{cname}_h{h}_K{K}_seed{seed}"
    run_dir  = os.path.join(base_runs_dir, run_name)

    eval_path = os.path.join(run_dir, "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            results = json.load(f)
        logger.info(f"=== {run_name} [SKIPPED, acc={results.get('accuracy', '?'):.4f}] ===")
        return results

    logger.info(f"=== {run_name} (K={K}, enc={condition.get('encoding_mode','rate')}) ===")

    _encode_fn = _make_encode_fn(encode_simultaneous_trial, condition, run_cfg)

    def make_loader(split):
        seed_off = {"train": 0, "val": 1, "test": 2}[split]
        ds = MultiQueryDataset(
            K=K, n_samples=run_cfg[f"n_{split}"], same_op=True, op_name=op,
            ops_list=run_cfg["ops_list"], seed=seed + seed_off,
        )
        return DataLoader(ds, batch_size=run_cfg["batch_size"], shuffle=(split == "train"))

    train_loader = make_loader("train")
    val_loader   = make_loader("val")
    test_loader  = make_loader("test")

    model = SNNSimultaneousModel(
        n_queries=K, n_hidden=h,
        win_len=run_cfg["win_len"], read_len=run_cfg["read_len"],
        d_max=run_cfg["d_max"],
        train_mode=condition["train_mode"],
        delay_param_type=condition.get("delay_param_type", run_cfg.get("delay_param_type", "sigmoid")),
        delay_step=condition.get("delay_step", run_cfg.get("delay_step", 1.0)),
        fixed_delay_value=condition.get("fixed_delay_value", run_cfg.get("fixed_delay_value", None)),
        lif_tau_m=run_cfg["lif_tau_m"], lif_threshold=run_cfg["lif_threshold"],
        lif_reset=run_cfg["lif_reset"], lif_refractory=run_cfg["lif_refractory"],
        dt=run_cfg["dt"], surrogate_beta=run_cfg["surrogate_beta"],
    )

    trainer = SimultaneousTrainer(model, run_cfg, run_dir, device, encode_fn=_encode_fn)
    trainer.save_config({
        **run_cfg, **condition, "K": K, "n_input": 2 * K,
        "experiment": "burst_comparison_simC",
    })
    log_rows = trainer.fit(train_loader, val_loader, run_cfg["epochs"])

    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device))
    results = evaluate_simultaneous(model, test_loader, run_cfg, device, encode_fn=_encode_fn)
    results.update({
        "op": op, "condition": cname, "train_mode": condition["train_mode"],
        "hidden_size": h, "K": K, "seed": seed,
        "encoding_mode": condition.get("encoding_mode", "rate"),
        "experiment": "burst_comparison_simC",
    })
    save_eval_results(results, eval_path)

    plot_dir = os.path.join(run_dir, "plots")
    plot_training_curves(log_rows, os.path.join(plot_dir, "training_curves.png"))
    d_ih = model.get_delays()["ih"].detach().cpu().numpy()
    plot_delay_distribution(d_ih, run_name, os.path.join(plot_dir, "delays_ih.png"))
    plot_delay_histogram(d_ih, os.path.join(plot_dir, "delays_ih_hist.png"), title=run_name)

    # Enhanced diagnostic plots
    save_run_diagnostic_plots(model, run_cfg, log_rows, results, run_dir, K, op, device)

    logger.info(f"K={K} enc={condition.get('encoding_mode','rate')}  "
                f"acc={results['accuracy']:.4f}  K/spk={results['throughput_K_per_spk']:.4f}")
    return results


def _save_summary_csv(all_results: dict, out_path: str):
    """Write flat summary CSV with one row per (condition, K, seed)."""
    rows = []
    for cname, k_seed_res in all_results.items():
        for K, seed_res in k_seed_res.items():
            for seed, res in seed_res.items():
                rows.append({
                    "condition": cname,
                    "K": K,
                    "seed": seed,
                    "encoding_mode": res.get("encoding_mode", "rate"),
                    "accuracy": res.get("accuracy", float("nan")),
                    "throughput_K_per_spk": res.get("throughput_K_per_spk", float("nan")),
                    "mean_hidden_spikes": res.get("mean_hidden_spikes", float("nan")),
                    "train_mode": res.get("train_mode", ""),
                })
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/burst_comparison_sequential.yaml")
    parser.add_argument("--plan",      default="sequential",
                        choices=["sequential", "simultaneous"],
                        help="Plan D (sequential) or Plan C (simultaneous)")
    parser.add_argument("--condition", default=None, help="Single condition name to run")
    parser.add_argument("--K",         default=None, type=int)
    parser.add_argument("--seed",      default=None, type=int)
    parser.add_argument("--epochs",    default=None, type=int)
    parser.add_argument("--device",    default="auto")
    parser.add_argument("--runs_dir",  default="runs")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    base_cfg = load_cfg(args.config)
    if args.epochs is not None:
        base_cfg["epochs"] = args.epochs

    sweep = base_cfg.get("sweep", {})
    K_vals   = sweep.get("K_values", [base_cfg.get("K", 2)])
    seeds    = sweep.get("seeds", [base_cfg.get("seed", 42)])
    tau_list = sorted(set(float(t) for t in sweep.get("accuracy_thresholds", [0.95, 0.90])),
                      reverse=True)

    raw_conds = sweep.get("conditions", [])
    conditions = []
    for c in raw_conds:
        cond = {
            "name":              c.get("name", c.get("train_mode", "wad")),
            "train_mode":        c.get("train_mode", base_cfg.get("train_mode", "weights_and_delays")),
            "delay_param_type":  c.get("delay_param_type", base_cfg.get("delay_param_type", "sigmoid")),
            "delay_step":        c.get("delay_step", base_cfg.get("delay_step", 1.0)),
            "fixed_delay_value": c.get("fixed_delay_value", base_cfg.get("fixed_delay_value", None)),
            "encoding_mode":     c.get("encoding_mode", base_cfg.get("encoding_mode", "rate")),
            "burst_n_spikes_on": c.get("burst_n_spikes_on", base_cfg.get("burst_n_spikes_on", 2)),
            "burst_n_spikes_off":c.get("burst_n_spikes_off", base_cfg.get("burst_n_spikes_off", 1)),
            "burst_phase_on":    c.get("burst_phase_on", base_cfg.get("burst_phase_on", 0.2)),
            "burst_phase_off":   c.get("burst_phase_off", base_cfg.get("burst_phase_off", 0.8)),
            "burst_jitter_ms":   c.get("burst_jitter_ms", base_cfg.get("burst_jitter_ms", 0)),
        }
        conditions.append(cond)

    # Single-run mode
    if args.K is not None:
        K = args.K
        seed = args.seed if args.seed is not None else seeds[0]
        cond = next((c for c in conditions if c["name"] == args.condition), conditions[0])
        run_fn = (run_single_sequential if args.plan == "sequential"
                  else run_single_simultaneous)
        run_fn(base_cfg, K, cond, seed, device, args.runs_dir)
        return

    # Filter conditions if requested
    if args.condition is not None:
        conditions = [c for c in conditions if c["name"] == args.condition]
        if not conditions:
            raise ValueError(f"Condition '{args.condition}' not found in config.")

    run_fn = (run_single_sequential if args.plan == "sequential"
              else run_single_simultaneous)
    exp_tag = "burst_comparison_seqD" if args.plan == "sequential" else "burst_comparison_simC"
    runs_dir = os.path.join(args.runs_dir, exp_tag)

    # Full sweep: conditions × K × seeds
    all_results: dict = {}   # [cname][K][seed]
    for cond in conditions:
        cname = cond["name"]
        all_results[cname] = {}
        for K in K_vals:
            all_results[cname][K] = {}
            for seed in (args.seed,) if args.seed is not None else seeds:
                cfg = copy.deepcopy(base_cfg)
                res = run_fn(cfg, K, cond, seed, device, runs_dir)
                all_results[cname][K][seed] = res

    # Build per-condition average results for max_K_at_threshold
    avg_results: dict = {}   # [cname][K]
    for cname, k_seed_res in all_results.items():
        avg_results[cname] = {}
        for K, seed_res in k_seed_res.items():
            accs = [r.get("accuracy", float("nan")) for r in seed_res.values()]
            avg_results[cname][K] = {
                "accuracy": sum(accs) / len(accs) if accs else float("nan"),
            }

    summary = {
        cname: {
            "max_K_by_tau": {str(t): max_K_at_threshold(avg_results[cname], t)
                             for t in tau_list},
            "avg_accuracy_by_K": {K: avg_results[cname][K]["accuracy"] for K in K_vals},
        }
        for cname in all_results
    }

    summary_json = os.path.join(runs_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    summary_csv = os.path.join(runs_dir, "summary.csv")
    _save_summary_csv(all_results, summary_csv)

    # Print result table
    print(f"\n{'='*72}")
    print(f"Burst vs Rate — Plan {'D (sequential)' if args.plan == 'sequential' else 'C (simultaneous)'}")
    print(f"{'='*72}")
    print(f"{'Condition':22s}  {'K':>4}  {'avg_acc':>8}  max_K@90%  max_K@95%")
    print("-" * 60)
    for cname, s in summary.items():
        for K in K_vals:
            acc = s["avg_accuracy_by_K"].get(K, float("nan"))
            mk90 = s["max_K_by_tau"].get("0.9", "?")
            mk95 = s["max_K_by_tau"].get("0.95", "?")
            print(f"{cname:22s}  {K:4d}  {acc:8.4f}  {str(mk90):>9}  {str(mk95):>9}")
        print()

    print(f"\nSummary JSON:  {summary_json}")
    print(f"Summary CSV:   {summary_csv}")


if __name__ == "__main__":
    main()

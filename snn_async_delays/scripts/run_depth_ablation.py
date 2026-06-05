"""
Depth Ablation for Plan D Sequential Temporal Multiplexing.

Runs 4 model variants × K_values × seeds, then writes a combined
summary CSV/JSON and generates comparison plots.

Usage:
    python -m scripts.run_depth_ablation \\
        --config configs/step2_depth_ablation.yaml --device cuda
    python -m scripts.run_depth_ablation --config configs/step2_depth_ablation.yaml \\
        --K 3 --model L2-h25h25-linear --seed 42 --device cuda
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
from torch.utils.data import DataLoader

from data.boolean_dataset import MultiQueryDataset
from data.encoding import encode_sequential_trial
from snn.model import SNNSimultaneousModel
from train.trainer import SimultaneousTrainer
from train.eval import evaluate_simultaneous, max_K_at_threshold, save_eval_results
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.viz import (plot_training_curves, plot_delay_histogram,
                       plot_depth_ablation_curves, save_run_diagnostic_plots)


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single(
    base_cfg: dict,
    model_cfg: dict,
    K: int,
    seed: int,
    device: str,
    runs_dir: str,
) -> dict:
    """Train and evaluate one (model_variant, K, seed) combination."""
    set_seed(seed)
    logger = setup_logger("depth_ablation")

    op       = base_cfg["op_name"]
    sub_win  = base_cfg.get("sub_win", 10)
    win_len  = K * sub_win
    d_max    = win_len
    mname    = model_cfg["name"]
    run_cfg  = {**base_cfg, "win_len": win_len, "d_max": d_max, "seed": seed}

    run_name = f"{mname}_K{K}_seed{seed}"
    run_dir  = os.path.join(runs_dir, run_name)
    eval_path = os.path.join(run_dir, "eval_results.json")

    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            results = json.load(f)
        logger.info(
            f"=== {run_name} [SKIPPED — done, "
            f"acc={results.get('accuracy', '?'):.4f}] ==="
        )
        return results

    logger.info(f"=== {run_name}  (K={K}, sub_win={sub_win}) ===")

    def make_loader(split):
        seed_off = {"train": 0, "val": 1, "test": 2}[split]
        n = run_cfg[f"n_{split}"]
        ds = MultiQueryDataset(
            K=K, n_samples=n, same_op=True, op_name=op,
            ops_list=run_cfg["ops_list"], seed=seed + seed_off,
        )
        return DataLoader(ds, batch_size=run_cfg["batch_size"],
                          shuffle=(split == "train"))

    train_loader = make_loader("train")
    val_loader   = make_loader("val")
    test_loader  = make_loader("test")

    num_layers   = model_cfg.get("num_hidden_layers", 1)
    hidden_sizes = model_cfg.get("hidden_sizes", [run_cfg.get("n_hidden", 50)])
    readout_type = model_cfg.get("readout_type", "linear")
    train_mode   = model_cfg.get("train_mode", "weights_and_delays")
    fixed_dv     = model_cfg.get("fixed_delay_value", None)

    model = SNNSimultaneousModel(
        n_queries         = K,
        n_hidden          = hidden_sizes[0],
        win_len           = win_len,
        read_len          = run_cfg["read_len"],
        d_max             = d_max,
        train_mode        = train_mode,
        delay_param_type  = model_cfg.get("delay_param_type",
                                          run_cfg.get("delay_param_type", "sigmoid")),
        delay_step        = run_cfg.get("delay_step", 1.0),
        fixed_delay_value = fixed_dv,
        lif_tau_m         = run_cfg["lif_tau_m"],
        lif_threshold     = run_cfg["lif_threshold"],
        lif_reset         = run_cfg["lif_reset"],
        lif_refractory    = run_cfg["lif_refractory"],
        dt                = run_cfg["dt"],
        surrogate_beta    = run_cfg["surrogate_beta"],
        n_input_channels  = 2,
        readout_type      = readout_type,
        num_hidden_layers = num_layers,
        hidden_sizes      = hidden_sizes,
    )

    trainer = SimultaneousTrainer(
        model, run_cfg, run_dir, device,
        encode_fn=encode_sequential_trial,
    )
    trainer.save_config({
        **run_cfg, **model_cfg,
        "K": K, "n_input": 2,
        "sub_win": sub_win,
        "num_hidden_layers": num_layers,
        "hidden_sizes": hidden_sizes,
        "readout_type": readout_type,
        "experiment": "depth_ablation",
    })
    log_rows = trainer.fit(train_loader, val_loader, run_cfg["epochs"])

    model.load_state_dict(
        torch.load(os.path.join(run_dir, "best_model.pt"),
                   map_location=device, weights_only=True)
    )
    results = evaluate_simultaneous(
        model, test_loader, run_cfg, device,
        encode_fn=encode_sequential_trial,
    )

    # ── Delay statistics (weight quantities, not per-batch) ──
    delays = model.get_delays()
    d_ih = delays["ih"].detach()
    results["delay_ih_mean"]   = float(d_ih.mean())
    results["delay_ih_std"]    = float(d_ih.std())
    results["delay_ih_median"] = float(d_ih.median())
    if "h1h2" in delays:
        d_h1h2 = delays["h1h2"].detach()
        results["delay_h1h2_mean"]   = float(d_h1h2.mean())
        results["delay_h1h2_std"]    = float(d_h1h2.std())
        results["delay_h1h2_median"] = float(d_h1h2.median())
    else:
        results["delay_h1h2_mean"] = results["delay_h1h2_std"] = results["delay_h1h2_median"] = None

    results.update({
        "model_name":       mname,
        "num_hidden_layers": num_layers,
        "hidden_sizes":     hidden_sizes,
        "readout_type":     readout_type,
        "train_mode":       train_mode,
        "op":               op,
        "K":                K,
        "seed":             seed,
        "sub_win":          sub_win,
        "run_dir":          run_dir,
        "experiment":       "depth_ablation",
    })
    save_eval_results(results, eval_path)

    # ── Plots ──
    save_run_diagnostic_plots(
        model=model,
        cfg={**run_cfg, "model_name": mname,
             "hidden_sizes": hidden_sizes,
             "readout_type": readout_type,
             "num_hidden_layers": num_layers},
        log_rows=log_rows,
        eval_results=results,
        run_dir=run_dir,
        K=K,
        op=op,
        device=device,
        seed=999,
    )

    logger.info(
        f"K={K}  acc={results['accuracy']:.4f}  "
        f"K/spk={results['throughput_K_per_spk']:.4f}  "
        f"delay_ih_mean={results['delay_ih_mean']:.2f}"
    )
    return results


def _write_summary_csv(all_results: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = [
        "model_name", "num_hidden_layers", "hidden_sizes", "readout_type",
        "train_mode", "K", "seed",
        "accuracy", "per_query_acc_mean",
        "mean_hidden_spikes", "layer1_hidden_spikes", "layer2_hidden_spikes",
        "throughput_K_per_spk", "mean_active_hidden_fraction",
        "delay_ih_mean", "delay_ih_median", "delay_ih_std",
        "delay_h1h2_mean", "delay_h1h2_median", "delay_h1h2_std",
        "run_dir",
    ]
    rows = []
    for r in all_results:
        pqa = r.get("per_query_acc", [])
        row = {c: r.get(c) for c in cols}
        row["per_query_acc_mean"] = float(sum(pqa) / len(pqa)) if pqa else None
        row["hidden_sizes"] = str(r.get("hidden_sizes", ""))
        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/step2_depth_ablation.yaml")
    parser.add_argument("--K",       default=None, type=int, help="Single K override")
    parser.add_argument("--model",   default=None, help="Single model name override")
    parser.add_argument("--seed",    default=None, type=int)
    parser.add_argument("--epochs",  default=None, type=int)
    parser.add_argument("--device",  default="auto")
    parser.add_argument("--runs_dir", default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_cfg = load_cfg(args.config)
    if args.epochs is not None:
        base_cfg["epochs"] = args.epochs

    runs_dir = args.runs_dir or os.path.join(base, "runs", "depth_ablation")

    sweep = base_cfg.get("sweep", {})
    K_vals     = [args.K]    if args.K    is not None else sweep.get("K_values", [2, 3, 4])
    seeds      = [args.seed] if args.seed is not None else sweep.get("seeds", [42, 0])
    model_cfgs = sweep.get("models", [])
    if args.model is not None:
        model_cfgs = [m for m in model_cfgs if m["name"] == args.model]
        if not model_cfgs:
            print(f"Model '{args.model}' not found in config sweep.models")
            return

    tau_list = sorted(
        {float(t) for t in sweep.get("accuracy_thresholds", [0.95, 0.90])},
        reverse=True,
    )

    all_results: list[dict] = []
    for model_cfg in model_cfgs:
        for K in K_vals:
            for seed in seeds:
                cfg = copy.deepcopy(base_cfg)
                res = run_single(cfg, model_cfg, K, seed, device, runs_dir)
                all_results.append(res)

    if not all_results:
        print("No runs completed.")
        return

    # ── Write summary ──
    plot_dir = os.path.join(runs_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    csv_path  = os.path.join(runs_dir, "depth_ablation_summary.csv")
    json_path = os.path.join(runs_dir, "depth_ablation_summary.json")

    _write_summary_csv(all_results, csv_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Aggregated table ──
    print(f"\n{'='*72}")
    print("Depth Ablation Summary — Plan D Sequential (NAND, MLP readout=linear)")
    print(f"{'='*72}")
    print(f"{'Model':25s}  {'K':>3}  {'Acc (mean±range)':>20}  {'K/spk':>8}")
    print("-" * 65)

    # Group by (model_name, K)
    from collections import defaultdict
    grouped: dict = defaultdict(list)
    for r in all_results:
        grouped[(r.get("model_name", "?"), r.get("K", 0))].append(r)

    for (mname, K), runs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        accs   = [r["accuracy"] for r in runs if r.get("accuracy") is not None]
        spks   = [r["throughput_K_per_spk"] for r in runs
                  if r.get("throughput_K_per_spk") is not None
                  and not (r["throughput_K_per_spk"] != r["throughput_K_per_spk"])]  # nan check
        if accs:
            acc_str = f"{sum(accs)/len(accs)*100:.1f}% ± {(max(accs)-min(accs))*100:.1f}%"
        else:
            acc_str = "n/a"
        spk_str = f"{sum(spks)/len(spks):.3f}" if spks else "n/a"
        print(f"{mname:25s}  {K:>3}  {acc_str:>20}  {spk_str:>8}")

    # Max K@tau by model
    print()
    models_all = {r.get("model_name") for r in all_results}
    for mname in sorted(models_all):
        m_runs = [r for r in all_results if r.get("model_name") == mname]
        # Average over seeds per K
        from collections import defaultdict as dd
        k_accs: dict = dd(list)
        for r in m_runs:
            k_accs[r["K"]].append(r["accuracy"])
        k_mean = {K: sum(v)/len(v) for K, v in k_accs.items()}
        for tau in tau_list:
            mk = max((K for K, acc in k_mean.items() if acc >= tau), default=0)
            print(f"  {mname:25s}  tau={tau:.0%}  max_K={mk}")

    # ── Generate summary plot ──
    plot_depth_ablation_curves(
        all_results,
        save_path=os.path.join(plot_dir, "depth_ablation_summary.png"),
    )

    print(f"\nSummary CSV  : {csv_path}")
    print(f"Summary JSON : {json_path}")
    print(f"Summary plot : {plot_dir}/depth_ablation_summary.png")


if __name__ == "__main__":
    main()

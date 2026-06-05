"""
Plan D hidden-size sweep — fills the K vs Neurons plot.

For each (h, K) combination, trains Plan D (shared 2-channel, sequential
injection) with both trainable-delay and d=0-baseline conditions.
Uses MLP readout for a fair comparison (same readout capacity for both).

After running, use scripts/plot_k_vs_neurons.py to generate plots A and B.

Usage (from snn_async_delays/):
    python -m scripts.run_plan_d_h_sweep
    python -m scripts.run_plan_d_h_sweep --device cuda
    python -m scripts.run_plan_d_h_sweep --K_values 1 2 3 --h_values 10 20 50
    python -m scripts.run_plan_d_h_sweep --dry_run   # print run list, no training
"""

import argparse
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
from utils.viz import (
    plot_training_curves, plot_delay_distribution, plot_delay_histogram,
)


BASE_CFG = {
    "dt": 1.0,
    "sub_win": 10,
    "read_len": 10,
    "lif_tau_m": 10.0,
    "lif_threshold": 1.0,
    "lif_reset": 0.0,
    "lif_refractory": 2,
    "surrogate_beta": 4.0,
    "delay_param_type": "sigmoid",
    "delay_step": 1.0,
    "lr_w": 1e-3,
    "lr_d": 1e-3,
    "lr_readout": 1e-3,
    "batch_size": 128,
    "epochs": 200,
    "spike_penalty": 0.0,
    "delay_penalty": 0.0,
    "grad_clip": 1.0,
    "r_on": 400.0,
    "r_off": 10.0,
    "ops_list": ["AND", "OR", "XOR", "XNOR", "NAND", "NOR", "A_IMP_B", "B_IMP_A"],
    "op_name": "NAND",
    "same_op": True,
    "n_train": 4000,
    "n_val": 1000,
    "n_test": 1000,
    "readout_type": "mlp",
    "experiment": "planD_h_sweep",
}

CONDITIONS = [
    {"name": "w_and_d",  "train_mode": "weights_and_delays", "fixed_delay_value": None},
    {"name": "d0",       "train_mode": "weights_only",        "fixed_delay_value": 0.0},
]

DEFAULT_H_VALUES = [10, 20, 30, 50, 75, 100]
DEFAULT_K_VALUES = [1, 2, 3, 4, 5]
DEFAULT_SEEDS    = [42, 0]


def run_single(cfg, K, h, condition, seed, device, runs_dir, dry_run=False):
    sub_win  = cfg["sub_win"]
    win_len  = K * sub_win
    d_max    = win_len
    cname    = condition["name"]
    run_name = f"planD_sweep_{cfg['op_name']}_{cname}_h{h}_K{K}_sw{sub_win}_seed{seed}"
    run_dir  = os.path.join(runs_dir, run_name)
    eval_path = os.path.join(run_dir, "eval_results.json")

    if dry_run:
        print(f"  [DRY] {run_name}")
        return None

    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            r = json.load(f)
        print(f"  [SKIP] {run_name}  acc={r['accuracy']:.4f}")
        return r

    set_seed(seed)
    logger = setup_logger("planD_h_sweep")
    logger.info(f"=== {run_name}  (h={h}, K={K}, cond={cname}, seed={seed}) ===")

    run_cfg = {
        **cfg,
        "n_hidden": h,
        "win_len": win_len,
        "d_max": d_max,
        "seed": seed,
        **{k: v for k, v in condition.items() if k != "name"},
    }

    def make_loader(split):
        n = run_cfg[f"n_{split}"]
        ds = MultiQueryDataset(
            K=K, n_samples=n, same_op=True, op_name=run_cfg["op_name"],
            ops_list=run_cfg["ops_list"],
            seed=seed + {"train": 0, "val": 1, "test": 2}[split],
        )
        return DataLoader(ds, batch_size=run_cfg["batch_size"],
                          shuffle=(split == "train"))

    model = SNNSimultaneousModel(
        n_queries        = K,
        n_hidden         = h,
        win_len          = win_len,
        read_len         = run_cfg["read_len"],
        d_max            = d_max,
        train_mode       = condition["train_mode"],
        delay_param_type = run_cfg["delay_param_type"],
        delay_step       = run_cfg.get("delay_step", 1.0),
        fixed_delay_value= condition.get("fixed_delay_value"),
        lif_tau_m        = run_cfg["lif_tau_m"],
        lif_threshold    = run_cfg["lif_threshold"],
        lif_reset        = run_cfg["lif_reset"],
        lif_refractory   = run_cfg["lif_refractory"],
        dt               = run_cfg["dt"],
        surrogate_beta   = run_cfg["surrogate_beta"],
        n_input_channels = 2,
        readout_type     = run_cfg["readout_type"],
    )

    trainer = SimultaneousTrainer(
        model, run_cfg, run_dir, device,
        encode_fn=encode_sequential_trial,
    )
    trainer.save_config({
        **run_cfg, **condition,
        "K": K, "n_input": 2, "sub_win": sub_win,
    })
    log_rows = trainer.fit(
        make_loader("train"), make_loader("val"), run_cfg["epochs"]
    )
    model.load_state_dict(
        torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device)
    )
    results = evaluate_simultaneous(
        model, make_loader("test"), run_cfg, device,
        encode_fn=encode_sequential_trial,
    )
    results.update({
        "op": run_cfg["op_name"], "condition": cname,
        "train_mode": condition["train_mode"],
        "hidden_size": h, "K": K, "sub_win": sub_win,
        "experiment": "planD_h_sweep",
    })
    save_eval_results(results, eval_path)

    plot_dir = os.path.join(run_dir, "plots")
    plot_training_curves(log_rows, os.path.join(plot_dir, "training_curves.png"))
    d_ih = model.get_delays()["ih"].detach().cpu().numpy()
    plot_delay_distribution(d_ih, run_name, os.path.join(plot_dir, "delays_ih.png"))
    plot_delay_histogram(d_ih, os.path.join(plot_dir, "delays_ih_hist.png"),
                         title=f"Delays {run_name}")

    logger.info(f"  acc={results['accuracy']:.4f}  K/spk={results['throughput_K_per_spk']:.4f}")
    return results


def aggregate_summary(results_by_h_K, K_values, h_values, tau_values=(0.95, 0.90)):
    """
    Build summary JSON from collected results.

    results_by_h_K: { (h, K, cond, seed): result_dict }
    """
    summary = {}
    for cond in CONDITIONS:
        cname = cond["name"]
        summary[cname] = {"min_h_by_K_tau": {}, "results": {}}
        for tau in tau_values:
            tau_key = str(tau)
            summary[cname]["min_h_by_K_tau"][tau_key] = {}
            for K in K_values:
                # Average accuracy over seeds for each h
                best_h = None
                for h in sorted(h_values):
                    accs = [
                        v["accuracy"] for (hh, kk, cc, ss), v
                        in results_by_h_K.items()
                        if hh == h and kk == K and cc == cname and v is not None
                    ]
                    if accs and sum(accs) / len(accs) >= tau:
                        best_h = h
                        break
                summary[cname]["min_h_by_K_tau"][tau_key][str(K)] = best_h

        # Store raw results
        for (h, K, cc, seed), v in results_by_h_K.items():
            if cc == cname and v is not None:
                key = f"h{h}_K{K}_seed{seed}"
                summary[cname]["results"][key] = v

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--K_values",  type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--h_values",  type=int, nargs="+", default=DEFAULT_H_VALUES)
    parser.add_argument("--seeds",     type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs",    type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Print run list without training")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(base, "runs", "planD_h_sweep")
    os.makedirs(runs_dir, exist_ok=True)

    cfg = dict(BASE_CFG)
    if args.epochs:
        cfg["epochs"] = args.epochs

    total = len(args.h_values) * len(args.K_values) * len(CONDITIONS) * len(args.seeds)
    print(f"Plan D h-sweep: {total} runs")
    print(f"  h     = {args.h_values}")
    print(f"  K     = {args.K_values}")
    print(f"  conds = {[c['name'] for c in CONDITIONS]}")
    print(f"  seeds = {args.seeds}")
    print(f"  device= {args.device}")
    print()

    results_by_h_K = {}
    run_idx = 0
    for cond in CONDITIONS:
        for h in args.h_values:
            for K in args.K_values:
                for seed in args.seeds:
                    run_idx += 1
                    print(f"[{run_idx}/{total}] h={h} K={K} cond={cond['name']} seed={seed}")
                    r = run_single(cfg, K, h, cond, seed, args.device,
                                   runs_dir, dry_run=args.dry_run)
                    results_by_h_K[(h, K, cond["name"], seed)] = r

    if not args.dry_run:
        summary = aggregate_summary(
            results_by_h_K, args.K_values, args.h_values
        )
        summary_path = os.path.join(runs_dir, "planD_h_sweep_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to {summary_path}")

        # Print min-h table
        print("\n=== Min h needed for 90% accuracy ===")
        print(f"{'K':>4}  {'w_and_d':>10}  {'d0':>10}")
        for K in args.K_values:
            h_wd = summary.get("w_and_d", {}).get("min_h_by_K_tau", {}).get("0.9", {}).get(str(K))
            h_d0 = summary.get("d0",      {}).get("min_h_by_K_tau", {}).get("0.9", {}).get(str(K))
            h_wd_str = str(h_wd) if h_wd is not None else f">{max(args.h_values)}"
            h_d0_str = str(h_d0) if h_d0 is not None else f">{max(args.h_values)}"
            print(f"{K:>4}  {h_wd_str:>10}  {h_d0_str:>10}")

        print(f"\nNext step: python -m scripts.plot_k_vs_neurons --summary {summary_path}")


if __name__ == "__main__":
    main()

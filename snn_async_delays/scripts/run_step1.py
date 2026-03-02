"""
Step 1 – Single-op solvability baseline.

Single run:
    python -m scripts.run_step1 --config configs/step1_singleop.yaml
    python -m scripts.run_step1 --config configs/step1_singleop.yaml \\
        --op NAND --train_mode weights_only --hidden_size 50

Sweep (all ops × modes × hidden sizes):
    python -m scripts.run_step1 --config configs/step1_singleop.yaml --sweep
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader

from data.boolean_dataset import MultiQueryDataset
from snn.model import SNNModel, make_slots
from train.trainer import Trainer
from train.eval import evaluate, save_eval_results
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.viz import plot_training_curves, plot_delay_distribution

import numpy as np


# ---------------------------------------------------------------------------

def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_cli(cfg: dict, args) -> dict:
    """Override cfg fields with CLI arguments."""
    if args.op           is not None: cfg["op_name"]    = args.op
    if args.train_mode   is not None: cfg["train_mode"] = args.train_mode
    if args.hidden_size  is not None: cfg["hidden_size"]= args.hidden_size
    if args.seed         is not None: cfg["seed"]       = args.seed
    if args.epochs       is not None: cfg["epochs"]     = args.epochs
    return cfg


def run_single(cfg: dict, device: str, base_runs_dir: str):
    set_seed(cfg["seed"])
    logger = setup_logger("step1")

    op        = cfg["op_name"]
    mode      = cfg["train_mode"]
    h         = cfg["hidden_size"]
    run_name  = f"step1_{op}_{mode}_h{h}_seed{cfg['seed']}"
    run_dir   = os.path.join(base_runs_dir, run_name)

    logger.info(f"=== {run_name} ===")

    # ---- Data ----
    K = 1   # Step 1: always single query
    slots = make_slots(K, cfg["win_len"], cfg["read_len"], cfg["gap_len"])

    def make_loader(split):
        seed_off = {"train": 0, "val": 1, "test": 2}[split]
        n        = cfg[f"n_{split}"]
        ds = MultiQueryDataset(
            K=K, n_samples=n, same_op=True,
            op_name=op, ops_list=cfg["ops_list"],
            seed=cfg["seed"] + seed_off,
        )
        shuffle = (split == "train")
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle)

    train_loader = make_loader("train")
    val_loader   = make_loader("val")
    test_loader  = make_loader("test")

    # ---- Model ----
    model = SNNModel(
        n_input=cfg["n_input"],
        n_hidden=h,
        d_max=cfg["d_max"],
        train_mode=mode,
        delay_param_type=cfg["delay_param_type"],
        use_output_layer=cfg["use_output_layer"],
        readout_source=cfg["readout_source"],
        lif_tau_m=cfg["lif_tau_m"],
        lif_threshold=cfg["lif_threshold"],
        lif_reset=cfg["lif_reset"],
        lif_refractory=cfg["lif_refractory"],
        dt=cfg["dt"],
        surrogate_beta=cfg["surrogate_beta"],
    )

    # ---- Train ----
    trainer = Trainer(model, slots, cfg, run_dir, device)
    trainer.save_config(cfg)
    log_rows = trainer.fit(train_loader, val_loader, cfg["epochs"])

    # ---- Test evaluation ----
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt")))
    results = evaluate(model, test_loader, slots, cfg, device)
    results.update({"op": op, "train_mode": mode, "hidden_size": h})
    save_eval_results(results, os.path.join(run_dir, "eval_results.json"))

    # ---- Plots ----
    plot_dir = os.path.join(run_dir, "plots")
    plot_training_curves(log_rows, os.path.join(plot_dir, "training_curves.png"))

    d_ih = model.get_delays()["ih"].detach().cpu().numpy()
    plot_delay_distribution(d_ih, f"Delays W_ih  ({run_name})",
                            os.path.join(plot_dir, "delays_ih.png"))

    logger.info(f"Test accuracy: {results['accuracy']:.4f}  "
                f"spikes: {results['mean_hidden_spikes']:.1f}")
    return results


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/step1_singleop.yaml")
    parser.add_argument("--op",          default=None)
    parser.add_argument("--train_mode",  default=None)
    parser.add_argument("--hidden_size", default=None, type=int)
    parser.add_argument("--seed",        default=None, type=int)
    parser.add_argument("--epochs",      default=None, type=int)
    parser.add_argument("--sweep",       action="store_true")
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--runs_dir",    default="runs")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    base_cfg = load_cfg(args.config)

    if args.sweep:
        sweep = base_cfg["sweep"]
        all_results = []
        for op in sweep["ops"]:
            for mode in sweep["train_modes"]:
                for h in sweep["hidden_sizes"]:
                    cfg = copy.deepcopy(base_cfg)
                    cfg["op_name"]    = op
                    cfg["train_mode"] = mode
                    cfg["hidden_size"]= h
                    res = run_single(cfg, device, args.runs_dir)
                    all_results.append(res)

        summary_path = os.path.join(args.runs_dir, "step1_sweep_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSweep summary saved to {summary_path}")
    else:
        cfg = merge_cli(base_cfg, args)
        run_single(cfg, device, args.runs_dir)


if __name__ == "__main__":
    main()

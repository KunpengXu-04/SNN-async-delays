"""
Step 3 – Mixed ops, K queries per trial.

n_input is automatically set to 2 + n_ops (op one-hot encoding).

Single run:
    python -m scripts.run_step3 --config configs/step3_multiquery_multiop.yaml

Sweep:
    python -m scripts.run_step3 --config configs/step3_multiquery_multiop.yaml --sweep
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

from data.boolean_dataset import MultiQueryDataset, OPS_LIST
from snn.model import SNNModel, make_slots
from train.trainer import Trainer
from train.eval import evaluate, max_K_at_threshold, save_eval_results
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.viz import plot_training_curves, plot_K_accuracy, plot_throughput, plot_delay_distribution


# ---------------------------------------------------------------------------

def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single(cfg: dict, K: int, device: str, base_runs_dir: str) -> dict:
    set_seed(cfg["seed"])
    logger = setup_logger("step3")

    ops_list = cfg["ops_list"]
    n_ops    = len(ops_list)
    n_input  = 2 + n_ops          # A, B + one-hot op

    mode     = cfg["train_mode"]
    h        = cfg["hidden_size"]
    run_name = f"step3_mixedops_{mode}_h{h}_K{K}_seed{cfg['seed']}"
    run_dir  = os.path.join(base_runs_dir, run_name)

    logger.info(f"=== {run_name} ===")

    slots = make_slots(K, cfg["win_len"], cfg["read_len"], cfg["gap_len"])

    def make_loader(split):
        seed_off = {"train": 0, "val": 1, "test": 2}[split]
        n        = cfg[f"n_{split}"]
        ds = MultiQueryDataset(
            K=K, n_samples=n, same_op=False,
            op_name=None, ops_list=ops_list,
            seed=cfg["seed"] + seed_off,
        )
        shuffle = (split == "train")
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle)

    train_loader = make_loader("train")
    val_loader   = make_loader("val")
    test_loader  = make_loader("test")

    model = SNNModel(
        n_input=n_input,
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

    trainer = Trainer(model, slots, cfg, run_dir, device)
    trainer.save_config({**cfg, "K": K, "n_input": n_input})
    log_rows = trainer.fit(train_loader, val_loader, cfg["epochs"])

    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt")))
    results = evaluate(model, test_loader, slots, cfg, device)

    # Per-op accuracy breakdown (test set)
    per_query_acc = results.get("per_query_acc", [])
    results.update({
        "train_mode": mode,
        "hidden_size": h,
        "K": K,
        "n_ops": n_ops,
        "per_query_acc": per_query_acc,
    })
    save_eval_results(results, os.path.join(run_dir, "eval_results.json"))

    plot_dir = os.path.join(run_dir, "plots")
    plot_training_curves(log_rows, os.path.join(plot_dir, "training_curves.png"))
    d_ih = model.get_delays()["ih"].detach().cpu().numpy()
    plot_delay_distribution(d_ih, f"Delays  ({run_name})",
                            os.path.join(plot_dir, "delays_ih.png"))

    logger.info(
        f"K={K}  acc={results['accuracy']:.4f}  "
        f"spk={results['mean_hidden_spikes']:.1f}  "
        f"throughput={results['throughput_K_per_spk']:.4f}"
    )
    return results


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/step3_multiquery_multiop.yaml")
    parser.add_argument("--K",           default=None, type=int)
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
    if args.train_mode  is not None: base_cfg["train_mode"]  = args.train_mode
    if args.hidden_size is not None: base_cfg["hidden_size"] = args.hidden_size
    if args.seed        is not None: base_cfg["seed"]        = args.seed
    if args.epochs      is not None: base_cfg["epochs"]      = args.epochs

    if args.sweep:
        sweep = base_cfg["sweep"]
        tau   = sweep.get("accuracy_threshold", 0.95)

        all_results: dict = {}
        for mode in sweep["train_modes"]:
            all_results[mode] = {}
            for h in sweep["hidden_sizes"]:
                for K in sweep["K_values"]:
                    cfg = copy.deepcopy(base_cfg)
                    cfg["train_mode"]  = mode
                    cfg["hidden_size"] = h
                    res = run_single(cfg, K, device, args.runs_dir)
                    all_results[mode][K] = res

        summary = {}
        for mode, k_results in all_results.items():
            mk = max_K_at_threshold(k_results, tau)
            summary[mode] = {"max_K": mk, "results_by_K": k_results}

        summary_path = os.path.join(args.runs_dir, "step3_sweep_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSweep summary saved to {summary_path}")

        K_vals = sweep["K_values"]
        acc_by_mode = {
            mode: [all_results[mode][K]["accuracy"] for K in K_vals]
            for mode in sweep["train_modes"]
        }
        thr_by_mode = {
            mode: [all_results[mode][K]["throughput_K_per_spk"] for K in K_vals]
            for mode in sweep["train_modes"]
        }
        plots_dir = os.path.join(args.runs_dir, "step3_plots")
        plot_K_accuracy(K_vals, acc_by_mode, tau,
                        os.path.join(plots_dir, "K_accuracy.png"))
        plot_throughput(K_vals, thr_by_mode,
                        os.path.join(plots_dir, "K_throughput.png"))
    else:
        K = args.K if args.K is not None else base_cfg["K"]
        run_single(base_cfg, K, device, args.runs_dir)


if __name__ == "__main__":
    main()

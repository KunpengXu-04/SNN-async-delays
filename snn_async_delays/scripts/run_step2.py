"""
Step 2 - Same op, K queries per trial; sweep K to find capacity and efficiency.

Single run:
    python -m scripts.run_step2 --config configs/step2_multiquery_sameop.yaml

Sweep:
    python -m scripts.run_step2 --config configs/step2_multiquery_sameop.yaml --sweep
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
from train.eval import evaluate, max_K_at_threshold, save_eval_results
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.viz import (
    plot_training_curves,
    plot_K_accuracy,
    plot_throughput,
    plot_delay_distribution,
    plot_delay_histogram,
    plot_metric_triplet,
)


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _condition_name(cond: dict) -> str:
    if cond.get("name"):
        return cond["name"]
    mode = cond.get("train_mode", "weights_and_delays")
    dtype = cond.get("delay_param_type", "sigmoid")
    step = cond.get("delay_step", 1)
    if dtype == "quantized":
        return f"{mode}_q{step}"
    return f"{mode}_{dtype}"


def _iter_conditions(base_cfg: dict):
    sweep = base_cfg.get("sweep", {})
    if "conditions" in sweep and sweep["conditions"]:
        conds = []
        for c in sweep["conditions"]:
            conds.append({
                "name": _condition_name(c),
                "train_mode": c.get("train_mode", base_cfg.get("train_mode", "weights_and_delays")),
                "delay_param_type": c.get("delay_param_type", base_cfg.get("delay_param_type", "sigmoid")),
                "delay_step": c.get("delay_step", base_cfg.get("delay_step", 1.0)),
                "fixed_delay_value": c.get("fixed_delay_value", base_cfg.get("fixed_delay_value", None)),
            })
        return conds

    # legacy fallback
    conds = []
    for mode in sweep.get("train_modes", [base_cfg.get("train_mode", "weights_and_delays")]):
        conds.append({
            "name": mode,
            "train_mode": mode,
            "delay_param_type": base_cfg.get("delay_param_type", "sigmoid"),
            "delay_step": base_cfg.get("delay_step", 1.0),
            "fixed_delay_value": base_cfg.get("fixed_delay_value", None),
        })
    return conds


def run_single(cfg: dict, K: int, condition: dict, device: str, base_runs_dir: str) -> dict:
    set_seed(cfg["seed"])
    logger = setup_logger("step2")

    op = cfg["op_name"]
    h = cfg["hidden_size"]
    cname = condition["name"]
    run_name = f"step2_{op}_{cname}_h{h}_K{K}_seed{cfg['seed']}"
    run_dir = os.path.join(base_runs_dir, run_name)

    logger.info(f"=== {run_name} ===")

    slots = make_slots(K, cfg["win_len"], cfg["read_len"], cfg["gap_len"])

    def make_loader(split):
        seed_off = {"train": 0, "val": 1, "test": 2}[split]
        n = cfg[f"n_{split}"]
        ds = MultiQueryDataset(
            K=K,
            n_samples=n,
            same_op=True,
            op_name=op,
            ops_list=cfg["ops_list"],
            seed=cfg["seed"] + seed_off,
        )
        shuffle = split == "train"
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle)

    train_loader = make_loader("train")
    val_loader = make_loader("val")
    test_loader = make_loader("test")

    model = SNNModel(
        n_input=cfg["n_input"],
        n_hidden=h,
        d_max=cfg["d_max"],
        train_mode=condition["train_mode"],
        delay_param_type=condition["delay_param_type"],
        delay_step=condition.get("delay_step", cfg.get("delay_step", 1.0)),
        fixed_delay_value=condition.get("fixed_delay_value", cfg.get("fixed_delay_value", None)),
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
    trainer.save_config({**cfg, **condition, "K": K})
    log_rows = trainer.fit(train_loader, val_loader, cfg["epochs"])

    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device))
    results = evaluate(model, test_loader, slots, cfg, device)
    results.update({
        "op": op,
        "condition": cname,
        "train_mode": condition["train_mode"],
        "delay_param_type": condition["delay_param_type"],
        "delay_step": condition.get("delay_step", cfg.get("delay_step", 1.0)),
        "fixed_delay_value": condition.get("fixed_delay_value", cfg.get("fixed_delay_value", None)),
        "hidden_size": h,
        "K": K,
    })
    save_eval_results(results, os.path.join(run_dir, "eval_results.json"))

    plot_dir = os.path.join(run_dir, "plots")
    plot_training_curves(log_rows, os.path.join(plot_dir, "training_curves.png"))
    d_ih = model.get_delays()["ih"].detach().cpu().numpy()
    plot_delay_distribution(d_ih, f"Delays ({run_name})", os.path.join(plot_dir, "delays_ih.png"))
    plot_delay_histogram(d_ih, os.path.join(plot_dir, "delays_ih_hist.png"), title=f"Delay Histogram ({run_name})")

    logger.info(
        f"K={K} acc={results['accuracy']:.4f} "
        f"K/spk={results['throughput_K_per_spk']:.4f} "
        f"ops/neuron/ms={results['ops_per_neuron_per_ms']:.6f}"
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/step2_multiquery_sameop.yaml")
    parser.add_argument("--K", default=None, type=int)
    parser.add_argument("--train_mode", default=None)
    parser.add_argument("--hidden_size", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runs_dir", default="runs")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    base_cfg = load_cfg(args.config)
    if args.train_mode is not None:
        base_cfg["train_mode"] = args.train_mode
    if args.hidden_size is not None:
        base_cfg["hidden_size"] = args.hidden_size
    if args.seed is not None:
        base_cfg["seed"] = args.seed
    if args.epochs is not None:
        base_cfg["epochs"] = args.epochs

    if args.sweep:
        sweep = base_cfg["sweep"]
        K_vals = sweep["K_values"]
        tau_list = sweep.get("accuracy_thresholds")
        if tau_list is None:
            tau_list = [sweep.get("accuracy_threshold", 0.95), 0.90]
        tau_list = sorted(set(float(t) for t in tau_list), reverse=True)

        conditions = _iter_conditions(base_cfg)
        all_results: dict = {}

        for cond in conditions:
            cname = cond["name"]
            all_results[cname] = {}
            for h in sweep["hidden_sizes"]:
                for K in K_vals:
                    cfg = copy.deepcopy(base_cfg)
                    cfg["hidden_size"] = h
                    res = run_single(cfg, K, cond, device, args.runs_dir)
                    all_results[cname][K] = res

        summary = {}
        for cname, k_results in all_results.items():
            max_k_by_tau = {str(t): max_K_at_threshold(k_results, t) for t in tau_list}
            summary[cname] = {
                "max_K_by_tau": max_k_by_tau,
                "results_by_K": k_results,
            }

        summary_path = os.path.join(args.runs_dir, "step2_sweep_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSweep summary saved to {summary_path}")

        acc_by_mode = {c: [all_results[c][K]["accuracy"] for K in K_vals] for c in all_results}
        thr_by_mode = {c: [all_results[c][K]["throughput_K_per_spk"] for K in K_vals] for c in all_results}
        dens_by_mode = {c: [all_results[c][K]["ops_per_neuron_per_ms"] for K in K_vals] for c in all_results}

        plots_dir = os.path.join(args.runs_dir, "step2_plots")
        plot_K_accuracy(K_vals, acc_by_mode, tau=tau_list[0], save_path=os.path.join(plots_dir, "K_accuracy.png"))
        plot_throughput(K_vals, thr_by_mode, save_path=os.path.join(plots_dir, "K_throughput.png"))
        plot_metric_triplet(
            K_vals,
            acc_by_mode,
            thr_by_mode,
            dens_by_mode,
            tau_list=tau_list,
            save_path=os.path.join(plots_dir, "K_metric_triplet.png"),
        )
    else:
        K = args.K if args.K is not None else base_cfg["K"]
        cond = {
            "name": _condition_name(base_cfg),
            "train_mode": base_cfg.get("train_mode", "weights_and_delays"),
            "delay_param_type": base_cfg.get("delay_param_type", "sigmoid"),
            "delay_step": base_cfg.get("delay_step", 1.0),
            "fixed_delay_value": base_cfg.get("fixed_delay_value", None),
        }
        run_single(base_cfg, K, cond, device, args.runs_dir)


if __name__ == "__main__":
    main()

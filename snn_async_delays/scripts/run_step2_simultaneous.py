"""
Step 2C — TRUE Temporal Multiplexing (Simultaneous Injection).

Sweeps K (number of simultaneous queries) for each training condition.
Uses SNNSimultaneousModel: n_input=2K dedicated channels, K-output readout,
single trial window T = win_len + read_len (constant across all K values).

Usage:
    # Sweep all K values and conditions
    python -m scripts.run_step2_simultaneous --config configs/step2_simultaneous.yaml

    # Single run
    python -m scripts.run_step2_simultaneous --config configs/step2_simultaneous.yaml \\
        --K 4 --train_mode weights_and_delays

    # GPU
    python -m scripts.run_step2_simultaneous --config configs/step2_simultaneous.yaml \\
        --device cuda
"""

import argparse
import copy
import json
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader

from data.boolean_dataset import MultiQueryDataset
from data.encoding import encode_simultaneous_trial
from snn.model import SNNSimultaneousModel
from train.trainer import SimultaneousTrainer
from train.eval import evaluate_simultaneous, max_K_at_threshold, save_eval_results
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.viz import (
    plot_training_curves, plot_delay_distribution, plot_delay_histogram,
    plot_multilayer_raster, plot_layer_flow,
)


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _condition_name(cond: dict) -> str:
    if cond.get("name"):
        return cond["name"]
    mode  = cond.get("train_mode", "weights_and_delays")
    dtype = cond.get("delay_param_type", "sigmoid")
    return f"{mode}_{dtype}"


def run_single(
    cfg: dict,
    K: int,
    condition: dict,
    device: str,
    base_runs_dir: str,
    run_name_override: str | None = None,
) -> dict:
    set_seed(cfg["seed"])
    logger = setup_logger("step2_simul")

    op    = cfg["op_name"]
    h     = cfg["n_hidden"]
    cname = condition["name"]
    enc_mode = cfg.get("encoding_mode", "rate")
    enc_suffix = f"_{enc_mode}" if enc_mode != "rate" else ""
    run_name = (run_name_override if run_name_override is not None
                else f"step2_simul_{op}_{cname}_h{h}_K{K}_seed{cfg['seed']}{enc_suffix}")
    run_dir = os.path.join(base_runs_dir, run_name)

    # Build encode_fn with burst params pre-bound (enc_mode="rate" → identical to old behavior)
    _encode_fn = partial(
        encode_simultaneous_trial,
        encoding_mode=enc_mode,
        burst_n_spikes_on=cfg.get("burst_n_spikes_on",  2),
        burst_n_spikes_off=cfg.get("burst_n_spikes_off", 1),
        burst_phase_on=cfg.get("burst_phase_on",   0.2),
        burst_phase_off=cfg.get("burst_phase_off",  0.8),
        burst_jitter_ms=cfg.get("burst_jitter_ms",  0),
    )

    # Skip if already completed
    eval_path = os.path.join(run_dir, "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            results = json.load(f)
        logger.info(
            f"=== {run_name} [SKIPPED -- already done, "
            f"acc={results.get('accuracy', '?'):.4f}] ==="
        )
        return results

    logger.info(f"=== {run_name} ===")

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
        return DataLoader(ds, batch_size=cfg["batch_size"],
                          shuffle=(split == "train"))

    train_loader = make_loader("train")
    val_loader   = make_loader("val")
    test_loader  = make_loader("test")

    model = SNNSimultaneousModel(
        n_queries=K,
        n_hidden=h,
        win_len=cfg["win_len"],
        read_len=cfg["read_len"],
        d_max=cfg["d_max"],
        train_mode=condition["train_mode"],
        delay_param_type=condition.get("delay_param_type", cfg.get("delay_param_type", "sigmoid")),
        delay_step=condition.get("delay_step", cfg.get("delay_step", 1.0)),
        fixed_delay_value=condition.get("fixed_delay_value", cfg.get("fixed_delay_value", None)),
        lif_tau_m=cfg["lif_tau_m"],
        lif_threshold=cfg["lif_threshold"],
        lif_reset=cfg["lif_reset"],
        lif_refractory=cfg["lif_refractory"],
        dt=cfg["dt"],
        surrogate_beta=cfg["surrogate_beta"],
    )

    trainer = SimultaneousTrainer(model, cfg, run_dir, device, encode_fn=_encode_fn)
    trainer.save_config({**cfg, **condition, "K": K,
                         "n_input": 2 * K, "experiment": "simultaneous"})
    log_rows = trainer.fit(train_loader, val_loader, cfg["epochs"])

    model.load_state_dict(
        torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device)
    )
    results = evaluate_simultaneous(model, test_loader, cfg, device, encode_fn=_encode_fn)
    results.update({
        "op":               op,
        "condition":        cname,
        "train_mode":       condition["train_mode"],
        "delay_param_type": condition.get("delay_param_type", "sigmoid"),
        "hidden_size":      h,
        "K":                K,
        "experiment":       "simultaneous",
    })
    save_eval_results(results, eval_path)

    plot_dir = os.path.join(run_dir, "plots")
    plot_training_curves(log_rows, os.path.join(plot_dir, "training_curves.png"))
    d_ih = model.get_delays()["ih"].detach().cpu().numpy()
    plot_delay_distribution(
        d_ih, f"Delays ({run_name})",
        os.path.join(plot_dir, "delays_ih.png")
    )
    plot_delay_histogram(
        d_ih, os.path.join(plot_dir, "delays_ih_hist.png"),
        title=f"Delay Histogram ({run_name})"
    )

    logger.info(
        f"K={K}  acc={results['accuracy']:.4f}  "
        f"K/spk={results['throughput_K_per_spk']:.4f}  "
        f"ops/neuron/ms={results['ops_per_neuron_per_ms']:.6f}"
    )

    _plot_spike_activity_simul(
        model=model, cfg=cfg, K=K, op=op, device=device,
        plot_dir=plot_dir, run_name=run_name, encode_fn=_encode_fn,
    )
    return results


@torch.no_grad()
def _plot_spike_activity_simul(model, cfg, K, op, device, plot_dir, run_name,
                                encode_fn=None):
    """Record one simultaneous trial (Plan C) and produce raster + layer-flow."""
    if encode_fn is None:
        encode_fn = encode_simultaneous_trial

    model.eval()
    ds = MultiQueryDataset(K=K, n_samples=1, same_op=True, op_name=op,
                           ops_list=cfg["ops_list"], seed=999)
    A, B, op_ids, labels = ds[0]
    A_b, B_b = A.unsqueeze(0), B.unsqueeze(0)

    spike_input = encode_fn(
        A_b, B_b, win_len=cfg["win_len"], read_len=cfg["read_len"],
        r_on=cfg["r_on"], r_off=cfg["r_off"], dt=cfg["dt"], device=device,
    )
    logits, info = model(spike_input.to(device), record=True)

    s_in  = spike_input[0].cpu().numpy()
    s_hid = info["hidden_spike_train"][0].numpy()
    d_ih  = model.get_delays()["ih"].detach().cpu().numpy()
    w_ih  = model.syn_ih.weight.detach().cpu().numpy()

    label_str = "  ".join(
        f"Q{k}={'1' if labels[k].item() > 0.5 else '0'}" for k in range(K)
    )
    plot_multilayer_raster(
        s_in, s_hid,
        save_path=os.path.join(plot_dir, "spike_raster.png"),
        win_len=cfg["win_len"], read_len=cfg["read_len"],
        title=f"{run_name}\n{label_str}", K=K,
    )
    plot_layer_flow(
        s_in, s_hid,
        save_path=os.path.join(plot_dir, "layer_flow.png"),
        win_len=cfg["win_len"], read_len=cfg["read_len"],
        delays=d_ih, weights=w_ih,
        title=f"Layer flow — {run_name}\n{label_str}", K=K,
        n_arrows=min(12, model.n_input),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/step2_simultaneous.yaml")
    parser.add_argument("--K",          default=None, type=int)
    parser.add_argument("--train_mode", default=None)
    parser.add_argument("--epochs",     default=None, type=int)
    parser.add_argument("--seed",       default=None, type=int)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--runs_dir",   default="runs")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    base_cfg = load_cfg(args.config)
    if args.epochs     is not None: base_cfg["epochs"] = args.epochs
    if args.seed       is not None: base_cfg["seed"]   = args.seed
    if args.train_mode is not None: base_cfg["train_mode"] = args.train_mode

    sweep = base_cfg.get("sweep", {})
    K_vals = sweep.get("K_values", [base_cfg.get("K", 4)])
    tau_list_raw = sweep.get("accuracy_thresholds",
                              [sweep.get("accuracy_threshold", 0.95), 0.90])
    tau_list = sorted(set(float(t) for t in tau_list_raw), reverse=True)

    raw_conds = sweep.get("conditions", [])
    conditions = []
    for c in raw_conds:
        conditions.append({
            "name":             _condition_name(c),
            "train_mode":       c.get("train_mode", base_cfg.get("train_mode", "weights_and_delays")),
            "delay_param_type": c.get("delay_param_type", base_cfg.get("delay_param_type", "sigmoid")),
            "delay_step":       c.get("delay_step", base_cfg.get("delay_step", 1.0)),
            "fixed_delay_value": c.get("fixed_delay_value", base_cfg.get("fixed_delay_value", None)),
        })
    if not conditions:
        conditions = [{
            "name":             "w_and_d_continuous",
            "train_mode":       base_cfg.get("train_mode", "weights_and_delays"),
            "delay_param_type": base_cfg.get("delay_param_type", "sigmoid"),
            "delay_step":       base_cfg.get("delay_step", 1.0),
            "fixed_delay_value": base_cfg.get("fixed_delay_value", None),
        }]

    if args.K is not None:
        # Single run
        K = args.K
        cond = conditions[0]
        if args.train_mode:
            cond = dict(cond, train_mode=args.train_mode,
                        name=args.train_mode)
        run_single(base_cfg, K, cond, device, args.runs_dir)
        return

    # Full sweep
    all_results: dict = {}   # all_results[cname][K]
    for cond in conditions:
        cname = cond["name"]
        all_results[cname] = {}
        for K in K_vals:
            cfg = copy.deepcopy(base_cfg)
            res = run_single(cfg, K, cond, device, args.runs_dir)
            all_results[cname][K] = res

    # Compute max K at each accuracy threshold
    summary = {}
    for cname, k_results in all_results.items():
        max_k_by_tau = {str(t): max_K_at_threshold(k_results, t) for t in tau_list}
        summary[cname] = {
            "max_K_by_tau":  max_k_by_tau,
            "results_by_K":  k_results,
        }

    summary_path = os.path.join(args.runs_dir, "step2_simultaneous_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print result table
    print(f"\n{'='*70}")
    print("Step 2C Simultaneous Sweep Results")
    print(f"{'='*70}")
    print(f"{'condition':30s}  {'K':>4}  {'accuracy':>9}  {'K/spk':>8}")
    print("-" * 60)
    for cname, k_results in all_results.items():
        for K in K_vals:
            r = k_results.get(K, {})
            acc = r.get("accuracy", float("nan"))
            thr = r.get("throughput_K_per_spk", float("nan"))
            print(f"{cname:30s}  {K:4d}  {acc:9.4f}  {thr:8.4f}")
        print()

    print(f"\nMax K at accuracy thresholds:")
    for cname, s in summary.items():
        for tau_s, max_k in s["max_K_by_tau"].items():
            print(f"  {cname:30s}  tau={tau_s}  max_K={max_k}")

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

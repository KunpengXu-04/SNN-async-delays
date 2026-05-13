"""
Step 2D — TRUE Temporal Multiplexing via Sequential Injection (Plan D).

K queries share 2 input channels (A, B). Query k fires during
sub-window [k*sub_win, (k+1)*sub_win). sub_win = win_len // K.

T = win_len + read_len = 30 (FIXED — does not grow with K).
n_input = 2 (not 2K).

With d=0, weights structurally cannot distinguish queries on the
same 2 channels. Only trainable delays allow each neuron to
"tune" its reception delay to a specific sub-window, enabling
genuine temporal multiplexing.

Usage:
    python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml
    python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --device cuda
    python -m scripts.run_step2_sequential --K 3 --train_mode weights_and_delays
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
from data.encoding import encode_sequential_trial
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
    return "{}_{}".format(
        cond.get("train_mode", "weights_and_delays"),
        cond.get("delay_param_type", "sigmoid"),
    )


def run_single(
    cfg: dict,
    K: int,
    condition: dict,
    device: str,
    base_runs_dir: str,
    run_name_override: str | None = None,
) -> dict:
    set_seed(cfg["seed"])
    logger = setup_logger("step2_seq")

    op    = cfg["op_name"]
    h     = cfg["n_hidden"]
    cname = condition["name"]
    sub_win = cfg.get("sub_win", 10)
    win_len = K * sub_win
    d_max   = win_len   # query 0 needs delay ≈ win_len to reach readout window
    run_cfg = {**cfg, "win_len": win_len, "d_max": d_max}

    readout_type = cfg.get("readout_type", "linear")
    rt_suffix = f"_rt{readout_type}" if readout_type != "linear" else ""
    run_name = (run_name_override if run_name_override is not None
                else f"step2_seq_{op}_{cname}_h{h}_K{K}_sw{sub_win}_seed{cfg['seed']}{rt_suffix}")
    run_dir  = os.path.join(base_runs_dir, run_name)

    eval_path = os.path.join(run_dir, "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            results = json.load(f)
        logger.info(
            f"=== {run_name} [SKIPPED -- done, "
            f"acc={results.get('accuracy', '?'):.4f}] ==="
        )
        return results

    logger.info(f"=== {run_name}  (K={K}, sub_win={sub_win} steps) ===")

    def make_loader(split):
        seed_off = {"train": 0, "val": 1, "test": 2}[split]
        n = cfg[f"n_{split}"]
        ds = MultiQueryDataset(
            K=K, n_samples=n, same_op=True, op_name=op,
            ops_list=cfg["ops_list"], seed=cfg["seed"] + seed_off,
        )
        return DataLoader(ds, batch_size=cfg["batch_size"],
                          shuffle=(split == "train"))

    train_loader = make_loader("train")
    val_loader   = make_loader("val")
    test_loader  = make_loader("test")

    # n_input = 2 (shared channels), n_queries = K (K-output readout)
    model = SNNSimultaneousModel(
        n_queries=K,
        n_hidden=h,
        win_len=win_len,
        read_len=run_cfg["read_len"],
        d_max=d_max,
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
        n_input_channels=2,   # <-- Plan D: 2 shared channels (not 2K)
        readout_type=readout_type,
    )

    trainer = SimultaneousTrainer(
        model, run_cfg, run_dir, device,
        encode_fn=encode_sequential_trial,   # <-- Plan D encoding
    )
    trainer.save_config({
        **run_cfg, **condition,
        "K": K, "n_input": 2,
        "sub_win": sub_win,
        "readout_type": readout_type,
        "experiment": "sequential_planD",
    })
    log_rows = trainer.fit(train_loader, val_loader, run_cfg["epochs"])

    model.load_state_dict(
        torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device)
    )
    results = evaluate_simultaneous(
        model, test_loader, run_cfg, device,
        encode_fn=encode_sequential_trial,   # <-- Plan D encoding
    )
    results.update({
        "op":          op,
        "condition":   cname,
        "train_mode":  condition["train_mode"],
        "hidden_size": h,
        "K":           K,
        "sub_win":     sub_win,
        "experiment":  "sequential_planD",
    })
    save_eval_results(results, eval_path)

    plot_dir = os.path.join(run_dir, "plots")
    plot_training_curves(log_rows, os.path.join(plot_dir, "training_curves.png"))
    d_ih = model.get_delays()["ih"].detach().cpu().numpy()
    plot_delay_distribution(
        d_ih, f"Delays ({run_name})",
        os.path.join(plot_dir, "delays_ih.png"),
    )
    plot_delay_histogram(
        d_ih, os.path.join(plot_dir, "delays_ih_hist.png"),
        title=f"Delay Histogram ({run_name})",
    )

    # ── Spike visualization for one example trial ──────────────────
    _plot_spike_activity(
        model=model, cfg=run_cfg, K=K, sub_win=sub_win,
        op=op, device=device, plot_dir=plot_dir, run_name=run_name,
    )

    logger.info(
        f"K={K} sub_win={sub_win}  acc={results['accuracy']:.4f}  "
        f"K/spk={results['throughput_K_per_spk']:.4f}"
    )
    return results


@torch.no_grad()
def _plot_spike_activity(model, cfg, K, sub_win, op, device, plot_dir, run_name):
    """Record a single trial and produce raster + layer-flow plots."""
    from data.boolean_dataset import MultiQueryDataset
    from torch.utils.data import DataLoader

    model.eval()
    ds = MultiQueryDataset(K=K, n_samples=1, same_op=True, op_name=op,
                           ops_list=cfg["ops_list"], seed=999)
    A, B, op_ids, labels = ds[0]
    A_b  = A.unsqueeze(0)
    B_b  = B.unsqueeze(0)
    op_b = op_ids.unsqueeze(0)

    spike_input = encode_sequential_trial(
        A_b, B_b,
        win_len=cfg["win_len"], read_len=cfg["read_len"],
        r_on=cfg["r_on"], r_off=cfg["r_off"],
        dt=cfg["dt"], device=device,
    )

    logits, info = model(spike_input.to(device), record=True)

    # Extract numpy arrays for a single sample
    s_in  = spike_input[0].cpu().numpy()            # [T, 2]
    s_hid = info["hidden_spike_train"][0].numpy()   # [T, n_hidden]
    d_ih  = model.get_delays()["ih"].detach().cpu().numpy()   # [2, n_hidden]
    w_ih  = model.syn_ih.weight.detach().cpu().numpy()         # [2, n_hidden]

    label_str = "  ".join(
        f"Q{k}={'NAND=1' if labels[k].item() > 0.5 else 'NAND=0'}"
        for k in range(K)
    )
    title_raster = f"{run_name}\n{label_str}"
    title_flow   = f"Layer flow — {run_name}\n{label_str}"

    plot_multilayer_raster(
        s_in, s_hid,
        save_path=os.path.join(plot_dir, "spike_raster.png"),
        win_len=cfg["win_len"], read_len=cfg["read_len"],
        title=title_raster, K=K, sub_win=sub_win,
    )
    plot_layer_flow(
        s_in, s_hid,
        save_path=os.path.join(plot_dir, "layer_flow.png"),
        win_len=cfg["win_len"], read_len=cfg["read_len"],
        delays=d_ih, weights=w_ih,
        title=title_flow, K=K, sub_win=sub_win,
        n_arrows=min(12, 2 * model.n_hidden),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/step2_sequential.yaml")
    parser.add_argument("--K",          default=None, type=int)
    parser.add_argument("--train_mode", default=None)
    parser.add_argument("--epochs",     default=None, type=int)
    parser.add_argument("--seed",       default=None, type=int)
    parser.add_argument("--n_hidden",     default=None, type=int)
    parser.add_argument("--readout_type", default=None)
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--runs_dir",     default="runs")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    base_cfg = load_cfg(args.config)
    if args.epochs     is not None: base_cfg["epochs"]   = args.epochs
    if args.seed       is not None: base_cfg["seed"]     = args.seed
    if args.n_hidden     is not None: base_cfg["n_hidden"]     = args.n_hidden
    if args.readout_type is not None: base_cfg["readout_type"] = args.readout_type
    if args.train_mode   is not None: base_cfg["train_mode"]   = args.train_mode

    sweep = base_cfg.get("sweep", {})
    K_vals   = sweep.get("K_values", [base_cfg.get("K", 2)])
    tau_list = sorted(
        set(float(t) for t in sweep.get("accuracy_thresholds", [0.95, 0.90])),
        reverse=True,
    )

    raw_conds = sweep.get("conditions", [])
    conditions = [{
        "name":              _condition_name(c),
        "train_mode":        c.get("train_mode",       base_cfg.get("train_mode", "weights_and_delays")),
        "delay_param_type":  c.get("delay_param_type", base_cfg.get("delay_param_type", "sigmoid")),
        "delay_step":        c.get("delay_step",       base_cfg.get("delay_step", 1.0)),
        "fixed_delay_value": c.get("fixed_delay_value",base_cfg.get("fixed_delay_value", None)),
    } for c in raw_conds] or [{
        "name": "w_and_d_continuous", "train_mode": "weights_and_delays",
        "delay_param_type": "sigmoid", "delay_step": 1.0, "fixed_delay_value": None,
    }]

    if args.K is not None:
        cond = conditions[0]
        if args.train_mode:
            cond = dict(cond, train_mode=args.train_mode, name=args.train_mode)
        run_single(base_cfg, args.K, cond, device, args.runs_dir)
        return

    all_results: dict = {}
    for cond in conditions:
        cname = cond["name"]
        all_results[cname] = {}
        for K in K_vals:
            cfg = copy.deepcopy(base_cfg)
            res = run_single(cfg, K, cond, device, args.runs_dir)
            all_results[cname][K] = res

    summary = {
        cname: {
            "max_K_by_tau":  {str(t): max_K_at_threshold(k_res, t) for t in tau_list},
            "results_by_K":  k_res,
        }
        for cname, k_res in all_results.items()
    }

    summary_path = os.path.join(args.runs_dir, "step2_sequential_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Result table ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("Plan D — Sequential Injection (2 shared channels)")
    print(f"{'='*72}")
    print("{:>4}  {:>8}  {:>12}  {:>12}  {:>6}".format(
        "K", "sub_win", "w_and_d", "w_only_d0", "delta"))
    print("-" * 55)
    wd  = all_results.get("w_and_d_continuous",  {})
    wo  = all_results.get("weights_only_d0", {})
    for K in K_vals:
        sw  = base_cfg.get("sub_win", 10)
        a   = wd.get(K, {}).get("accuracy", float("nan"))
        c   = wo.get(K, {}).get("accuracy", float("nan"))
        print("{:>4}  {:>8}  {:>12.4f}  {:>12.4f}  {:>+6.4f}".format(
            K, sw, a, c, a - c))

    print("\nMax K at accuracy thresholds:")
    for cname, s in summary.items():
        for tau_s, mk in s["max_K_by_tau"].items():
            print("  {:30s}  tau={:4s}  max_K={:>2}".format(cname, tau_s, str(mk)))

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

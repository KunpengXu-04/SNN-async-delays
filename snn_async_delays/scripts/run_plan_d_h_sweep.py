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
from functools import partial

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
    save_run_diagnostic_plots,
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
    "homeo_lambda": 0.0,      # homeostatic firing-rate reg strength (0 = off)
    "homeo_target": 0.005,    # target per-neuron rate = target rho (spikes/neuron/step);
                              # 0.005 ~ the sparse regime (rho of thr=0.5, well below rate)
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

# Burst params proven in Section 29 (value=1 -> 2 spikes @ phase 0.20; value=0 -> 1 spike @ 0.80).
# Ignored when encoding_mode == "rate".
BURST_PARAMS = {
    "burst_n_spikes_on":  2,
    "burst_n_spikes_off": 1,
    "burst_phase_on":     0.20,
    "burst_phase_off":    0.80,
    "burst_jitter_ms":    0,
}

DEFAULT_H_VALUES = [10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200]
DEFAULT_K_VALUES = [1, 2, 3, 4, 5, 6, 8]
DEFAULT_SEEDS    = [42, 0]


def build_encode_fn(base_encode, cfg):
    """Wrap the sequential encoder with the encoding_mode + burst params from cfg."""
    return partial(
        base_encode,
        encoding_mode     = cfg.get("encoding_mode", "rate"),
        burst_n_spikes_on = cfg.get("burst_n_spikes_on",  BURST_PARAMS["burst_n_spikes_on"]),
        burst_n_spikes_off= cfg.get("burst_n_spikes_off", BURST_PARAMS["burst_n_spikes_off"]),
        burst_phase_on    = cfg.get("burst_phase_on",     BURST_PARAMS["burst_phase_on"]),
        burst_phase_off   = cfg.get("burst_phase_off",    BURST_PARAMS["burst_phase_off"]),
        burst_jitter_ms   = cfg.get("burst_jitter_ms",    BURST_PARAMS["burst_jitter_ms"]),
    )


def run_single(cfg, K, h, condition, seed, device, runs_dir, dry_run=False):
    sub_win  = cfg["sub_win"]
    win_len  = K * sub_win
    d_max    = int(cfg.get("d_max", win_len))
    cname    = condition["name"]
    # Match the 2026-06 rename convention (wad/d0) so [SKIP] reuses the existing
    # runs/NAND_neuron_sweep_(planD) rate cells instead of retraining them.
    cname_short = {"w_and_d": "wad"}.get(cname, cname)
    # NAND keeps its bare name (backward-compatible reuse); other ops get an op
    # prefix so different ops can share a folder without run-name collisions.
    op_tag = "" if cfg["op_name"] == "NAND" else f"{cfg['op_name']}_"
    # tag non-default threshold + homeo so tuning sweeps can share one folder
    thr = cfg.get("lif_threshold", 1.0)
    thr_tag = "" if abs(thr - 1.0) < 1e-9 else f"_thr{thr}"
    hl = cfg.get("homeo_lambda", 0.0)
    homeo_tag = "" if not hl else f"_hl{hl}t{cfg.get('homeo_target', 0.005)}"
    observation_mode = cfg.get("observation_mode", "late_window")
    observation_tag = "" if observation_mode == "late_window" else f"_obs{observation_mode}"
    readout_type = cfg.get("readout_type", "mlp")
    readout_tag = "" if readout_type == "mlp" else f"_ro{readout_type}"
    run_name = (
        f"{op_tag}{cname_short}_h{h}_K{K}_sw{sub_win}_seed{seed}"
        f"{thr_tag}{homeo_tag}{observation_tag}{readout_tag}"
    )
    run_dir  = os.path.join(runs_dir, run_name)
    evaluation_split = cfg.get("evaluation_split", "test")
    if evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation_split must be 'val' or 'test'")
    result_filename = cfg.get(
        "result_filename",
        "eval_results.json" if evaluation_split == "test" else "validation_results.json",
    )
    eval_path = os.path.join(run_dir, result_filename)

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
        fixed_delay_distribution=condition.get("fixed_delay_distribution"),
        fixed_delay_seed=condition.get("fixed_delay_seed", seed),
        fixed_delay_low=condition.get("fixed_delay_low", 0.0),
        fixed_delay_high=condition.get("fixed_delay_high", None),
        shared_delay=condition.get("shared_delay", False),
        delay_init_mode=cfg.get("delay_init_mode", "constant"),
        delay_init_raw=cfg.get("delay_init_raw", -2.0),
        delay_init_std=cfg.get("delay_init_std", 0.25),
        lif_tau_m        = run_cfg["lif_tau_m"],
        lif_threshold    = run_cfg["lif_threshold"],
        lif_reset        = run_cfg["lif_reset"],
        lif_refractory   = run_cfg["lif_refractory"],
        dt               = run_cfg["dt"],
        surrogate_beta   = run_cfg["surrogate_beta"],
        n_input_channels = 2,
        readout_type     = run_cfg["readout_type"],
        observation_mode = run_cfg.get("observation_mode", "late_window"),
    )

    encode_fn = build_encode_fn(encode_sequential_trial, run_cfg)
    trainer = SimultaneousTrainer(
        model, run_cfg, run_dir, device,
        encode_fn=encode_fn,
    )
    trainer.save_config({
        **run_cfg, **condition,
        "K": K, "n_input": 2, "sub_win": sub_win,
    })
    log_rows = trainer.fit(
        make_loader("train"), make_loader("val"), run_cfg["epochs"]
    )
    model.load_state_dict(
        torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device,
                   weights_only=True)
    )
    results = evaluate_simultaneous(
        model, make_loader(evaluation_split), run_cfg, device,
        encode_fn=encode_fn,
    )
    results.update({
        "op": run_cfg["op_name"], "condition": cname,
        "train_mode": condition["train_mode"],
        "hidden_size": h, "K": K, "sub_win": sub_win,
        "encoding_mode": run_cfg.get("encoding_mode", "rate"),
        "experiment": run_cfg.get("experiment", "planD_h_sweep"),
        "evaluation_split": evaluation_split,
    })
    save_eval_results(results, eval_path)

    plot_dir = os.path.join(run_dir, "plots")
    plot_training_curves(log_rows, os.path.join(plot_dir, "training_curves.png"))
    d_ih = model.get_delays()["ih"].detach().cpu().numpy()
    plot_delay_distribution(d_ih, run_name, os.path.join(plot_dir, "delays_ih.png"))
    plot_delay_histogram(d_ih, os.path.join(plot_dir, "delays_ih_hist.png"),
                         title=f"Delays {run_name}")

    # Full per-run diagnostic set (panel / spike flow / weight & delay heatmaps /
    # diagnostic_data.npz), encoding-aware via cfg["encoding_mode"]. Non-fatal so a
    # plotting error never aborts a long sweep.
    if not cfg.get("no_diag", False):
        try:
            save_run_diagnostic_plots(
                model=model,
                cfg={**run_cfg, **condition, "K": K, "n_input": 2, "sub_win": sub_win},
                log_rows=log_rows, eval_results=results, run_dir=run_dir,
                K=K, op=run_cfg["op_name"], device=device, seed=999,
            )
            # Mechanism figure (input->hidden d_ih routing) replaces the dense
            # weight-fan spike flow for these MLP-readout cells.
            from scripts.plot_burst_mechanism import plot_run_mechanism
            plot_run_mechanism(run_dir)
            flow = os.path.join(plot_dir, "layer_to_layer_spike_flow_sample0.png")
            if os.path.exists(flow):
                os.remove(flow)
        except Exception as exc:
            logger.warning(f"  diagnostic plots failed: {exc}")

    logger.info(f"  acc={results['accuracy']:.4f}  K/spk={results['throughput_K_per_spk']:.4f}")
    return results


def aggregate_summary(results_by_h_K, K_values, h_values, tau_values=(0.95, 0.90),
                      conditions=None):
    """
    Build summary JSON from collected results.

    results_by_h_K: { (h, K, cond, seed): result_dict }
    conditions: list of condition dicts to summarise (default: all CONDITIONS).
    """
    summary = {}
    for cond in (conditions or CONDITIONS):
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
                        v["worst_query_accuracy"] for (hh, kk, cc, ss), v
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
    parser.add_argument("--runs_dir",  default=None,
                        help="Output folder under runs/ (default: planD_h_sweep). "
                             "Absolute paths are used as-is. Point at an existing folder "
                             "to reuse its completed cells via [SKIP].")
    parser.add_argument("--encoding_mode", default="rate",
                        choices=["rate", "burst", "burst_jitter"],
                        help="Input encoding for the sweep (default: rate).")
    parser.add_argument(
        "--observation_mode", default="late_window",
        choices=["late_window", "all_time", "time_binned"],
        help="Explicit hidden-spike observation interface.",
    )
    parser.add_argument("--no_diag", action="store_true",
                        help="Skip the per-run diagnostic plot set (faster; keeps "
                             "only training_curves + delay plots).")
    parser.add_argument("--op", default=None,
                        help="Single boolean op for same_op runs (default: NAND). "
                             "Must be in ops_list, e.g. XOR, XNOR, AND.")
    parser.add_argument("--conditions", nargs="+", default=["w_and_d", "d0"],
                        choices=["w_and_d", "d0"],
                        help="Which training conditions to run (default: both).")
    parser.add_argument("--d0_h", type=int, nargs="+", default=None,
                        help="Sparse hidden sizes used ONLY for the d0 control "
                             "(default: same as --h_values). d0 never crosses the "
                             "threshold, so a few h suffice as the flat-line control.")
    parser.add_argument("--lif_threshold", type=float, nargs="+", default=None,
                        help="Hidden LIF firing threshold(s), default 1.0. Lower it "
                             "(e.g. 0.3) for sparse burst so the net fires at init and "
                             "low-K can train. Pass MULTIPLE values (e.g. 0.3 0.4 0.5) "
                             "to run a tuning sweep in one folder (threshold tagged in "
                             "the run name; per-cell evals record spk/tr for the "
                             "accuracy-vs-sparsity tradeoff). Use the SAME single value "
                             "for the NAND comparison run.")
    parser.add_argument("--homeo_lambda", type=float, default=None,
                        help="Homeostatic firing-rate reg strength (default 0 = off). "
                             "Pulls each hidden neuron's rate toward --homeo_target so "
                             "the net can fire+train at a HIGH (sparse) threshold "
                             "without dying. Try 0.5-5.0.")
    parser.add_argument("--homeo_target", type=float, default=None,
                        help="Target per-neuron firing rate (spikes/neuron/step) for "
                             "the homeostatic reg (default 0.02).")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.runs_dir is None:
        runs_dir = os.path.join(base, "runs", "planD_h_sweep")
    elif os.path.isabs(args.runs_dir):
        runs_dir = args.runs_dir
    else:
        runs_dir = os.path.join(base, args.runs_dir) \
            if args.runs_dir.startswith("runs") \
            else os.path.join(base, "runs", args.runs_dir)
    os.makedirs(runs_dir, exist_ok=True)

    cfg = dict(BASE_CFG)
    cfg["encoding_mode"] = args.encoding_mode
    cfg["observation_mode"] = args.observation_mode
    cfg["no_diag"] = args.no_diag
    thr_list = args.lif_threshold if args.lif_threshold else [BASE_CFG["lif_threshold"]]
    multi_thr = len(thr_list) > 1
    if args.homeo_lambda is not None:
        cfg["homeo_lambda"] = args.homeo_lambda
    if args.homeo_target is not None:
        cfg["homeo_target"] = args.homeo_target
    if args.op:
        if args.op not in cfg["ops_list"]:
            parser.error(f"--op {args.op} not in ops_list {cfg['ops_list']}")
        cfg["op_name"] = args.op
    if args.encoding_mode in ("burst", "burst_jitter"):
        cfg.update(BURST_PARAMS)
        if args.encoding_mode == "burst_jitter":
            cfg["burst_jitter_ms"] = 1
    if args.epochs:
        cfg["epochs"] = args.epochs

    sel_conds = [c for c in CONDITIONS if c["name"] in args.conditions]

    def h_for(cname):
        return args.d0_h if (cname == "d0" and args.d0_h) else args.h_values

    all_h = sorted({h for c in sel_conds for h in h_for(c["name"])})
    total = sum(len(h_for(c["name"])) for c in sel_conds) \
        * len(args.K_values) * len(args.seeds) * len(thr_list)

    print(f"Plan D h-sweep: {total} runs")
    print(f"  op       = {cfg['op_name']}")
    print(f"  h        = {args.h_values}")
    if args.d0_h:
        print(f"  d0_h     = {args.d0_h}  (d0 control only)")
    print(f"  K        = {args.K_values}")
    print(f"  conds    = {[c['name'] for c in sel_conds]}")
    print(f"  seeds    = {args.seeds}")
    print(f"  thresh   = {thr_list}" + ("  (tuning sweep; summary skipped)" if multi_thr else ""))
    print(f"  encoding = {args.encoding_mode}")
    print(f"  observe  = {args.observation_mode}")
    print(f"  runs_dir = {runs_dir}")
    print(f"  device   = {args.device}")
    print()

    results_by_h_K = {}
    run_idx = 0
    for cond in sel_conds:
        for h in h_for(cond["name"]):
            for K in args.K_values:
                for thr in thr_list:
                    cfg["lif_threshold"] = thr
                    for seed in args.seeds:
                        run_idx += 1
                        print(f"[{run_idx}/{total}] op={cfg['op_name']} h={h} K={K} "
                              f"cond={cond['name']} thr={thr} seed={seed}")
                        r = run_single(cfg, K, h, cond, seed, args.device,
                                       runs_dir, dry_run=args.dry_run)
                        if not multi_thr:
                            results_by_h_K[(h, K, cond["name"], seed)] = r

    if multi_thr:
        print("\nMulti-threshold tuning sweep: per-run evals + configs saved "
              "(threshold in run name & config.json). Min-h summary skipped "
              "(mixes thresholds). Compare thresholds from the per-run evals.")
    elif not args.dry_run:
        summary = aggregate_summary(
            results_by_h_K, args.K_values, all_h, conditions=sel_conds
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
            h_wd_str = str(h_wd) if h_wd is not None else f">{max(all_h)}"
            h_d0_str = str(h_d0) if h_d0 is not None else f">{max(all_h)}"
            print(f"{K:>4}  {h_wd_str:>10}  {h_d0_str:>10}")

        print(f"\nNext step: python -m scripts.plot_k_vs_neurons --summary {summary_path}")


if __name__ == "__main__":
    main()

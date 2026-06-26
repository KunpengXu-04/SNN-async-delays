"""
Step 4 — Topologies 2 & 3: many-query, one-out / many-query, many-op, one-out.

Keeps the existing Plan D (SC) input structure EXACTLY -- K queries
injected sequentially into 2 (same-op) or 2+n_ops (mixed-op) shared
channels -- but collapses the K-logit readout into a SINGLE aggregate
logit (majority vote, or AND-of-results as a secondary stress test).

config `same_op: true`  -> Topology 2 (many-query, one-out, single op)
config `same_op: false` -> Topology 3 (many-query, many-op, one-out)

Reuses SNNSimultaneousModel / encode_sequential_trial / MultiQueryDataset
(with its new `aggregate=` param) unchanged: win_len = K * sub_win exactly
as Plan D, n_queries=1 (single aggregate head). Because model.n_queries no
longer equals the true multiplexing load K, evaluate_simultaneous is called
with K_query=K explicitly, and diagnostic plots are requested with K=K
(not model.n_queries) so the enhanced-flow figure correctly shows K
colored sub-windows funneling into the one readout vote line.

Usage:
    python -m scripts.run_step4_many_query_one_out \\
        --config configs/step4_many_query_one_out.yaml --device cuda
    python -m scripts.run_step4_many_query_one_out \\
        --config configs/step4_many_many_one_out.yaml --device cuda
    python -m scripts.run_step4_many_query_one_out \\
        --config configs/step4_many_query_one_out.yaml \\
        --K 3 --model wad_mlp --seed 42 --device cuda
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
from train.eval import evaluate_simultaneous, save_eval_results
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.viz import save_run_diagnostic_plots


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _balanced_metrics(binary_confusion) -> dict:
    """Balanced accuracy / F1 / trivial-baseline accuracy from a [[tn,fp],[fn,tp]]
    confusion matrix -- needed because aggregate='and' label positivity shrinks
    as 0.5^K, and raw accuracy alone can hide a trivial always-predict-0 model.
    """
    (tn, fp), (fn, tp) = binary_confusion
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    balanced_acc = (tpr + tnr) / 2
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    if precision == precision and tpr == tpr and (precision + tpr) > 0:
        f1 = 2 * precision * tpr / (precision + tpr)
    else:
        f1 = float("nan")
    total = tn + fp + fn + tp
    trivial_baseline = (tn + fp) / total if total > 0 else float("nan")
    return {
        "balanced_accuracy": balanced_acc,
        "f1": f1,
        "trivial_baseline_acc": trivial_baseline,
    }


def run_single(
    base_cfg: dict,
    model_cfg: dict,
    K: int,
    seed: int,
    device: str,
    runs_dir: str,
    aggregate: str,
    extra_tag: str = "",
) -> dict:
    """Train and evaluate one (model_variant, K, seed, aggregate) combination."""
    set_seed(seed)
    logger = setup_logger("step4_many_query_one_out")

    same_op   = base_cfg.get("same_op", True)
    ops_list  = base_cfg["ops_list"]
    n_ops     = base_cfg.get("n_ops", 0) if not same_op else 0
    op_name   = base_cfg.get("op_name") if same_op else None
    sub_win   = base_cfg.get("sub_win", 10)
    win_len   = K * sub_win
    d_max     = win_len
    mname     = model_cfg["name"]
    n_input   = 2 + n_ops

    run_cfg = {
        **base_cfg,
        "win_len": win_len,
        "d_max":   d_max,
        "seed":    seed,
        "n_ops":   n_ops,
    }

    run_name  = f"{mname}_K{K}_seed{seed}{extra_tag}"
    run_dir   = os.path.join(runs_dir, run_name)
    eval_path = os.path.join(run_dir, "eval_results.json")

    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            results = json.load(f)
        logger.info(
            f"=== {run_name} [SKIPPED — done, "
            f"acc={results.get('accuracy', '?'):.4f}] ==="
        )
        return results

    logger.info(
        f"=== {run_name}  (K={K}, same_op={same_op}, aggregate={aggregate}, "
        f"sub_win={sub_win}, n_input={n_input}) ==="
    )

    def make_loader(split):
        seed_off = {"train": 0, "val": 1, "test": 2}[split]
        n = run_cfg[f"n_{split}"]
        ds = MultiQueryDataset(
            K=K, n_samples=n, same_op=same_op, op_name=op_name,
            ops_list=ops_list, seed=seed + seed_off,
            op_sampling=run_cfg.get("op_sampling", "uniform"),
            aggregate=aggregate,
        )
        return DataLoader(ds, batch_size=run_cfg["batch_size"],
                          shuffle=(split == "train"))

    train_loader = make_loader("train")
    val_loader   = make_loader("val")
    test_loader  = make_loader("test")

    num_layers   = model_cfg.get("num_hidden_layers", 1)
    hidden_sizes = model_cfg.get("hidden_sizes", [run_cfg.get("n_hidden", 50)])
    readout_type = model_cfg.get("readout_type", "mlp")
    train_mode   = model_cfg.get("train_mode", "weights_and_delays")
    fixed_dv     = model_cfg.get("fixed_delay_value", None)

    model = SNNSimultaneousModel(
        n_queries         = 1,   # single aggregate logit -- decoupled from K
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
        n_input_channels  = n_input,
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
        "K": K, "n_input": n_input,
        "sub_win": sub_win,
        "same_op": same_op,
        "aggregate": aggregate,
        "num_hidden_layers": num_layers,
        "hidden_sizes": hidden_sizes,
        "readout_type": readout_type,
        "experiment": "step4_many_query_one_out" if same_op else "step4_many_many_one_out",
    })
    log_rows = trainer.fit(train_loader, val_loader, run_cfg["epochs"])

    model.load_state_dict(
        torch.load(os.path.join(run_dir, "best_model.pt"),
                   map_location=device, weights_only=True)
    )
    # model.n_queries == 1, but the true multiplexing load is K (sub-windows) --
    # must override K_query for correct throughput / ops_per_neuron accounting.
    results = evaluate_simultaneous(
        model, test_loader, run_cfg, device,
        encode_fn=encode_sequential_trial,
        K_query=K,
    )
    results.update(_balanced_metrics(results["binary_confusion"]))

    # Delay statistics
    delays = model.get_delays()
    d_ih = delays["ih"].detach()
    results["delay_ih_mean"]   = float(d_ih.mean())
    results["delay_ih_std"]    = float(d_ih.std())
    results["delay_ih_median"] = float(d_ih.median())

    results.update({
        "model_name":        mname,
        "num_hidden_layers": num_layers,
        "hidden_sizes":      hidden_sizes,
        "readout_type":      readout_type,
        "train_mode":        train_mode,
        "same_op":           same_op,
        "aggregate":         aggregate,
        "n_ops":             n_ops,
        "n_input":           n_input,
        "K":                 K,
        "seed":              seed,
        "sub_win":           sub_win,
        "win_len":           win_len,
        "d_max":             d_max,
        "run_dir":           run_dir,
        "experiment":        "step4_many_query_one_out" if same_op else "step4_many_many_one_out",
    })
    save_eval_results(results, eval_path)

    save_run_diagnostic_plots(
        model=model,
        cfg={**run_cfg, "model_name": mname,
             "hidden_sizes": hidden_sizes,
             "readout_type": readout_type,
             "num_hidden_layers": num_layers},
        log_rows=log_rows,
        eval_results=results,
        run_dir=run_dir,
        K=K,   # true sub-window count, NOT model.n_queries (=1)
        op=(op_name if same_op else "mixed"),
        device=device,
        seed=999,
    )

    logger.info(
        f"K={K}  model={mname}  aggregate={aggregate}  acc={results['accuracy']:.4f}  "
        f"bal_acc={results['balanced_accuracy']:.4f}  "
        f"K/spk={results['throughput_K_per_spk']:.4f}  "
        f"delay_ih_mean={results['delay_ih_mean']:.2f}"
    )
    return results


def _write_summary_csv(all_results: list, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = [
        "model_name", "train_mode", "readout_type", "aggregate", "K", "seed",
        "accuracy", "balanced_accuracy", "f1", "trivial_baseline_acc",
        "mean_hidden_spikes", "throughput_K_per_spk",
        "mean_active_hidden_fraction",
        "delay_ih_mean", "delay_ih_median", "delay_ih_std",
        "same_op", "n_ops", "n_input", "sub_win", "win_len", "d_max",
        "run_dir",
    ]
    rows = []
    for r in all_results:
        row = {c: r.get(c) for c in cols}
        row["hidden_sizes"] = str(r.get("hidden_sizes", ""))
        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/step4_many_query_one_out.yaml")
    parser.add_argument("--K",        default=None, type=int)
    parser.add_argument("--model",    default=None, help="Single model name override")
    parser.add_argument("--seed",     default=None, type=int)
    parser.add_argument("--epochs",   default=None, type=int)
    parser.add_argument("--device",   default="auto")
    parser.add_argument("--runs_dir", default=None)
    parser.add_argument("--skip_secondary", action="store_true",
                        help="Skip the secondary_aggregate stress-test sweep, if configured")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    base     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_cfg = load_cfg(args.config)
    if args.epochs is not None:
        base_cfg["epochs"] = args.epochs

    same_op = base_cfg.get("same_op", True)
    tag = "many_query_one_out" if same_op else "many_many_one_out"
    runs_dir = args.runs_dir or os.path.join(base, "runs", f"{tag}_(step4)")

    primary_aggregate = base_cfg.get("aggregate", "majority")

    sweep      = base_cfg.get("sweep", {})
    K_vals     = [args.K]    if args.K    is not None else sweep.get("K_values", [1, 3, 5])
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

    all_results = []
    for model_cfg in model_cfgs:
        for K in K_vals:
            for seed in seeds:
                cfg = copy.deepcopy(base_cfg)
                res = run_single(cfg, model_cfg, K, seed, device, runs_dir,
                                 aggregate=primary_aggregate)
                all_results.append(res)

    secondary = base_cfg.get("secondary_aggregate")
    if secondary and not args.skip_secondary and args.K is None:
        sec_value = secondary["value"]
        sec_Ks    = secondary.get("K_values", [])
        for model_cfg in model_cfgs:
            for K in sec_Ks:
                for seed in seeds:
                    cfg = copy.deepcopy(base_cfg)
                    res = run_single(cfg, model_cfg, K, seed, device, runs_dir,
                                     aggregate=sec_value, extra_tag=f"_{sec_value}")
                    all_results.append(res)

    if not all_results:
        print("No runs completed.")
        return

    csv_path  = os.path.join(runs_dir, f"step4_{tag}_summary.csv")
    json_path = os.path.join(runs_dir, f"step4_{tag}_summary.json")

    _write_summary_csv(all_results, csv_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*72}")
    print(f"Step 4 Summary — {'Many-Query, One-Out' if same_op else 'Many-Query, Many-Op, One-Out'}")
    print(f"{'='*72}")
    print(f"{'Model':16s}  {'Agg':9s}  {'K':>3}  {'Acc (mean±range)':>20}  {'BalAcc':>8}  {'K/spk':>8}")
    print("-" * 80)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r.get("model_name", "?"), r.get("aggregate", "?"), r.get("K", 0))].append(r)

    for (mname, agg, K), runs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        accs = [r["accuracy"] for r in runs if r.get("accuracy") is not None]
        bals = [r["balanced_accuracy"] for r in runs if r.get("balanced_accuracy") is not None]
        spks = [r["throughput_K_per_spk"] for r in runs
                if r.get("throughput_K_per_spk") is not None
                and r["throughput_K_per_spk"] == r["throughput_K_per_spk"]]
        acc_str = (f"{sum(accs)/len(accs)*100:.1f}% "
                   f"± {(max(accs)-min(accs))*100:.1f}%") if accs else "n/a"
        bal_str = f"{sum(bals)/len(bals)*100:.1f}%" if bals else "n/a"
        spk_str = f"{sum(spks)/len(spks):.3f}" if spks else "n/a"
        print(f"{mname:16s}  {agg:9s}  {K:>3}  {acc_str:>20}  {bal_str:>8}  {spk_str:>8}")

    print()
    models_all = {(r.get("model_name"), r.get("aggregate")) for r in all_results}
    for mname, agg in sorted(models_all):
        m_runs = [r for r in all_results
                  if r.get("model_name") == mname and r.get("aggregate") == agg]
        from collections import defaultdict as dd
        k_accs = dd(list)
        for r in m_runs:
            k_accs[r["K"]].append(r["accuracy"])
        k_mean = {K: sum(v)/len(v) for K, v in k_accs.items()}
        for tau in tau_list:
            mk = max((K for K, acc in k_mean.items() if acc >= tau), default=0)
            print(f"  {mname:16s}  agg={agg:9s}  tau={tau:.0%}  max_K={mk}")

    print(f"\nSummary CSV  : {csv_path}")
    print(f"Summary JSON : {json_path}")


if __name__ == "__main__":
    main()

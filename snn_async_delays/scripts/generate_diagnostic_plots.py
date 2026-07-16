"""
Retroactively generate diagnostic plots for completed runs that lack them.

Scans a runs directory, finds runs with eval_results.json + best_model.pt
but without plots/diagnostic_panel.png, then loads each model and generates
the full diagnostic plot suite.

Works with both 1-layer and 2-layer SNNSimultaneousModel runs.

Usage (from snn_async_delays/):
    # depth ablation runs (default)
    python -m scripts.generate_diagnostic_plots

    # any runs directory
    python -m scripts.generate_diagnostic_plots --runs_dir runs/step2_planD

    # dry run - list what would be generated
    python -m scripts.generate_diagnostic_plots --dry_run

    # force regenerate even if plots already exist
    python -m scripts.generate_diagnostic_plots --force
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from snn.model import SNNSimultaneousModel
from utils.viz import save_run_diagnostic_plots


def _load_model_from_run(run_dir: str, device: str):
    """Load model and config from a completed run directory."""
    cfg_path  = os.path.join(run_dir, "config.json")
    ckpt_path = os.path.join(run_dir, "best_model.pt")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    num_layers   = cfg.get("num_hidden_layers", 1)
    hidden_sizes = cfg.get("hidden_sizes")
    if hidden_sizes is None:
        hidden_sizes = [cfg.get("n_hidden", cfg.get("hidden_size", 50))]
    if isinstance(hidden_sizes, str):
        import ast
        hidden_sizes = ast.literal_eval(hidden_sizes)

    model = SNNSimultaneousModel(
        n_queries         = cfg["K"],
        n_hidden          = hidden_sizes[0],
        win_len           = cfg["win_len"],
        read_len          = cfg["read_len"],
        d_max             = cfg["d_max"],
        train_mode        = cfg["train_mode"],
        delay_param_type  = cfg.get("delay_param_type", "sigmoid"),
        delay_step        = cfg.get("delay_step", 1.0),
        fixed_delay_value = cfg.get("fixed_delay_value", None),
        fixed_delay_distribution = cfg.get("fixed_delay_distribution", None),
        fixed_delay_seed = cfg.get("fixed_delay_seed", 0),
        fixed_delay_low = cfg.get("fixed_delay_low", 0.0),
        fixed_delay_high = cfg.get("fixed_delay_high", None),
        shared_delay = cfg.get("shared_delay", False),
        delay_init_mode = cfg.get("delay_init_mode", "constant"),
        delay_init_raw = cfg.get("delay_init_raw", -2.0),
        delay_init_std = cfg.get("delay_init_std", 0.25),
        lif_tau_m         = cfg["lif_tau_m"],
        lif_threshold     = cfg["lif_threshold"],
        lif_reset         = cfg["lif_reset"],
        lif_refractory    = cfg["lif_refractory"],
        dt                = cfg["dt"],
        surrogate_beta    = cfg["surrogate_beta"],
        n_input_channels  = cfg.get("n_input", 2),
        readout_type      = cfg.get("readout_type", "linear"),
        num_hidden_layers = num_layers,
        hidden_sizes      = hidden_sizes,
        use_output_spikes = cfg.get("use_output_spikes", False),
        n_output_neurons  = cfg.get("n_output_neurons", None),
        lif_output_threshold = cfg.get("lif_output_threshold", None),
        observation_mode  = cfg.get("observation_mode", "late_window"),
        opponent_output_mode = cfg.get("opponent_output_mode", None),
        output_window_len = cfg.get("output_window_len", None),
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, cfg


def _load_log_rows(run_dir: str) -> list:
    import csv
    log_path = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(log_path):
        return []
    rows = []
    with open(log_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k != "epoch" else int(v)
                         for k, v in row.items()})
    return rows


def discover_runs(runs_dir: str, force: bool = False) -> list[dict]:
    """Find completed runs that need diagnostic plots."""
    runs = []
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        if not os.path.exists(os.path.join(run_dir, "eval_results.json")):
            continue
        if not os.path.exists(os.path.join(run_dir, "best_model.pt")):
            continue
        if not os.path.exists(os.path.join(run_dir, "config.json")):
            continue

        panel_path = os.path.join(run_dir, "plots", "diagnostic_panel.png")
        if not force and os.path.exists(panel_path):
            runs.append({"run_dir": run_dir, "name": name, "status": "skip"})
            continue

        runs.append({"run_dir": run_dir, "name": name, "status": "todo"})
    return runs


def main():
    parser = argparse.ArgumentParser()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--runs_dir", default=os.path.join(base, "runs", "depth_ablation"))
    parser.add_argument("--device",   default="auto")
    parser.add_argument("--dry_run",  action="store_true")
    parser.add_argument("--force",    action="store_true",
                        help="Regenerate even if diagnostic_panel.png already exists")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if not os.path.isdir(args.runs_dir):
        print(f"Directory not found: {args.runs_dir}")
        return

    runs = discover_runs(args.runs_dir, force=args.force)
    todo  = [r for r in runs if r["status"] == "todo"]
    skip  = [r for r in runs if r["status"] == "skip"]

    print(f"Found {len(runs)} completed run(s) in {args.runs_dir}")
    print(f"  Need plots : {len(todo)}")
    print(f"  Already OK : {len(skip)}")

    if args.dry_run:
        print("\nWould generate plots for:")
        for r in todo:
            print(f"  {r['name']}")
        return

    if not todo:
        print("Nothing to do.")
        return

    print()
    for i, r in enumerate(todo, 1):
        run_dir = r["run_dir"]
        name    = r["name"]
        print(f"[{i:3d}/{len(todo)}] {name} ...", end=" ", flush=True)

        try:
            model, cfg = _load_model_from_run(run_dir, device)
            log_rows   = _load_log_rows(run_dir)

            with open(os.path.join(run_dir, "eval_results.json"),
                      encoding="utf-8") as f:
                eval_results = json.load(f)

            K  = cfg["K"]
            op = cfg.get("op_name", cfg.get("op", "NAND"))

            save_run_diagnostic_plots(
                model        = model,
                cfg          = cfg,
                log_rows     = log_rows,
                eval_results = eval_results,
                run_dir      = run_dir,
                K            = K,
                op           = op,
                device       = device,
                seed         = 999,
            )

            print(f"OK  (acc={eval_results.get('accuracy', 0):.3f})")

        except Exception as exc:
            print(f"ERROR: {exc}")

    print(f"\nDone. Diagnostic plots saved to each run's plots/ directory.")


if __name__ == "__main__":
    main()

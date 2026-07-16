"""
Backfill diagnostic_data.npz + enhanced plots for runs that completed
before save_run_diagnostic_plots was wired in.

For each run directory that has best_model.pt + config.json but is missing
plots/diagnostic_data.npz, this script:
  1. Reconstructs the model from the saved config
  2. Calls save_run_diagnostic_plots (one forward pass, no training)
  3. Writes diagnostic_data.npz + enhanced_raster.png + enhanced_flow.png

Usage:
    python -m scripts.backfill_diagnostic_plots --runs_dir runs/burst_comparison_seqD
    python -m scripts.backfill_diagnostic_plots --runs_dir runs/burst_comparison_seqD --force
    python -m scripts.backfill_diagnostic_plots --runs_dir runs  # scan all subdirs
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from snn.model import SNNSimultaneousModel
from utils.viz import save_run_diagnostic_plots
from utils.logger import setup_logger


def load_model_from_config(cfg: dict, device: str) -> SNNSimultaneousModel:
    model = SNNSimultaneousModel(
        n_queries=cfg["K"],
        n_hidden=cfg["n_hidden"],
        win_len=cfg["win_len"],
        read_len=cfg["read_len"],
        d_max=cfg["d_max"],
        train_mode=cfg.get("train_mode", "weights_and_delays"),
        delay_param_type=cfg.get("delay_param_type", "sigmoid"),
        delay_step=cfg.get("delay_step", 1.0),
        fixed_delay_value=cfg.get("fixed_delay_value", None),
        fixed_delay_distribution=cfg.get("fixed_delay_distribution", None),
        fixed_delay_seed=cfg.get("fixed_delay_seed", 0),
        fixed_delay_low=cfg.get("fixed_delay_low", 0.0),
        fixed_delay_high=cfg.get("fixed_delay_high", None),
        shared_delay=cfg.get("shared_delay", False),
        delay_init_mode=cfg.get("delay_init_mode", "constant"),
        delay_init_raw=cfg.get("delay_init_raw", -2.0),
        delay_init_std=cfg.get("delay_init_std", 0.25),
        lif_tau_m=cfg["lif_tau_m"],
        lif_threshold=cfg["lif_threshold"],
        lif_reset=cfg["lif_reset"],
        lif_refractory=cfg["lif_refractory"],
        dt=cfg["dt"],
        surrogate_beta=cfg["surrogate_beta"],
        n_input_channels=cfg.get("n_input", 2),
        readout_type=cfg.get("readout_type", "linear"),
        observation_mode=cfg.get("observation_mode", "late_window"),
        opponent_output_mode=cfg.get("opponent_output_mode", None),
        output_window_len=cfg.get("output_window_len", None),
    )
    return model


def collect_run_dirs(root: str) -> list[str]:
    """Find all run dirs (depth 1 or 2) that have best_model.pt + config.json."""
    found = []
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        if (os.path.exists(os.path.join(entry.path, "best_model.pt")) and
                os.path.exists(os.path.join(entry.path, "config.json"))):
            found.append(entry.path)
        else:
            # depth-2: runs/<group>/<run>/
            for sub in os.scandir(entry.path):
                if sub.is_dir() and (
                    os.path.exists(os.path.join(sub.path, "best_model.pt")) and
                    os.path.exists(os.path.join(sub.path, "config.json"))
                ):
                    found.append(sub.path)
    return sorted(found)


def needs_backfill(run_dir: str, force: bool) -> bool:
    npz_path = os.path.join(run_dir, "plots", "diagnostic_data.npz")
    if force:
        return True
    return not os.path.exists(npz_path)


def backfill_run(run_dir: str, device: str, logger) -> str:
    """Process one run. Returns 'done', 'skip', or 'error'."""
    cfg_path   = os.path.join(run_dir, "config.json")
    model_path = os.path.join(run_dir, "best_model.pt")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    K  = cfg.get("K")
    op = cfg.get("op_name", cfg.get("op", "NAND"))
    if K is None:
        logger.warning(f"  SKIP {run_dir}: no K in config")
        return "skip"

    try:
        model = load_model_from_config(cfg, device)
        state = torch.load(model_path, map_location=device,
                           weights_only=False)
        model.load_state_dict(state)
        model.to(device)
    except Exception as exc:
        logger.warning(f"  ERROR loading model {run_dir}: {exc}")
        return "error"

    # load_train_log for log_rows (best effort — empty list is fine)
    log_rows = []
    log_path = os.path.join(run_dir, "train_log.csv")
    if os.path.exists(log_path):
        try:
            import csv
            with open(log_path, encoding="utf-8") as f:
                raw = list(csv.DictReader(f))
            int_keys   = {"epoch"}
            float_keys = {"train_loss", "val_loss", "train_acc", "val_acc",
                          "mean_hidden_spikes", "time_s"}
            for row in raw:
                for k in int_keys:
                    if k in row:
                        row[k] = int(row[k])
                for k in float_keys:
                    if k in row:
                        row[k] = float(row[k])
            log_rows = raw
        except Exception:
            pass

    # Load test or validation results (best effort). Calibration runs
    # deliberately never create eval_results.json.
    eval_results = {}
    eval_path = os.path.join(run_dir, "eval_results.json")
    if not os.path.exists(eval_path):
        eval_path = os.path.join(run_dir, "validation_results.json")
    if os.path.exists(eval_path):
        try:
            with open(eval_path, encoding="utf-8") as f:
                eval_results = json.load(f)
        except Exception:
            pass

    try:
        save_run_diagnostic_plots(
            model, cfg, log_rows, eval_results,
            run_dir, K, op, device,
        )
        logger.info(f"  OK  {os.path.basename(run_dir)}")
        return "done"
    except Exception as exc:
        logger.warning(f"  ERROR plotting {run_dir}: {exc}")
        return "error"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs/burst_comparison_seqD")
    parser.add_argument("--force",    action="store_true",
                        help="Regenerate even if diagnostic_data.npz already exists")
    parser.add_argument("--device",   default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger = setup_logger("backfill")
    logger.info(f"Scanning {args.runs_dir} for runs needing backfill (device={device})")

    run_dirs = collect_run_dirs(args.runs_dir)
    to_process = [d for d in run_dirs if needs_backfill(d, args.force)]

    logger.info(f"Found {len(run_dirs)} runs, {len(to_process)} need backfill")

    counts = {"done": 0, "skip": 0, "error": 0}
    for rd in to_process:
        status = backfill_run(rd, device, logger)
        counts[status] = counts.get(status, 0) + 1

    logger.info(f"Done: {counts['done']}  Skip: {counts['skip']}  Error: {counts['error']}")


if __name__ == "__main__":
    main()

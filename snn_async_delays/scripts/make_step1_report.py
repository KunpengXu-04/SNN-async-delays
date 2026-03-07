"""
Generate a clean Step1 report and CSV summary from runs/step1_*.

Usage:
    python -m scripts.make_step1_report
"""

from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
CSV_PATH = ROOT / "step1_summary.csv"
REPORT_PATH = ROOT / "REPORT_step1.md"


def parse_run(run_dir: Path) -> dict | None:
    if not run_dir.is_dir() or not run_dir.name.startswith("step1_"):
        return None

    cfg_path = run_dir / "config.json"
    ev_path = run_dir / "eval_results.json"
    log_path = run_dir / "train_log.csv"
    if not (cfg_path.exists() and ev_path.exists()):
        return None

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    with open(ev_path, encoding="utf-8") as f:
        ev = json.load(f)

    row = {
        "run_id": run_dir.name,
        "op": cfg.get("op_name", ""),
        "train_mode": cfg.get("train_mode", ""),
        "hidden_size": cfg.get("hidden_size", 0),
        "seed": cfg.get("seed", 0),
        "delay_param_type": cfg.get("delay_param_type", "sigmoid"),
        "delay_step": cfg.get("delay_step", 1.0),
        "fixed_delay_value": cfg.get("effective_fixed_delay_value", cfg.get("fixed_delay_value", None)),
        "test_acc": ev.get("accuracy", 0.0),
        "mean_hidden_spikes": ev.get("mean_hidden_spikes", 0.0),
        "throughput_K_per_spk": ev.get("throughput_K_per_spk", 0.0),
        "ops_per_neuron_per_ms": ev.get("ops_per_neuron_per_ms", 0.0),
        "mean_active_hidden_fraction": ev.get("mean_active_hidden_fraction", 0.0),
    }

    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            logs = list(csv.DictReader(f))
        if logs:
            last = logs[-1]
            best = max(logs, key=lambda x: float(x.get("val_acc", 0.0)))
            row["final_train_acc"] = float(last.get("train_acc", 0.0))
            row["final_val_acc"] = float(last.get("val_acc", 0.0))
            row["best_val_acc"] = float(best.get("val_acc", 0.0))
            row["best_val_epoch"] = int(float(best.get("epoch", 0)))

    return row


def write_csv(rows: list[dict]):
    if not rows:
        return
    cols = [
        "run_id",
        "op",
        "train_mode",
        "hidden_size",
        "seed",
        "delay_param_type",
        "delay_step",
        "fixed_delay_value",
        "test_acc",
        "best_val_acc",
        "best_val_epoch",
        "final_train_acc",
        "final_val_acc",
        "mean_hidden_spikes",
        "throughput_K_per_spk",
        "ops_per_neuron_per_ms",
        "mean_active_hidden_fraction",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _mean(vals):
    return sum(vals) / max(len(vals), 1)


def _std(vals):
    if len(vals) <= 1:
        return 0.0
    m = _mean(vals)
    return (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5


def _ci95(vals):
    if len(vals) <= 1:
        return 0.0
    return 1.96 * _std(vals) / (len(vals) ** 0.5)


def write_report(rows: list[dict]):
    by_mode = defaultdict(list)
    by_group = defaultdict(list)
    for r in rows:
        by_mode[r["train_mode"]].append(r)
        key = (r["op"], r["train_mode"], r["hidden_size"])
        by_group[key].append(r)

    lines = []
    A = lines.append

    A("# REPORT - Step 1: Single-op Solvability Baseline")
    A("")
    A(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    A("")
    A("## Key Notes")
    A("")
    A("- Delay semantics: continuous delay with floor/ceil interpolation (not integer-only hard rounding).")
    A("- `weights_only` baseline uses fixed configured delay (recommended and configured as 0 ms for synchronous baseline).")
    A("")

    A("## Mode Summary")
    A("")
    A("| mode | n | acc_mean | acc_std | acc_ci95 | spikes_mean |")
    A("|---|---:|---:|---:|---:|---:|")
    for mode, grp in sorted(by_mode.items()):
        accs = [float(x["test_acc"]) for x in grp]
        spk = [float(x["mean_hidden_spikes"]) for x in grp]
        A(f"| {mode} | {len(grp)} | {_mean(accs):.4f} | {_std(accs):.4f} | {_ci95(accs):.4f} | {_mean(spk):.2f} |")

    A("")
    A("## Grouped Results (op, mode, hidden)")
    A("")
    A("| op | mode | hidden | n_seeds | acc_mean | acc_ci95 | k/spk_mean | density_mean |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for key in sorted(by_group.keys()):
        grp = by_group[key]
        accs = [float(x["test_acc"]) for x in grp]
        thr = [float(x["throughput_K_per_spk"]) for x in grp]
        dens = [float(x["ops_per_neuron_per_ms"]) for x in grp]
        A(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(grp)} | {_mean(accs):.4f} | {_ci95(accs):.4f} | {_mean(thr):.4f} | {_mean(dens):.6f} |"
        )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    rows = []
    if not RUNS_DIR.exists():
        print(f"Runs directory not found: {RUNS_DIR}")
        return

    for d in sorted(RUNS_DIR.iterdir()):
        r = parse_run(d)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No step1 runs found.")
        return

    write_csv(rows)
    write_report(rows)
    print(f"Wrote CSV: {CSV_PATH}")
    print(f"Wrote report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

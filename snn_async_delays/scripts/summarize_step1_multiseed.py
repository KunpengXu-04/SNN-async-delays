"""
Summarize Step 1 runs with multi-seed statistics.

Outputs:
  - step1_summary_<YYYYMMDD>.csv
  - step1_summary_grouped_<YYYYMMDD>.csv
"""

from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def _to_float(v):
    try:
        if v is None:
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def parse_run(run_dir: Path) -> dict | None:
    if not run_dir.is_dir() or not run_dir.name.startswith("step1_"):
        return None

    cfg_path = run_dir / "config.json"
    ev_path = run_dir / "eval_results.json"
    if not (cfg_path.exists() and ev_path.exists()):
        return None

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    with open(ev_path, encoding="utf-8") as f:
        ev = json.load(f)

    return {
        "run_id": run_dir.name,
        "op": cfg.get("op_name"),
        "train_mode": cfg.get("train_mode"),
        "hidden_size": cfg.get("hidden_size"),
        "seed": cfg.get("seed"),
        "delay_param_type": cfg.get("delay_param_type"),
        "delay_step": cfg.get("delay_step"),
        "fixed_delay_value": cfg.get("effective_fixed_delay_value", cfg.get("fixed_delay_value")),
        "test_acc": _to_float(ev.get("accuracy")),
        "mean_hidden_spikes": _to_float(ev.get("mean_hidden_spikes")),
        "throughput_K_per_spk": _to_float(ev.get("throughput_K_per_spk")),
        "ops_per_neuron_per_ms": _to_float(ev.get("ops_per_neuron_per_ms")),
        "mean_active_hidden_fraction": _to_float(ev.get("mean_active_hidden_fraction")),
    }


def mean(vals):
    return sum(vals) / max(len(vals), 1)


def std(vals):
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5


def ci95(vals):
    if len(vals) <= 1:
        return 0.0
    return 1.96 * std(vals) / (len(vals) ** 0.5)


def write_csv(path: Path, rows: list[dict], cols: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    rows = []
    for d in sorted(RUNS.iterdir()):
        r = parse_run(d)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No Step1 runs found.")
        return

    stamp = datetime.now().strftime("%Y%m%d")
    per_run_path = ROOT / f"step1_summary_{stamp}.csv"

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
        "mean_hidden_spikes",
        "throughput_K_per_spk",
        "ops_per_neuron_per_ms",
        "mean_active_hidden_fraction",
    ]
    write_csv(per_run_path, rows, cols)

    grouped = {}
    for r in rows:
        key = (
            r["op"],
            r["train_mode"],
            r["hidden_size"],
            r["delay_param_type"],
            r["delay_step"],
            r["fixed_delay_value"],
        )
        grouped.setdefault(key, []).append(r)

    g_rows = []
    for key, grp in grouped.items():
        accs = [_to_float(x["test_acc"]) for x in grp]
        spks = [_to_float(x["mean_hidden_spikes"]) for x in grp]
        thrs = [_to_float(x["throughput_K_per_spk"]) for x in grp]
        dens = [_to_float(x["ops_per_neuron_per_ms"]) for x in grp]

        g_rows.append(
            {
                "op": key[0],
                "train_mode": key[1],
                "hidden_size": key[2],
                "delay_param_type": key[3],
                "delay_step": key[4],
                "fixed_delay_value": key[5],
                "n_seeds": len(grp),
                "acc_mean": mean(accs),
                "acc_std": std(accs),
                "acc_ci95": ci95(accs),
                "spikes_mean": mean(spks),
                "throughput_mean": mean(thrs),
                "density_mean": mean(dens),
            }
        )

    grouped_path = ROOT / f"step1_summary_grouped_{stamp}.csv"
    g_cols = [
        "op",
        "train_mode",
        "hidden_size",
        "delay_param_type",
        "delay_step",
        "fixed_delay_value",
        "n_seeds",
        "acc_mean",
        "acc_std",
        "acc_ci95",
        "spikes_mean",
        "throughput_mean",
        "density_mean",
    ]
    write_csv(grouped_path, g_rows, g_cols)

    print(f"Wrote: {per_run_path}")
    print(f"Wrote: {grouped_path}")


if __name__ == "__main__":
    main()

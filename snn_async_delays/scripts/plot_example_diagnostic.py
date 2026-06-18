"""
Generate example diagnostic panels from saved diagnostic_data.npz files.
No model re-run needed.

Usage (from snn_async_delays/):
    python -m scripts.plot_example_diagnostic
"""

import os, json, csv, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.viz import replot_run_diagnostics

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(BASE, "runs")


def _load_meta(run_dir):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "eval_results.json"), encoding="utf-8") as f:
        ev = json.load(f)
    log_rows = []
    csv_path = os.path.join(run_dir, "train_log.csv")
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                log_rows.append({k: (float(v) if k != "epoch" else int(float(v)))
                                  for k, v in row.items()})
    return cfg, ev, log_rows


examples = [
    # K=1 dense raster (single query, full window dedicated to one op)
    ("step3_planD_4ops_16k/w_and_d_K1_seed42",   "4-op 16k  w+d  K=1  acc=96.5%"),
    ("step3_planD_4ops_16k/d0_control_K1_seed42", "4-op 16k  d=0  K=1  acc=64%"),
    # K=3 multi-query comparison
    ("step3_planD/w_and_d_K3_seed42",   "w+d  K=3  (with delays)"),
    ("step3_planD/d0_control_K3_seed42", "d=0  K=3  (no delays)"),
]

for rel_path, label in examples:
    run_dir = os.path.join(RUNS, rel_path)
    npz_path = os.path.join(run_dir, "plots", "diagnostic_data.npz")
    if not os.path.exists(npz_path):
        print(f"[skip] {label} -- no diagnostic_data.npz")
        continue

    cfg, ev, log_rows = _load_meta(run_dir)
    K    = cfg.get("K", ev.get("K", 1))
    seed = cfg.get("seed", 42)

    print(f"Plotting: {label}  (K={K}, seed={seed})")
    replot_run_diagnostics(run_dir, cfg, log_rows, ev, K=K, seed=seed)
    out = os.path.join(run_dir, "plots", "diagnostic_panel.png")
    print(f"  -> {out}\n")

print("Done.")

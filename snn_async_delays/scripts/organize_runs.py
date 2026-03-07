"""
Organize runs/ non-destructively by generating metadata indexes.
"""

from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"D:\xukun\Documents\IC\SNN\SNN_project\snn_async_delays")
RUNS = ROOT / "runs"
META = RUNS / "_meta"
META.mkdir(exist_ok=True)

rows = []
for d in sorted(RUNS.iterdir()):
    if not d.is_dir() or d.name.startswith("_"):
        continue

    step = "other"
    if d.name.startswith("step1_"):
        step = "step1"
    elif d.name.startswith("step2_"):
        step = "step2"
    elif d.name.startswith("step3_"):
        step = "step3"

    cfg = {}
    ev = {}
    cfg_path = d / "config.json"
    ev_path = d / "eval_results.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    if ev_path.exists():
        with open(ev_path, encoding="utf-8") as f:
            ev = json.load(f)

    rows.append({
        "run_dir": d.name,
        "step": step,
        "op": cfg.get("op_name", ""),
        "train_mode": cfg.get("train_mode", ""),
        "delay_param_type": cfg.get("delay_param_type", ""),
        "delay_step": cfg.get("delay_step", ""),
        "hidden_size": cfg.get("hidden_size", ""),
        "K": ev.get("K", cfg.get("K", "")),
        "seed": cfg.get("seed", ""),
        "accuracy": ev.get("accuracy", ""),
        "throughput_K_per_spk": ev.get("throughput_K_per_spk", ""),
        "ops_per_neuron_per_ms": ev.get("ops_per_neuron_per_ms", ""),
        "updated": datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    })

inv_csv = META / "run_inventory.csv"
with open(inv_csv, "w", newline="", encoding="utf-8") as f:
    cols = list(rows[0].keys()) if rows else [
        "run_dir","step","op","train_mode","delay_param_type","delay_step",
        "hidden_size","K","seed","accuracy","throughput_K_per_spk","ops_per_neuron_per_ms","updated"
    ]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

by_step = defaultdict(list)
for r in rows:
    by_step[r["step"]].append(r)

md = []
md.append("# Runs Inventory")
md.append("")
md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
md.append("")
md.append("## Counts")
md.append("")
md.append("| step | count |")
md.append("|---|---:|")
for step, grp in sorted(by_step.items()):
    md.append(f"| {step} | {len(grp)} |")

if by_step.get("step1"):
    md.append("")
    md.append("## Step1 by train_mode")
    md.append("")
    md.append("| train_mode | count |")
    md.append("|---|---:|")
    mode_count = defaultdict(int)
    for r in by_step["step1"]:
        mode_count[r["train_mode"]] += 1
    for k, v in sorted(mode_count.items()):
        md.append(f"| {k} | {v} |")

md.append("")
md.append("## Notes")
md.append("")
md.append("- This is non-destructive organization: run directories are unchanged.")
md.append("- Use `run_inventory.csv` for filtering/sorting in Excel or pandas.")

inv_md = META / "run_inventory.md"
with open(inv_md, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

# Archive top-level summary files (copy only)
archive = META / "summaries"
archive.mkdir(exist_ok=True)
for f in RUNS.glob("*_summary*.json"):
    target = archive / f.name
    target.write_bytes(f.read_bytes())

print(f"Wrote: {inv_csv}")
print(f"Wrote: {inv_md}")
print(f"Archived summaries to: {archive}")

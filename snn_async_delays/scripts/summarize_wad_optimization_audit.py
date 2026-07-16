"""Summarize WAD optimization audit Stage A without modifying run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def mean(values):
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/exploratory/wad_optimization_audit_v1/stage_a")
    parser.add_argument("--output-dir", default="docs/generated/wad_optimization_audit_v1")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.output_dir)
    if not root.is_absolute(): root = BASE / root
    if not out.is_absolute(): out = BASE / out
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for result_path in sorted(root.rglob("validation_results.json")):
        run_dir = result_path.parent
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        with (run_dir / "train_log.csv").open(newline="", encoding="utf-8") as handle:
            log = list(csv.DictReader(handle))
        final = log[-1]
        delay_grads = [float(x["delay_grad_norm"]) for x in log]
        row = {
            "threshold": float(cfg["lif_threshold"]), "condition": cfg["name"],
            "seed": int(cfg["seed"]), "worst_query_accuracy": result["worst_query_accuracy"],
            "exact_trial_accuracy": result["exact_trial_accuracy"],
            "mean_hidden_spikes": result["mean_hidden_spikes"],
            "final_delay_grad_norm": float(final["delay_grad_norm"]),
            "mean_delay_grad_norm": mean(delay_grads),
            "nonzero_finite_delay_grad_all_epochs": all(math.isfinite(x) and x > 0 for x in delay_grads),
            "final_mean_abs_delay_movement": float(final["mean_abs_delay_movement"]),
            "final_delay_saturation_fraction": float(final["delay_saturation_fraction"]),
            "run_dir": str(run_dir.relative_to(BASE)),
        }
        rows.append(row)

    expected = 18
    if len(rows) != expected:
        raise SystemExit(f"Incomplete Stage A: found {len(rows)}/{expected} results")

    groups = defaultdict(list)
    for row in rows: groups[(row["threshold"], row["condition"])].append(row)
    grouped = []
    for (threshold, condition), items in sorted(groups.items()):
        grouped.append({
            "threshold": threshold, "condition": condition, "n": len(items),
            "mean_worst_query_accuracy": mean([x["worst_query_accuracy"] for x in items]),
            "min_worst_query_accuracy": min(x["worst_query_accuracy"] for x in items),
            "max_worst_query_accuracy": max(x["worst_query_accuracy"] for x in items),
            "mean_exact_trial_accuracy": mean([x["exact_trial_accuracy"] for x in items]),
            "mean_hidden_spikes": mean([x["mean_hidden_spikes"] for x in items]),
            "mean_final_delay_grad_norm": mean([x["final_delay_grad_norm"] for x in items]),
            "mean_delay_grad_norm": mean([x["mean_delay_grad_norm"] for x in items]),
            "min_final_delay_movement": min(x["final_mean_abs_delay_movement"] for x in items),
            "mean_final_delay_movement": mean([x["final_mean_abs_delay_movement"] for x in items]),
            "max_final_saturation_fraction": max(x["final_delay_saturation_fraction"] for x in items),
            "all_epoch_gradients_nonzero_finite": all(x["nonzero_finite_delay_grad_all_epochs"] for x in items),
        })

    decisions = []
    for threshold in sorted({x["threshold"] for x in rows}):
        items = [x for x in rows if x["threshold"] == threshold]
        wad = [x for x in items if x["condition"] == "w_and_d"]
        viable = (
            all(1 <= x["mean_hidden_spikes"] <= 70 for x in items)
            and all(.55 <= x["worst_query_accuracy"] <= .90 for x in items)
            and all(x["nonzero_finite_delay_grad_all_epochs"] for x in wad)
            and all(x["final_mean_abs_delay_movement"] >= .05 for x in wad)
        )
        decisions.append({
            "threshold": threshold, "viable_all_cells": viable,
            "across_condition_mean_hidden_spikes": mean([x["mean_hidden_spikes"] for x in items]),
            "distance_to_target_10": abs(mean([x["mean_hidden_spikes"] for x in items]) - 10),
        })
    viable = [x for x in decisions if x["viable_all_cells"]]
    selected = min(viable, key=lambda x: x["distance_to_target_10"])["threshold"] if viable else None

    for name, data in (("stage_a_cells.csv", rows), ("stage_a_grouped.csv", grouped)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    decision = {"complete": True, "n_cells": len(rows), "selection_rule_applied_per_cell": True,
                "thresholds": decisions, "selected_threshold": selected}
    (out / "stage_a_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({"grouped": grouped, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()

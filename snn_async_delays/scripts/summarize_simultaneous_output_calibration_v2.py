"""Combine spatial-v1 and temporal-scaffold-v2 evidence and apply the v2 rule."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
SPATIAL = BASE / "runs/exploratory/simultaneous_output_interface_calibration_v1/spatial"
TEMPORAL = BASE / "runs/exploratory/simultaneous_output_interface_calibration_v2"
OUT = BASE / "docs/generated/simultaneous_output_interface_calibration_v2"


def avg(xs): return sum(xs) / len(xs)


def collect(root: Path, interface: str) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("validation_results.json")):
        cfg = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        result = json.loads(path.read_text(encoding="utf-8"))
        spikes = result.get("per_query_output_spikes") or []
        rows.append({
            "interface": interface, "threshold": float(cfg["lif_output_threshold"]),
            "seed": int(cfg["seed"]), "worst_balanced": result["worst_query_balanced_accuracy"],
            "per_query_balanced": result["per_query_balanced_accuracy"],
            "exact_trial_accuracy": result["exact_trial_accuracy"],
            "silent_rate": result["output_silent_rate"], "tie_rate": result["output_tie_rate"],
            "collision_rate": result["output_collision_rate"],
            "mean_output_spikes_per_query": avg(spikes),
        })
    return rows


def main() -> None:
    rows = collect(SPATIAL, "spatial_d0") + collect(TEMPORAL, "temporal_scaffold")
    if len(rows) != 18:
        raise SystemExit(f"Incomplete v2 evidence: found {len(rows)}/18 (9 reused spatial + 9 new temporal)")
    grouped = defaultdict(list)
    for row in rows: grouped[(row["threshold"], row["interface"])].append(row)
    decisions = []
    for threshold in sorted({x["threshold"] for x in rows}):
        arm_summaries, arms_viable = {}, True
        for interface in ("spatial_d0", "temporal_scaffold"):
            items = grouped[(threshold, interface)]
            summary = {
                "mean_worst_balanced": avg([x["worst_balanced"] for x in items]),
                "mean_silent_rate": avg([x["silent_rate"] for x in items]),
                "mean_tie_rate": avg([x["tie_rate"] for x in items]),
                "mean_collision_rate": avg([x["collision_rate"] for x in items]),
                "mean_output_spikes_per_query": avg([x["mean_output_spikes_per_query"] for x in items]),
            }
            summary["viable"] = (.50 <= summary["mean_worst_balanced"] <= .95
                                   and summary["mean_silent_rate"] < .50
                                   and summary["mean_tie_rate"] < .50
                                   and summary["mean_collision_rate"] < .25)
            arms_viable = arms_viable and summary["viable"]
            arm_summaries[interface] = summary
        pooled_spikes = avg([x["mean_output_spikes_per_query"] for x in rows if x["threshold"] == threshold])
        decisions.append({"threshold": threshold, "arms": arm_summaries,
                          "viable": arms_viable, "pooled_mean_output_spikes": pooled_spikes,
                          "distance_to_one_spike": abs(pooled_spikes - 1)})
    viable = [x for x in decisions if x["viable"]]
    selected = min(viable, key=lambda x: x["distance_to_one_spike"])["threshold"] if viable else None
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    output = {"complete": True, "n_evidence_cells": len(rows), "thresholds": decisions,
              "selected_output_threshold": selected}
    (OUT / "decision.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    metrics = [
        ("worst_balanced", "Worst balanced accuracy", 0.5),
        ("exact_trial_accuracy", "Exact-trial accuracy", None),
        ("silent_rate", "Silent-window rate", 0.5),
        ("mean_output_spikes_per_query", "Output spikes / query-window", 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    thresholds = sorted({x["threshold"] for x in rows})
    styles = {
        "spatial_d0": ("#4C72B0", -0.04, "spatial d0"),
        "temporal_scaffold": ("#DD8452", 0.04, "temporal scaffold"),
    }
    for ax, (key, title, reference) in zip(axes.flat, metrics):
        for interface, (color, offset, label) in styles.items():
            means = []
            for threshold in thresholds:
                values = [r[key] for r in rows
                          if r["interface"] == interface and r["threshold"] == threshold]
                x = threshold + offset
                ax.scatter([x] * len(values), values, color=color, alpha=.55, s=24)
                means.append(avg(values))
            ax.plot([x + offset for x in thresholds], means, color=color,
                    marker="o", linewidth=1.8, label=label)
        if reference is not None:
            ax.axhline(reference, color="#555", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Output threshold")
        ax.set_xticks(thresholds)
        ax.grid(True, alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Output-interface calibration v2: seed-level evidence")
    fig.savefig(OUT / "threshold_summary.png", dpi=180)
    plt.close(fig)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

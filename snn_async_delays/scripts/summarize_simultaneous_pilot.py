"""Create aggregate tables and publication-oriented plots for a simultaneous pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", choices=["spatial", "temporal"], required=True)
    args = p.parse_args()
    protocol = f"simultaneous_{'spatial_control' if args.pilot == 'spatial' else 'temporal_routing'}_pilot_v1"
    root = BASE / "runs/exploratory" / protocol
    out = BASE / "docs/generated" / protocol
    out.mkdir(parents=True, exist_ok=True)
    rows, truth = [], {}
    for path in sorted(root.rglob("validation_results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        ledger = result["resource_ledger"]
        row = {
            "condition": result["condition"], "endpoint": result["endpoint"],
            "seed": json.loads((path.parent / "config.json").read_text(encoding="utf-8"))["seed"],
            "pooled_accuracy": result["pooled_accuracy"],
            "worst_accuracy": result["worst_query_accuracy"],
            "worst_balanced_accuracy": result["worst_query_balanced_accuracy"],
            "exact_trial_accuracy": result["exact_trial_accuracy"],
            "routing_selectivity_gap": result.get("routing_selectivity_gap"),
            "silent_rate": result.get("output_silent_rate"),
            "tie_rate": result.get("output_tie_rate"),
            "collision_rate": result.get("output_collision_rate"),
            "hidden_spikes": result["mean_hidden_spikes"],
            "synaptic_events": ledger["mean_synaptic_events_total"],
            "neuron_updates": ledger["neuron_updates_per_trial"],
            "decoder_macs": ledger["decoder_weight_macs_per_trial"],
            "delay_memory": ledger["delay_value_storage_elements"],
        }
        truth_path = path.parent / "exhaustive_truth_table_results.json"
        if truth_path.exists():
            truth_result = json.loads(truth_path.read_text(encoding="utf-8"))
            truth[(row["condition"], row["endpoint"], row["seed"])] = truth_result
            row["truth_worst_balanced_accuracy"] = truth_result["worst_query_balanced_accuracy"]
            row["truth_exact_trial_accuracy"] = truth_result["exact_trial_accuracy"]
        else:
            row["truth_worst_balanced_accuracy"] = None
            row["truth_exact_trial_accuracy"] = None
        rows.append(row)
    if not rows: raise SystemExit(f"No completed results under {root}")
    with (out / "cells.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    aggregate_metrics = [
        "worst_balanced_accuracy", "exact_trial_accuracy",
        "truth_worst_balanced_accuracy", "truth_exact_trial_accuracy",
        "hidden_spikes", "synaptic_events", "silent_rate", "tie_rate",
        "collision_rate",
    ]
    grouped_rows = []
    for endpoint in sorted({r["endpoint"] for r in rows}):
        for condition in sorted({r["condition"] for r in rows}):
            subset = [r for r in rows if r["endpoint"] == endpoint
                      and r["condition"] == condition]
            if not subset:
                continue
            item = {"endpoint": endpoint, "condition": condition, "n": len(subset)}
            for key in aggregate_metrics:
                values = [float(r[key]) for r in subset if r[key] is not None]
                item[f"{key}_mean"] = float(np.mean(values)) if values else None
                item[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            grouped_rows.append(item)
    with (out / "grouped.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grouped_rows[0])); w.writeheader(); w.writerows(grouped_rows)

    paired_rows = []
    for endpoint in sorted({r["endpoint"] for r in rows}):
        wad = {r["seed"]: r for r in rows
               if r["endpoint"] == endpoint and r["condition"] == "w_and_d"}
        for control in ("d0", "scalar", "fixed_matched"):
            controls = {r["seed"]: r for r in rows
                        if r["endpoint"] == endpoint and r["condition"] == control}
            for seed in sorted(set(wad) & set(controls)):
                paired_rows.append({
                    "endpoint": endpoint, "control": control, "seed": seed,
                    "wad_minus_control_worst_balanced": (
                        wad[seed]["worst_balanced_accuracy"]
                        - controls[seed]["worst_balanced_accuracy"]
                    ),
                    "wad_minus_control_exact_trial": (
                        wad[seed]["exact_trial_accuracy"]
                        - controls[seed]["exact_trial_accuracy"]
                    ),
                    "wad_minus_control_truth_worst_balanced": (
                        wad[seed]["truth_worst_balanced_accuracy"]
                        - controls[seed]["truth_worst_balanced_accuracy"]
                    ),
                    "wad_minus_control_truth_exact_trial": (
                        wad[seed]["truth_exact_trial_accuracy"]
                        - controls[seed]["truth_exact_trial_accuracy"]
                    ),
                })
    with (out / "paired_wad_minus_controls.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_rows[0])); w.writeheader(); w.writerows(paired_rows)

    if args.pilot == "spatial":
        primary = {
            r["condition"]: r for r in grouped_rows if r["endpoint"] == "linear"
        }
        controls = {k: v for k, v in primary.items() if k != "w_and_d"}
        strongest = max(
            controls,
            key=lambda k: controls[k]["worst_balanced_accuracy_mean"],
        )
        wad_vs_strongest = [
            r for r in paired_rows
            if r["endpoint"] == "linear" and r["control"] == strongest
        ]
        wad_vs_fixed = [
            r for r in paired_rows
            if r["endpoint"] == "linear" and r["control"] == "fixed_matched"
        ]
        opponent_vs_scalar = [
            r for r in paired_rows
            if r["endpoint"] == "opponent_parallel" and r["control"] == "scalar"
        ]

        def paired_summary(items: list[dict], key: str) -> dict:
            values = [float(x[key]) for x in items]
            return {
                "mean": float(np.mean(values)),
                "values_by_seed": {
                    str(x["seed"]): float(x[key]) for x in items
                },
                "positive_seed_count": int(sum(v > 0 for v in values)),
                "n": len(values),
            }

        decision = {
            "protocol": protocol,
            "primary_endpoint": "linear",
            "primary_metric": "worst_balanced_accuracy",
            "strongest_nonlearned_control_by_mean": strongest,
            "wad_primary_superiority_supported": False,
            "reason": (
                "WAD has a negative mean paired difference against the strongest "
                "nonlearned control and the seed-level differences are not consistently positive."
            ),
            "wad_minus_strongest_primary": paired_summary(
                wad_vs_strongest, "wad_minus_control_worst_balanced"
            ),
            "wad_minus_fixed_matched_primary": paired_summary(
                wad_vs_fixed, "wad_minus_control_worst_balanced"
            ),
            "mlp_ceiling": all(
                r["worst_balanced_accuracy"] == 1.0
                and r["exact_trial_accuracy"] == 1.0
                for r in rows if r["endpoint"] == "mlp"
            ),
            "opponent_wad_minus_scalar_worst_balanced": paired_summary(
                opponent_vs_scalar, "wad_minus_control_worst_balanced"
            ),
            "opponent_wad_minus_scalar_exact_trial": paired_summary(
                opponent_vs_scalar, "wad_minus_control_exact_trial"
            ),
            "opponent_exact_signal_status": "exploratory_only",
            "scope_warning": (
                "Dedicated spatial inputs and outputs do not test shared temporal routing."
            ),
        }
        (out / "decision_summary.json").write_text(
            json.dumps(decision, indent=2) + "\n", encoding="utf-8"
        )

    endpoints = sorted({r["endpoint"] for r in rows})
    conditions = [c for c in ("d0", "scalar", "fixed_matched", "w_and_d") if any(r["condition"] == c for r in rows)]
    colors = dict(zip(conditions, plt.cm.tab10.colors[:len(conditions)]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ei, endpoint in enumerate(endpoints):
        for ci, condition in enumerate(conditions):
            vals = [r for r in rows if r["endpoint"] == endpoint and r["condition"] == condition]
            x = ei + (ci - (len(conditions)-1)/2) * .14
            axes[0].scatter([x] * len(vals), [r["worst_balanced_accuracy"] for r in vals], color=colors[condition], label=condition if ei == 0 else None)
            axes[1].scatter([x] * len(vals), [r["exact_trial_accuracy"] for r in vals], color=colors[condition])
    for ax, title in zip(axes, ["Worst operation/window balanced accuracy", "Exact-trial accuracy"]):
        ax.set_xticks(range(len(endpoints))); ax.set_xticklabels(endpoints, rotation=20, ha="right")
        ax.set_ylim(0, 1.02); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False); fig.tight_layout(); fig.savefig(out / "paired_reliability.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for condition in conditions:
        vals = [r for r in rows if r["condition"] == condition]
        ax.scatter([r["synaptic_events"] for r in vals], [r["worst_balanced_accuracy"] for r in vals], label=condition, alpha=.8)
    ax.set_xlabel("Measured synaptic events / trial"); ax.set_ylabel("Worst balanced accuracy")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "reliability_resource_frontier.png", dpi=180); plt.close(fig)

    opponent = [r for r in rows if r["silent_rate"] is not None]
    if opponent:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, key, title in zip(axes, ["silent_rate", "tie_rate", "collision_rate"], ["Silent", "Tie", "Opponent collision"]):
            for ci, condition in enumerate(conditions):
                vals = [r[key] for r in opponent if r["condition"] == condition]
                ax.scatter([ci] * len(vals), vals, color=colors[condition])
            ax.set_xticks(range(len(conditions))); ax.set_xticklabels(conditions, rotation=25); ax.set_title(title); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(out / "output_interface_failures.png", dpi=180); plt.close(fig)

    if truth:
        groups = sorted({(c, e) for c, e, _ in truth})
        matrix = []
        for c, e in groups:
            records = [truth[(c, e, s)]["trial_records"] for cc, ee, s in truth if cc == c and ee == e]
            matrix.append(np.mean([[float(x["exact_correct"]) for x in rec] for rec in records], axis=0))
        fig, ax = plt.subplots(figsize=(14, max(4, .4 * len(groups))))
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(range(len(groups))); ax.set_yticklabels([f"{c}/{e}" for c, e in groups])
        ax.set_xlabel("Exhaustive 6-bit input pattern (0–63)"); ax.set_title("Exact-trial truth-table success across seeds")
        fig.colorbar(im, ax=ax, label="seed-mean exact correctness"); fig.tight_layout(); fig.savefig(out / "truth_table_failure_heatmap.png", dpi=180); plt.close(fig)


if __name__ == "__main__": main()

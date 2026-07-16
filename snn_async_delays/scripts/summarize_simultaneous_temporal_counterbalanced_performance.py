"""Summarize the locked counterbalanced temporal performance matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from scripts.audit_temporal_checkpoint_mechanism import (
    arrival_distributions, build_model, exhaustive_batch, window_trace_rows,
)


BASE = Path(__file__).resolve().parents[1]
PROTOCOL = "simultaneous_temporal_counterbalanced_performance_v1"
ROOT = BASE / "runs/exploratory" / PROTOCOL
OUT = BASE / "docs/generated" / PROTOCOL
REQUIRED = [
    "config.json", "best_model.pt", "last_model.pt", "train_log.csv",
    "validation_results.json", "exhaustive_truth_table_results.json",
    "plots/diagnostic_data.npz", "plots/diagnostic_panel.png",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    prereg = yaml.safe_load(
        (BASE / "configs/simultaneous_temporal_counterbalanced_performance_v1.yaml")
        .read_text(encoding="utf-8")
    )
    expected = prereg["cells"]
    paths = sorted(ROOT.rglob("validation_results.json"))
    if len(paths) != expected:
        raise SystemExit(f"Incomplete matrix: found {len(paths)}, expected {expected}")
    cells, long_rows = [], []
    for path in paths:
        run = path.parent
        missing = [name for name in REQUIRED if not (run / name).exists()]
        if missing:
            raise SystemExit(f"Incomplete artifacts in {run}: {missing}")
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        result = json.loads(path.read_text(encoding="utf-8"))
        order_id = run.parent.name
        model = build_model(cfg)
        model.load_state_dict(torch.load(
            run / "last_model.pt", map_location="cpu", weights_only=True
        ))
        model.eval()
        spike_input, labels = exhaustive_batch(cfg, "cpu")
        with torch.no_grad():
            _, info = model(spike_input, record=True)
        traces = window_trace_rows(cfg["name"], info, labels, cfg)
        _, realized = arrival_distributions(
            model, spike_input, info["hidden_spike_train"]
        )
        start, width = int(cfg["win_len"]), int(cfg["output_window_len"])
        arrival_mass = [
            float(realized[start + q * width:start + (q + 1) * width].sum())
            for q in range(int(cfg["K"]))
        ]
        ledger = result["resource_ledger"]
        cell = {
            "order_id": order_id, "condition": cfg["name"], "seed": cfg["seed"],
            "query_ops": json.dumps(cfg["query_ops"]),
            "worst_balanced_accuracy": result["worst_query_balanced_accuracy"],
            "exact_trial_accuracy": result["exact_trial_accuracy"],
            "pooled_accuracy": result["pooled_accuracy"],
            "min_output_current_support": min(
                item["output_current_support_fraction"] for item in traces
            ),
            "min_output_activity": min(item["output_activity_fraction"] for item in traces),
            "min_realized_arrival_mass": min(arrival_mass),
            "hidden_spikes": result["mean_hidden_spikes"],
            "synaptic_events": ledger["mean_synaptic_events_total"],
            "neuron_updates": ledger["neuron_updates_per_trial"],
            "delay_memory": ledger["delay_value_storage_elements"],
        }
        cells.append(cell)
        for position, operation in enumerate(cfg["query_ops"]):
            long_rows.append({
                "order_id": order_id, "condition": cfg["name"], "seed": cfg["seed"],
                "operation": operation, "position": position,
                "balanced_accuracy": result["per_query_balanced_accuracy"][position],
                "output_current_support": traces[position]["output_current_support_fraction"],
                "output_activity": traces[position]["output_activity_fraction"],
                "signed_current_class_gap": traces[position]["signed_current_class_gap"],
                "realized_arrival_mass": arrival_mass[position],
            })

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "cells.csv", cells)
    write_csv(OUT / "operation_position.csv", long_rows)

    seed_rows = []
    for condition in prereg["conditions"]:
        for seed in prereg["held_out_seeds"]:
            lr = [r for r in long_rows if r["condition"] == condition and r["seed"] == seed]
            cr = [r for r in cells if r["condition"] == condition and r["seed"] == seed]
            if len(lr) != 9 or len(cr) != 3:
                raise SystemExit(f"Unbalanced condition/seed: {condition}/{seed}")
            seed_rows.append({
                "condition": condition, "seed": seed,
                "primary_min_operation_position_balanced": min(r["balanced_accuracy"] for r in lr),
                "mean_operation_position_balanced": float(np.mean([r["balanced_accuracy"] for r in lr])),
                "mean_exact_trial_accuracy": float(np.mean([r["exact_trial_accuracy"] for r in cr])),
                "worst_order_balanced_accuracy": min(r["worst_balanced_accuracy"] for r in cr),
                "min_output_current_support": min(r["output_current_support"] for r in lr),
                "mean_synaptic_events": float(np.mean([r["synaptic_events"] for r in cr])),
            })
    write_csv(OUT / "condition_seed_primary.csv", seed_rows)

    summary_rows = []
    for condition in prereg["conditions"]:
        sr = [r for r in seed_rows if r["condition"] == condition]
        item = {"condition": condition, "n": len(sr)}
        for key in (
            "primary_min_operation_position_balanced",
            "mean_operation_position_balanced", "mean_exact_trial_accuracy",
            "worst_order_balanced_accuracy", "min_output_current_support",
            "mean_synaptic_events",
        ):
            values = [r[key] for r in sr]
            item[f"{key}_mean"] = float(np.mean(values))
            item[f"{key}_sd"] = float(np.std(values, ddof=1))
        summary_rows.append(item)
    write_csv(OUT / "condition_summary.csv", summary_rows)

    marginal_rows = []
    for condition in prereg["conditions"]:
        for effect_type, levels in (("operation", ["XOR", "NAND", "NOR"]),
                                    ("position", [0, 1, 2])):
            for level in levels:
                values = [r["balanced_accuracy"] for r in long_rows
                          if r["condition"] == condition and r[effect_type] == level]
                marginal_rows.append({
                    "condition": condition, "effect_type": effect_type,
                    "level": level, "n": len(values),
                    "balanced_accuracy_mean": float(np.mean(values)),
                    "balanced_accuracy_sd": float(np.std(values, ddof=1)),
                })
    write_csv(OUT / "marginal_effects.csv", marginal_rows)

    controls = [c for c in prereg["conditions"] if c != "w_and_d"]
    by_condition = {r["condition"]: r for r in summary_rows}
    primary_key = "primary_min_operation_position_balanced_mean"
    best_control_primary = max(by_condition[c][primary_key] for c in controls)
    strongest_controls = [
        c for c in controls
        if np.isclose(by_condition[c][primary_key], best_control_primary,
                      atol=1e-12, rtol=0.0)
    ]
    wad_seed = {r["seed"]: r for r in seed_rows if r["condition"] == "w_and_d"}
    paired_rows = []
    for control in controls:
        control_seed = {r["seed"]: r for r in seed_rows if r["condition"] == control}
        for seed in prereg["held_out_seeds"]:
            paired_rows.append({
                "control": control, "seed": seed,
                "wad_minus_control_primary": (
                    wad_seed[seed]["primary_min_operation_position_balanced"]
                    - control_seed[seed]["primary_min_operation_position_balanced"]
                ),
                "wad_minus_control_exact": (
                    wad_seed[seed]["mean_exact_trial_accuracy"]
                    - control_seed[seed]["mean_exact_trial_accuracy"]
                ),
                "wad_minus_control_mean_balanced": (
                    wad_seed[seed]["mean_operation_position_balanced"]
                    - control_seed[seed]["mean_operation_position_balanced"]
                ),
            })
    write_csv(OUT / "paired_wad_minus_controls.csv", paired_rows)

    rules = prereg["locked_decision_rules"]
    routing = {}
    for condition in prereg["conditions"]:
        values = [r["primary_min_operation_position_balanced"]
                  for r in seed_rows if r["condition"] == condition]
        routing[condition] = bool(
            all(v >= rules["routing_floor_each_seed"] for v in values)
            and np.mean(values) >= rules["routing_mean_primary_floor"]
        )
    strongest_pairs = [r for r in paired_rows if r["control"] in strongest_controls]
    mean_primary_delta = float(np.mean([r["wad_minus_control_primary"] for r in strongest_pairs]))
    mean_exact_delta = float(np.mean([r["wad_minus_control_exact"] for r in strongest_pairs]))
    positive_by_control = {
        control: sum(r["wad_minus_control_primary"] > 0 for r in strongest_pairs
                     if r["control"] == control)
        for control in strongest_controls
    }
    fixed_full_pairs = [r for r in paired_rows if r["control"] == "fixed_full_support"]
    decision = {
        "protocol": PROTOCOL,
        "complete": True,
        "routing_floor_supported": routing,
        "strongest_nonlearned_controls": strongest_controls,
        "primary_comparator_is_tied": len(strongest_controls) > 1,
        "wad_minus_strongest_primary_mean": mean_primary_delta,
        "wad_minus_strongest_exact_mean": mean_exact_delta,
        "wad_positive_primary_seeds_by_control": positive_by_control,
        "wad_superiority_supported": bool(
            mean_primary_delta >= rules["wad_superiority_mean_margin"]
            and all(n >= rules["wad_superiority_positive_seeds_required"]
                    for n in positive_by_control.values())
            and mean_exact_delta >= rules["wad_exact_noninferiority_margin"]
        ),
        "secondary_wad_minus_fixed_full_mean_balanced": float(np.mean([
            r["wad_minus_control_mean_balanced"] for r in fixed_full_pairs
        ])),
        "secondary_wad_minus_fixed_full_exact": float(np.mean([
            r["wad_minus_control_exact"] for r in fixed_full_pairs
        ])),
        "test_split_opened": False,
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )

    operations = ["XOR", "NAND", "NOR"]
    fig, axes = plt.subplots(1, len(prereg["conditions"]), figsize=(17, 3.7), sharey=True)
    for ax, condition in zip(axes, prereg["conditions"]):
        matrix = np.zeros((3, 3))
        for oi, operation in enumerate(operations):
            for position in range(3):
                vals = [r["balanced_accuracy"] for r in long_rows
                        if r["condition"] == condition and r["operation"] == operation
                        and r["position"] == position]
                matrix[oi, position] = np.mean(vals)
        im = ax.imshow(matrix, vmin=.5, vmax=1.0, cmap="viridis")
        ax.set_title(condition); ax.set_xticks(range(3)); ax.set_xlabel("position")
        ax.set_yticks(range(3)); ax.set_yticklabels(operations)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i,j] < .72 else "black", fontsize=8)
    fig.colorbar(im, ax=axes, label="seed-mean balanced accuracy", shrink=.8)
    fig.subplots_adjust(wspace=.28, right=.94)
    fig.savefig(OUT / "operation_position_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for condition in prereg["conditions"]:
        sr = [r for r in seed_rows if r["condition"] == condition]
        axes[0].scatter([condition] * len(sr),
                        [r["primary_min_operation_position_balanced"] for r in sr],
                        label=condition)
        axes[1].scatter([r["mean_synaptic_events"] for r in sr],
                        [r["primary_min_operation_position_balanced"] for r in sr],
                        label=condition)
    axes[0].axhline(rules["routing_floor_each_seed"], color="black", ls="--")
    axes[0].tick_params(axis="x", rotation=25); axes[0].set_ylabel("primary min balanced")
    axes[1].set_xlabel("mean synaptic events"); axes[1].set_ylabel("primary min balanced")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "reliability_resource_summary.png", dpi=180)
    plt.close(fig)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

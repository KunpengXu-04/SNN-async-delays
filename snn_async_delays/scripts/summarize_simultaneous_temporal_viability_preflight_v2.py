"""Apply the non-accuracy, mechanism-valid gates for temporal preflight v2."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from scripts.audit_temporal_checkpoint_mechanism import (
    arrival_distributions,
    build_model,
    exhaustive_batch,
    window_gradient_rows,
    window_trace_rows,
)


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "runs/exploratory/simultaneous_temporal_viability_preflight_v2"
OUT = BASE / "docs/generated/simultaneous_temporal_viability_preflight_v2"
REQUIRED = [
    "config.json", "best_model.pt", "last_model.pt", "train_log.csv",
    "validation_results.json", "exhaustive_truth_table_results.json",
    "plots/diagnostic_data.npz", "plots/diagnostic_panel.png",
]


def finite(value) -> bool:
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return True


def window_mass(hist: np.ndarray, cfg: dict) -> list[float]:
    start, width = int(cfg["win_len"]), int(cfg["output_window_len"])
    return [
        float(hist[start + q * width:start + (q + 1) * width].sum())
        for q in range(int(cfg["K"]))
    ]


def main() -> None:
    prereg = yaml.safe_load(
        (BASE / "configs/simultaneous_temporal_viability_preflight_v2.yaml")
        .read_text(encoding="utf-8")
    )
    rows = []
    for seed in prereg["held_out_execution_seeds"]:
        for condition in prereg["conditions"]:
            run = ROOT / f"{condition}_{prereg['endpoint']}_seed{seed}"
            missing = [name for name in REQUIRED if not (run / name).exists()]
            if missing:
                raise SystemExit(f"Incomplete v2 cell {run.name}: {missing}")
            cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
            result = json.loads(
                (run / "validation_results.json").read_text(encoding="utf-8")
            )
            with (run / "train_log.csv").open(encoding="utf-8", newline="") as handle:
                logs = list(csv.DictReader(handle))
            model = build_model(cfg)
            model.load_state_dict(torch.load(
                run / "last_model.pt", map_location="cpu", weights_only=True
            ))
            model.eval()
            spike_input, labels = exhaustive_batch(cfg, "cpu")
            gradients = [
                item for item in window_gradient_rows(model, spike_input, labels, cfg)
                if item["objective"] == "combined"
            ]
            with torch.no_grad():
                _, info = model(spike_input, record=True)
            traces = window_trace_rows(condition, info, labels, cfg)
            _, realized = arrival_distributions(
                model, spike_input, info["hidden_spike_train"]
            )
            final = logs[-1]
            rows.append({
                "condition": condition, "seed": int(seed),
                "artifacts_complete": True,
                "finite": finite(result) and all(
                    finite(float(value)) for row in logs for value in row.values()
                ),
                "loss_reduction": cfg.get("loss_reduction"),
                "output_current_support": [
                    item["output_current_support_fraction"] for item in traces
                ],
                "output_activity": [
                    item["output_activity_fraction"] for item in traces
                ],
                "hidden_emission_support_descriptive": [
                    item["hidden_activity_fraction"] for item in traces
                ],
                "signed_current_class_gap_descriptive": [
                    item["signed_current_class_gap"] for item in traces
                ],
                "realized_arrival_mass": window_mass(realized, cfg),
                "window_output_weight_grad_norm": [
                    item["weight_ho_grad_norm"] for item in gradients
                ],
                "window_total_delay_grad_norm": [
                    item["delay_ih_grad_norm"] + item["delay_ho_grad_norm"]
                    for item in gradients
                ],
                "final_delay_movement": float(final["mean_abs_delay_movement"]),
                "final_delay_saturation_fraction": float(
                    final["delay_saturation_fraction"]
                ),
                "worst_balanced_descriptive": result["worst_query_balanced_accuracy"],
                "per_window_balanced_descriptive": result["per_query_balanced_accuracy"],
                "exact_trial_descriptive": result["exact_trial_accuracy"],
                "output_silent_rate_descriptive": result["output_silent_rate"],
                "output_tie_rate_descriptive": result["output_tie_rate"],
            })

    gates = prereg["locked_viability_gates"]

    def selected(condition: str) -> list[dict]:
        return [row for row in rows if row["condition"] == condition]

    def every_window(condition: str, key: str, threshold: float) -> bool:
        return all(min(row[key]) >= threshold for row in selected(condition))

    checks = {
        "all_cells_complete_and_finite": all(
            row["artifacts_complete"] and row["finite"] for row in rows
        ),
        "all_cells_use_window_class_balanced_loss": all(
            row["loss_reduction"] == "window_class_balanced" for row in rows
        ),
        "scaffold_output_current_support": every_window(
            "temporal_scaffold", "output_current_support",
            gates["scaffold_min_output_current_support_each_window"],
        ),
        "scaffold_output_activity": every_window(
            "temporal_scaffold", "output_activity",
            gates["scaffold_min_output_activity_each_window"],
        ),
        "fixed_full_output_current_support": every_window(
            "fixed_full_support", "output_current_support",
            gates["fixed_full_min_output_current_support_each_window"],
        ),
        "fixed_full_output_activity": every_window(
            "fixed_full_support", "output_activity",
            gates["fixed_full_min_output_activity_each_window"],
        ),
        "fixed_full_window_output_gradient": every_window(
            "fixed_full_support", "window_output_weight_grad_norm",
            gates["fixed_full_min_window_output_weight_grad_norm"],
        ),
        "wad_output_current_support": every_window(
            "w_and_d", "output_current_support",
            gates["wad_min_output_current_support_each_window"],
        ),
        "wad_output_activity": every_window(
            "w_and_d", "output_activity",
            gates["wad_min_output_activity_each_window"],
        ),
        "wad_realized_arrival_mass": every_window(
            "w_and_d", "realized_arrival_mass",
            gates["wad_min_realized_arrival_mass_each_window"],
        ),
        "wad_window_output_gradient": every_window(
            "w_and_d", "window_output_weight_grad_norm",
            gates["wad_min_window_output_weight_grad_norm"],
        ),
        "wad_window_delay_gradient": every_window(
            "w_and_d", "window_total_delay_grad_norm",
            gates["wad_min_window_total_delay_grad_norm"],
        ),
        "wad_delay_movement": all(
            row["final_delay_movement"] >= gates["wad_min_final_delay_movement"]
            for row in selected("w_and_d")
        ),
        "wad_delay_not_saturated": all(
            row["final_delay_saturation_fraction"]
            < gates["wad_max_final_delay_saturation_fraction"]
            for row in selected("w_and_d")
        ),
    }
    decision = {
        "complete": True,
        "cells": rows,
        "checks": checks,
        "preflight_passed": all(checks.values()),
        "accuracy_used_for_decision": False,
        "seed0_used_for_decision": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    flat_rows = []
    for row in rows:
        flat = {
            key: (json.dumps(value) if isinstance(value, list) else value)
            for key, value in row.items()
        }
        flat_rows.append(flat)
    with (OUT / "cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {1: "#1f77b4", 42: "#ff7f0e"}
    positions = {name: i for i, name in enumerate(prereg["conditions"])}
    for row in rows:
        x = positions[row["condition"]] + (-.08 if row["seed"] == 1 else .08)
        axes[0, 0].scatter(x, min(row["output_current_support"]),
                           color=colors[row["seed"]], label=f"seed {row['seed']}")
        axes[0, 1].scatter(x, min(row["output_activity"]), color=colors[row["seed"]])
        axes[1, 1].scatter(x, row["worst_balanced_descriptive"],
                           color=colors[row["seed"]])
        if row["condition"] == "w_and_d":
            axes[1, 0].plot(range(3), row["realized_arrival_mass"], marker="o",
                            color=colors[row["seed"]], label=f"seed {row['seed']}")
    axes[0, 0].axhline(.50, color="black", ls="--", lw=1, label="gate")
    axes[0, 1].axhline(.10, color="black", ls="--", lw=1)
    axes[1, 0].axhline(.05, color="black", ls="--", lw=1, label="gate")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(list(positions), rotation=20, ha="right")
    axes[0, 0].set_title("Minimum output-current support across windows")
    axes[0, 1].set_title("Minimum output activity across windows")
    axes[1, 0].set_title("WAD realized arrival mass by window")
    axes[1, 0].set_xticks(range(3)); axes[1, 0].set_xlabel("window")
    axes[1, 1].set_title("Worst-window balanced accuracy (descriptive)")
    for ax in axes.ravel():
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0, 0].legend(by_label.values(), by_label.keys(), frameon=False)
    axes[1, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "preflight_v2_summary.png", dpi=180)
    plt.close(fig)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

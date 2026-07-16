"""Summarize Phase-0 Stage-A baseline/output-interface calibration."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


BASE = Path(__file__).resolve().parents[1]
PROTOCOL = "spatial_vs_temporal_pareto_phase0"
ROOT = BASE / "runs/exploratory" / PROTOCOL / "stage_a"
OUT = BASE / "docs/generated" / PROTOCOL / "stage_a"
REQUIRED = [
    "config.json", "best_model.pt", "last_model.pt", "train_log.csv",
    "validation_results.json", "exhaustive_truth_table_results.json",
    "plots/diagnostic_data.npz", "plots/diagnostic_panel.png",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    protocol = yaml.safe_load(
        (BASE / "configs/spatial_vs_temporal_pareto_phase0.yaml")
        .read_text(encoding="utf-8")
    )
    stage = protocol["stage_a"]
    paths = sorted(ROOT.rglob("validation_results.json"))
    if len(paths) != int(stage["cells"]):
        raise SystemExit(f"Stage A incomplete: {len(paths)}/{stage['cells']}")

    rows = []
    for path in paths:
        run = path.parent
        missing = [name for name in REQUIRED if not (run / name).exists()]
        if missing:
            raise SystemExit(f"Missing artifacts in {run}: {missing}")
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        val = json.loads(path.read_text(encoding="utf-8"))
        truth = json.loads(
            (run / "exhaustive_truth_table_results.json").read_text(encoding="utf-8")
        )
        ledger = val["resource_ledger"]
        rows.append({
            "hidden_size": int(cfg["n_hidden"]),
            "seed": int(cfg["seed"]),
            "balanced_accuracy": val["worst_query_balanced_accuracy"],
            "exact_trial_accuracy": val["exact_trial_accuracy"],
            "exhaustive_complete": truth["exact_truth_table_completion"],
            "silent_rate": val["output_silent_rate"],
            "collision_rate": val["output_collision_rate"],
            "mean_output_spikes": val["per_query_output_spikes"][0],
            "target_timing_hit_rate": val["target_timing_hit_rate"],
            "mean_abs_target_timing_error_steps": val["mean_abs_target_timing_error_steps"],
            "hidden_spikes": val["mean_hidden_spikes"],
            "neuron_updates": ledger["neuron_updates_per_trial"],
            "dense_synapse_macs": ledger["dense_synapse_macs_per_trial"],
            "synaptic_events": ledger["mean_synaptic_events_total"],
        })

    gates = stage["locked_selection"]
    for row in rows:
        row.update({
            "passes_accuracy": row["balanced_accuracy"] >= gates["balanced_accuracy_each_seed"],
            "passes_truth_table": bool(row["exhaustive_complete"])
            == bool(gates["exhaustive_truth_table_complete_each_seed"]),
            "passes_silence": row["silent_rate"] <= gates["max_silent_rate_each_seed"],
            "passes_collision": row["collision_rate"] <= gates["max_collision_rate_each_seed"],
            "passes_timing_hit": row["target_timing_hit_rate"]
            >= gates["min_target_timing_hit_rate_each_seed"],
            "passes_timing_error": row["mean_abs_target_timing_error_steps"] is not None
            and row["mean_abs_target_timing_error_steps"]
            <= gates["max_mean_abs_target_timing_error_steps_each_seed"],
            "passes_output_spike_count": gates["mean_output_spikes_lower"]
            <= row["mean_output_spikes"] <= gates["mean_output_spikes_upper"],
        })
        row["passes_all_locked_gates"] = all(
            row[key] for key in (
                "passes_accuracy", "passes_truth_table", "passes_silence",
                "passes_collision", "passes_timing_hit", "passes_timing_error",
                "passes_output_spike_count",
            )
        )

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "cells.csv", rows)
    summary = []
    passing = []
    for hidden in stage["hidden_sizes"]:
        hr = [row for row in rows if row["hidden_size"] == hidden]
        passed = all(
            row["balanced_accuracy"] >= gates["balanced_accuracy_each_seed"]
            and bool(row["exhaustive_complete"])
            == bool(gates["exhaustive_truth_table_complete_each_seed"])
            and row["silent_rate"] <= gates["max_silent_rate_each_seed"]
            and row["collision_rate"] <= gates["max_collision_rate_each_seed"]
            and row["target_timing_hit_rate"] >= gates["min_target_timing_hit_rate_each_seed"]
            and row["mean_abs_target_timing_error_steps"] is not None
            and row["mean_abs_target_timing_error_steps"]
            <= gates["max_mean_abs_target_timing_error_steps_each_seed"]
            and gates["mean_output_spikes_lower"] <= row["mean_output_spikes"]
            <= gates["mean_output_spikes_upper"]
            for row in hr
        )
        if passed:
            passing.append(hidden)
        summary.append({
            "hidden_size": hidden,
            "n": len(hr),
            "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in hr])),
            "balanced_accuracy_sd": float(np.std([r["balanced_accuracy"] for r in hr], ddof=1)),
            "exact_trial_mean": float(np.mean([r["exact_trial_accuracy"] for r in hr])),
            "silent_rate_mean": float(np.mean([r["silent_rate"] for r in hr])),
            "collision_rate_mean": float(np.mean([r["collision_rate"] for r in hr])),
            "mean_output_spikes": float(np.mean([r["mean_output_spikes"] for r in hr])),
            "target_timing_hit_rate_mean": float(np.mean([r["target_timing_hit_rate"] for r in hr])),
            "mean_abs_target_timing_error_steps": float(np.mean([
                r["mean_abs_target_timing_error_steps"] for r in hr
                if r["mean_abs_target_timing_error_steps"] is not None
            ])) if any(r["mean_abs_target_timing_error_steps"] is not None for r in hr) else None,
            "passes_all_locked_gates": passed,
        })
    write_csv(OUT / "hidden_summary.csv", summary)
    decision = {
        "protocol": PROTOCOL,
        "stage": "a",
        "complete": True,
        "outcome": "passed" if passing else "failed_no_passing_hidden_size",
        "selected_baseline_hidden_per_query": min(passing) if passing else None,
        "passing_hidden_sizes": passing,
        "cells_passing_all_locked_gates": sum(
            bool(row["passes_all_locked_gates"]) for row in rows
        ),
        "cells_total": len(rows),
        "stage_b_unlocked": bool(passing),
        "test_split_opened": False,
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    panels = [
        (axes[0, 0], "balanced_accuracy", "validation balanced accuracy",
         gates["balanced_accuracy_each_seed"]),
        (axes[0, 1], "target_timing_hit_rate", "target timing hit rate",
         gates["min_target_timing_hit_rate_each_seed"]),
        (axes[1, 0], "silent_rate", "output silent rate",
         gates["max_silent_rate_each_seed"]),
        (axes[1, 1], "collision_rate", "output collision rate",
         gates["max_collision_rate_each_seed"]),
    ]
    colors = plt.cm.tab10.colors
    for seed_index, seed in enumerate(stage["seeds"]):
        seed_rows = sorted(
            (row for row in rows if row["seed"] == seed),
            key=lambda row: row["hidden_size"],
        )
        for ax, metric, _, _ in panels:
            ax.plot(
                [row["hidden_size"] for row in seed_rows],
                [row[metric] for row in seed_rows],
                marker="o", linewidth=1.2, color=colors[seed_index],
                label=f"seed {seed}",
            )
    for ax, _, ylabel, threshold in panels:
        ax.axhline(threshold, color="black", ls="--", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("hidden neurons")
        ax.grid(alpha=.25)
        ax.set_xticks(stage["hidden_sizes"])
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stage_a_reliability_timing.png", dpi=180)
    plt.close(fig)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

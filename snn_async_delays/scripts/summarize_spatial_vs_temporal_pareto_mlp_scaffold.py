"""Summarize and plot the exploratory MLP T-by-hidden Pareto scaffold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from scripts.run_spatial_vs_temporal_pareto_mlp_scaffold import load_protocol


BASE = Path(__file__).resolve().parents[1]
PROTOCOL = "spatial_vs_temporal_pareto_mlp_scaffold_v2"
CONDITION_TITLES = {
    "spatial_independent_d0": "Spatial independent d0\n(total hidden = 2h)",
    "shared_spatial_d0": "Shared spatial d0\n(all-time MLP)",
    "shared_temporal_d0": "Shared temporal d0\n(windowed MLP)",
    "shared_temporal_oracle": "Shared temporal oracle\n(windowed MLP)",
    "shared_temporal_wad": "Shared temporal WAD\n(windowed MLP)",
}
REQUIRED = [
    "config.json", "best_model.pt", "last_model.pt", "train_log.csv",
    "validation_results.json", "exhaustive_truth_table_results.json",
    "resource_ledger.json", "plots/diagnostic_data.npz",
    "plots/diagnostic_panel.png",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_grid(rows: list[dict], condition: str, metric: str,
               widths: list[int], latencies: list[int], reduction: str) -> np.ndarray:
    grid = np.full((len(widths), len(latencies)), np.nan)
    for yi, width in enumerate(widths):
        for xi, latency in enumerate(latencies):
            values = [
                float(row[metric]) for row in rows
                if row["condition"] == condition
                and row["surface_hidden_width"] == width
                and row["total_latency_steps"] == latency
            ]
            if values:
                grid[yi, xi] = (
                    float(np.mean(values)) if reduction == "mean" else float(np.min(values))
                )
    return grid


def _heatmap_figure(rows: list[dict], conditions: list[str], widths: list[int],
                    latencies: list[int], metric: str, reduction: str,
                    path: Path, label: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.3), squeeze=False)
    axes_flat = axes.ravel()
    for ax, condition in zip(axes_flat, conditions):
        grid = _mean_grid(rows, condition, metric, widths, latencies, reduction)
        image = ax.imshow(grid, origin="lower", vmin=0.5, vmax=1.0,
                          cmap="viridis", aspect="auto")
        ax.set_title(CONDITION_TITLES[condition], fontsize=10)
        ax.set_xticks(range(len(latencies)), latencies)
        ax.set_yticks(range(len(widths)), widths)
        ax.set_xlabel("total latency T (steps)")
        ax.set_ylabel("surface h (per-query for spatial; total for shared)")
        for yi in range(len(widths)):
            for xi in range(len(latencies)):
                value = grid[yi, xi]
                if np.isfinite(value):
                    ax.text(xi, yi, f"{value:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if value < .72 else "black")
    axes_flat[-1].axis("off")
    color_axis = axes_flat[-1].inset_axes([.40, .10, .10, .78])
    fig.colorbar(image, cax=color_axis, label=label)
    fig.suptitle(
        f"Exploratory MLP scaffold — {reduction} over seeds\n"
        "Non-spiking decoder; not a formal Stage-B result", fontsize=13,
    )
    fig.subplots_adjust(wspace=.28, hspace=.36, right=.93, top=.87)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points), key=lambda point: (point[0], -point[1]))
    result, best = [], -np.inf
    for resource, accuracy in unique:
        if accuracy > best + 1e-12:
            result.append((resource, accuracy))
            best = accuracy
    return result


def _pareto_figure(rows: list[dict], conditions: list[str], path: Path) -> None:
    resource_fields = [
        ("hidden_neurons_total", "hidden neurons"),
        ("latency_steps", "latency T"),
        ("neuron_updates_per_trial", "neuron updates / trial"),
        ("dense_synapse_macs_per_trial", "dense synapse MACs / trial"),
        ("mean_synaptic_events_total", "measured synaptic events / trial"),
        ("delay_value_storage_elements", "stored delay values"),
    ]
    colors = dict(zip(conditions, plt.cm.tab10.colors[:len(conditions)]))
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))
    for ax, (resource_key, x_label) in zip(axes.ravel(), resource_fields):
        for condition in conditions:
            condition_rows = [row for row in rows if row["condition"] == condition]
            by_point: dict[tuple[int, int], list[dict]] = {}
            for row in condition_rows:
                by_point.setdefault(
                    (row["total_latency_steps"], row["surface_hidden_width"]), []
                ).append(row)
            points = []
            for point_rows in by_point.values():
                resource = float(np.mean([row[resource_key] for row in point_rows]))
                accuracy = float(np.mean([
                    row["worst_query_balanced_accuracy"] for row in point_rows
                ]))
                points.append((resource, accuracy))
            if not points:
                continue
            ax.scatter(
                [point[0] for point in points], [point[1] for point in points],
                s=22, alpha=.58, color=colors[condition], label=condition,
            )
            front = _frontier(points)
            ax.plot([point[0] for point in front], [point[1] for point in front],
                    color=colors[condition], lw=1.5)
        ax.axhline(.90, color="black", ls="--", lw=.8, alpha=.6)
        ax.set_xlabel(x_label)
        ax.set_ylabel("seed-mean worst-query balanced accuracy")
        ax.set_ylim(.45, 1.02)
        ax.grid(alpha=.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, -.01), fontsize=8)
    fig.suptitle(
        "Exploratory reliability–resource frontiers (MLP output; lower resource is better)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-kind", choices=["exploratory", "smoke"],
                        default="exploratory")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    root = BASE / "runs" / args.root_kind / PROTOCOL
    output = BASE / "docs" / "generated" / PROTOCOL / args.root_kind
    output.mkdir(parents=True, exist_ok=True)
    conditions = list(protocol["conditions"])
    latencies = [int(value) for value in protocol["surface"]["total_latency_steps"]]
    widths = [int(value) for value in protocol["surface"]["hidden_widths"]]
    seeds = [int(value) for value in protocol["surface"]["seeds"]]
    expected = int(protocol["surface"]["cells"])

    result_paths = sorted(root.rglob("validation_results.json")) if root.exists() else []
    if args.root_kind == "exploratory" and len(result_paths) != expected and not args.allow_incomplete:
        raise SystemExit(f"incomplete formal exploratory grid: found {len(result_paths)}, expected {expected}")
    rows, incomplete_artifacts = [], []
    for result_path in result_paths:
        run_dir = result_path.parent
        missing = [name for name in REQUIRED if not (run_dir / name).exists()]
        if missing:
            incomplete_artifacts.append({
                "run": str(run_dir.relative_to(BASE)), "missing": missing,
            })
            if not args.allow_incomplete:
                raise SystemExit(f"incomplete artifacts in {run_dir}: {missing}")
            continue
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        ledger = result["resource_ledger"]
        row = {
            "condition": result["condition"],
            "T": int(result["total_latency_steps"]),
            "total_latency_steps": int(result["total_latency_steps"]),
            "surface_hidden_width": int(result["surface_hidden_width"]),
            "hidden_width_semantics": result["hidden_width_semantics"],
            "seed": int(cfg["seed"]),
            "pooled_accuracy": float(result["pooled_accuracy"]),
            "worst_query_accuracy": float(result["worst_query_accuracy"]),
            "worst_query_balanced_accuracy": float(result["worst_query_balanced_accuracy"]),
            "exact_trial_accuracy": float(result["exact_trial_accuracy"]),
            "per_query_balanced_accuracy": json.dumps(result["per_query_balanced_accuracy"]),
            "routing_selectivity_gap": result["routing_selectivity_gap"],
            "mean_hidden_spikes": float(result["mean_hidden_spikes"]),
            "mean_active_hidden_fraction": float(result["mean_active_hidden_fraction"]),
        }
        for field in (
            "hidden_neurons_total", "latency_steps", "neuron_updates_per_trial",
            "dense_synapse_macs_per_trial", "mean_synaptic_events_total",
            "delay_value_storage_elements", "delay_buffer_elements_per_sample",
            "neuron_state_elements_per_sample", "decoder_parameters",
            "decoder_weight_macs_per_trial", "trainable_parameters",
            "trainable_delay_parameters", "model_scalar_storage_elements",
        ):
            row[field] = ledger[field]
        rows.append(row)

    if not rows:
        raise SystemExit("no complete result cells found")
    _write_csv(output / "cells.csv", rows)

    aggregate = []
    for condition in conditions:
        for latency in latencies:
            for width in widths:
                selected = [row for row in rows if row["condition"] == condition
                            and row["total_latency_steps"] == latency
                            and row["surface_hidden_width"] == width]
                if not selected:
                    continue
                aggregate.append({
                    "condition": condition, "T": latency,
                    "surface_hidden_width": width, "n_seeds": len(selected),
                    "worst_balanced_mean": float(np.mean([
                        row["worst_query_balanced_accuracy"] for row in selected
                    ])),
                    "worst_balanced_min_seed": float(np.min([
                        row["worst_query_balanced_accuracy"] for row in selected
                    ])),
                    "exact_trial_mean": float(np.mean([
                        row["exact_trial_accuracy"] for row in selected
                    ])),
                    "pooled_mean": float(np.mean([
                        row["pooled_accuracy"] for row in selected
                    ])),
                    "hidden_neurons_total": float(np.mean([
                        row["hidden_neurons_total"] for row in selected
                    ])),
                    "neuron_updates_per_trial": float(np.mean([
                        row["neuron_updates_per_trial"] for row in selected
                    ])),
                    "dense_synapse_macs_per_trial": float(np.mean([
                        row["dense_synapse_macs_per_trial"] for row in selected
                    ])),
                    "mean_synaptic_events_total": float(np.mean([
                        row["mean_synaptic_events_total"] for row in selected
                    ])),
                })
    _write_csv(output / "surface_summary.csv", aggregate)

    factor_rows = []
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        grid = _mean_grid(
            selected, condition, "worst_query_balanced_accuracy",
            widths, latencies, "mean",
        )
        grand = float(np.mean(grid))
        h_marginal = np.mean(grid, axis=1)
        t_marginal = np.mean(grid, axis=0)
        interaction = grid - h_marginal[:, None] - t_marginal[None, :] + grand
        ss_h = len(latencies) * float(np.sum((h_marginal - grand) ** 2))
        ss_t = len(widths) * float(np.sum((t_marginal - grand) ** 2))
        ss_interaction = float(np.sum(interaction ** 2))
        ss_total = ss_h + ss_t + ss_interaction
        factor_rows.append({
            "condition": condition,
            "hidden_marginal_range": float(np.ptp(h_marginal)),
            "latency_marginal_range": float(np.ptp(t_marginal)),
            "grid_ss_hidden_fraction": ss_h / ss_total if ss_total else 0.0,
            "grid_ss_latency_fraction": ss_t / ss_total if ss_total else 0.0,
            "grid_ss_interaction_fraction": (
                ss_interaction / ss_total if ss_total else 0.0
            ),
        })
    _write_csv(output / "factor_effect_summary.csv", factor_rows)

    robust_points = []
    for item in aggregate:
        selected = [
            row for row in rows
            if row["condition"] == item["condition"]
            and row["T"] == item["T"]
            and row["surface_hidden_width"] == item["surface_hidden_width"]
        ]
        worst_balanced_min = min(
            row["worst_query_balanced_accuracy"] for row in selected
        )
        exact_min = min(row["exact_trial_accuracy"] for row in selected)
        if worst_balanced_min >= .90 and exact_min >= .90:
            first = selected[0]
            robust_points.append({
                "condition": item["condition"], "T": item["T"],
                "surface_hidden_width": item["surface_hidden_width"],
                "worst_balanced_min_seed": worst_balanced_min,
                "exact_trial_min_seed": exact_min,
                **{field: float(np.mean([row[field] for row in selected])) for field in (
                    "hidden_neurons_total", "neuron_updates_per_trial",
                    "dense_synapse_macs_per_trial", "mean_synaptic_events_total",
                    "delay_buffer_elements_per_sample",
                    "delay_value_storage_elements", "decoder_weight_macs_per_trial",
                )},
            })
    _write_csv(output / "robust_feasible_points_90.csv", robust_points)

    chosen = {}
    for condition in {item["condition"] for item in robust_points}:
        candidates = [item for item in robust_points if item["condition"] == condition]
        chosen[condition] = min(
            candidates,
            key=lambda item: (
                item["hidden_neurons_total"], item["T"],
                item["dense_synapse_macs_per_trial"],
            ),
        )
    matched_rows = []
    baseline = chosen.get("spatial_independent_d0")
    if baseline is not None:
        for condition, item in chosen.items():
            matched_rows.append({
                **item,
                **{
                    f"ratio_to_spatial_{field}": item[field] / baseline[field]
                    for field in (
                        "hidden_neurons_total", "neuron_updates_per_trial",
                        "dense_synapse_macs_per_trial", "mean_synaptic_events_total",
                        "delay_buffer_elements_per_sample",
                        "delay_value_storage_elements", "decoder_weight_macs_per_trial",
                    )
                },
            })
        _write_csv(output / "matched_90_resource_comparison.csv", matched_rows)
        plot_fields = [
            ("hidden_neurons_total", "hidden\nneurons"),
            ("neuron_updates_per_trial", "neuron\nupdates"),
            ("dense_synapse_macs_per_trial", "dense\nMACs"),
            ("mean_synaptic_events_total", "measured\nevents"),
            ("delay_buffer_elements_per_sample", "delay-buffer\nelements"),
        ]
        conditions_to_plot = [
            condition for condition in (
                "spatial_independent_d0", "shared_temporal_oracle"
            ) if condition in chosen
        ]
        x = np.arange(len(plot_fields))
        width = .34
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        for index, condition in enumerate(conditions_to_plot):
            values = [
                chosen[condition][field] / baseline[field]
                for field, _ in plot_fields
            ]
            bars = ax.bar(
                x + (index - (len(conditions_to_plot) - 1) / 2) * width,
                values, width, label=condition,
            )
            ax.bar_label(bars, labels=[f"{value:.2f}x" for value in values],
                         padding=3, fontsize=8)
        ax.axhline(1.0, color="black", ls="--", lw=.8)
        ax.set_xticks(x, [label for _, label in plot_fields])
        ax.set_ylabel("resource ratio to spatial baseline")
        ax.set_title(
            "Matched robust reliability ≥0.90 (both seeds, worst-balanced and exact)\n"
            "selected at minimum hidden count then minimum latency"
        )
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(output / "fig_matched_90_resource_comparison.png", dpi=190,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    _heatmap_figure(
        rows, conditions, widths, latencies, "worst_query_balanced_accuracy", "mean",
        output / "figC_nhid_T_plane.png", "worst-query balanced accuracy",
    )
    _heatmap_figure(
        rows, conditions, widths, latencies, "worst_query_balanced_accuracy", "min",
        output / "figC_nhid_T_plane_worst_seed.png", "minimum-seed worst-query balanced accuracy",
    )
    _heatmap_figure(
        rows, conditions, widths, latencies, "exact_trial_accuracy", "mean",
        output / "figC_exact_trial_T_plane.png", "exact-trial accuracy",
    )
    _pareto_figure(rows, conditions, output / "pareto_resource_frontiers.png")

    by_key = {
        (row["condition"], row["T"], row["surface_hidden_width"], row["seed"]): row
        for row in rows
    }
    paired = []
    for latency in latencies:
        for width in widths:
            for seed in seeds:
                d0 = by_key.get(("shared_temporal_d0", latency, width, seed))
                oracle = by_key.get(("shared_temporal_oracle", latency, width, seed))
                wad = by_key.get(("shared_temporal_wad", latency, width, seed))
                if d0 and oracle:
                    paired.append({
                        "T": latency, "surface_hidden_width": width, "seed": seed,
                        "oracle_minus_temporal_d0_worst_balanced": (
                            oracle["worst_query_balanced_accuracy"]
                            - d0["worst_query_balanced_accuracy"]
                        ),
                        "oracle_minus_temporal_d0_exact": (
                            oracle["exact_trial_accuracy"] - d0["exact_trial_accuracy"]
                        ),
                        "wad_minus_temporal_d0_worst_balanced": (
                            wad["worst_query_balanced_accuracy"]
                            - d0["worst_query_balanced_accuracy"] if wad else None
                        ),
                    })
    _write_csv(output / "paired_temporal_controls.csv", paired)

    viability = {}
    for condition in ("spatial_independent_d0", "shared_temporal_oracle"):
        per_seed = {
            seed: any(
                row["condition"] == condition and row["seed"] == seed
                and row["worst_query_balanced_accuracy"] >= .90
                for row in rows
            )
            for seed in seeds
        }
        viability[condition] = {"per_seed_any_surface_point_ge_0p90": per_seed,
                                "all_seeds": all(per_seed.values())}
    decision = {
        "protocol": PROTOCOL,
        "root_kind": args.root_kind,
        "expected_cells": expected if args.root_kind == "exploratory" else None,
        "complete_cells": len(rows),
        "incomplete_artifacts": incomplete_artifacts,
        "spatial_and_oracle_viability": viability,
        "oracle_better_than_temporal_d0_any_matched_cell": any(
            row["oracle_minus_temporal_d0_worst_balanced"] > 0 for row in paired
        ),
        "mean_oracle_minus_temporal_d0_worst_balanced": (
            float(np.mean([row["oracle_minus_temporal_d0_worst_balanced"] for row in paired]))
            if paired else None
        ),
        "robust_feasible_conditions_at_0p90": sorted(chosen),
        "matched_0p90_selected_points": chosen,
        "oracle_area_ratio_to_spatial_at_0p90": (
            chosen["shared_temporal_oracle"]["hidden_neurons_total"]
            / chosen["spatial_independent_d0"]["hidden_neurons_total"]
            if {"shared_temporal_oracle", "spatial_independent_d0"} <= set(chosen)
            else None
        ),
        "direct_spiking_output_solved": False,
        "formal_phase0_stage_b_unlocked": False,
        "publication_claim_allowed": False,
    }
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

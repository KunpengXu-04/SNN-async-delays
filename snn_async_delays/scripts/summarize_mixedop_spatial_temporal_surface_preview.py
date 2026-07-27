"""Aggregate and plot the 72-cell mixed-operation surface preview."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_mixedop_spatial_temporal_surface_preview import (
    BASE, CONDITIONS, PROTOCOL, _complete, _run_directory, build_config,
    grid_specs, load_protocol,
)


LABELS = {
    "spatial_independent_d0": "Spatial independent d0",
    "shared_temporal_oracle": "Oracle — hand-scheduled upper bound",
    "shared_temporal_wad": "WAD — learned-delay diagnostic",
}
COLORS = {
    "spatial_independent_d0": "#4C78A8",
    "shared_temporal_oracle": "#59A14F",
    "shared_temporal_wad": "#E15759",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_rows(K_values: tuple[int, ...] = (5, 8)) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = load_protocol()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in grid_specs(protocol, "formal"):
        if int(spec["K"]) not in K_values:
            continue
        cfg = build_config(protocol, spec)
        run_dir = _run_directory(protocol, cfg)
        if not _complete(run_dir, protocol):
            missing.append(str(run_dir.relative_to(BASE)))
            continue
        result = json.loads((run_dir / "validation_results.json").read_text(encoding="utf-8"))
        ledger = json.loads((run_dir / "resource_ledger.json").read_text(encoding="utf-8"))
        row = {
            "K": cfg["K"], "condition": cfg["condition"],
            "total_hidden_neurons": cfg["surface_total_hidden"],
            "output_window_len": cfg["output_window_len"],
            "T": cfg["T"], "seed": cfg["seed"],
            "worst_query_balanced_accuracy": result["worst_query_balanced_accuracy"],
            "mean_balanced_accuracy": result["balanced_accuracy"],
            "exact_trial_accuracy": result["exact_trial_accuracy"],
            "pooled_accuracy": result["pooled_accuracy"],
            "routing_selectivity_gap": result.get("routing_selectivity_gap"),
            "per_query_balanced_accuracy": result["per_query_balanced_accuracy"],
            "per_output_window_hidden_spikes": result["per_output_window_hidden_spikes"],
            "delay_query_mean_steps": result["delay_query_mean_steps"],
            "delay_query_std_steps": result["delay_query_std_steps"],
            "delay_query_schedule_mae_steps": result["delay_query_schedule_mae_steps"],
            "run_dir": str(run_dir.relative_to(BASE)),
        }
        for key, value in ledger.items():
            if isinstance(value, (int, float)) or value is None:
                row[f"resource_{key}"] = value
        rows.append(row)
    if missing:
        raise SystemExit(f"formal surface incomplete ({len(missing)} cells); first: {missing[0]}")
    expected = 36 * len(K_values)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} immutable cells, found {len(rows)}")
    return protocol, rows


def _matrix(rows: list[dict[str, Any]], K: int, condition: str, metric: str):
    subset = [r for r in rows if r["K"] == K and r["condition"] == condition]
    latencies = sorted({int(r["T"]) for r in subset})
    widths = sorted({int(r["total_hidden_neurons"]) for r in subset})
    matrix = np.full((len(widths), len(latencies)), np.nan)
    for row in subset:
        matrix[widths.index(int(row["total_hidden_neurons"])), latencies.index(int(row["T"]))] = float(row[metric])
    if np.isnan(matrix).any():
        raise ValueError(f"incomplete {K}/{condition}/{metric} matrix")
    return widths, latencies, matrix


def _edges(values: list[int]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mids = (values[:-1] + values[1:]) / 2
    return np.r_[values[0] - (mids[0] - values[0]), mids, values[-1] + (values[-1] - mids[-1])]


def _surface_figure(rows: list[dict[str, Any]], K: int, metric: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for ax, condition in zip(axes, CONDITIONS):
        widths, latencies, matrix = _matrix(rows, K, condition, metric)
        image = ax.pcolormesh(_edges(latencies), _edges(widths), matrix, cmap="viridis", vmin=0.5, vmax=1.0, shading="flat")
        X, Y = np.meshgrid(latencies, widths)
        for y_idx, hidden in enumerate(widths):
            for x_idx, latency in enumerate(latencies):
                value = matrix[y_idx, x_idx]
                ax.text(latency, hidden, f"{value:.2f}", ha="center", va="center", color="white" if value < .76 else "black", fontsize=9, fontweight="bold")
        for level in (0.50, 0.75, 0.90):
            if float(matrix.min()) < level < float(matrix.max()):
                contour = ax.contour(X, Y, matrix, levels=[level], colors="white", linewidths=1.0)
                ax.clabel(contour, fmt={level: f"{level:.2f}"}, fontsize=7)
        x_dense = np.linspace(min(latencies), max(latencies), 200)
        costs = np.linspace(min(latencies) * min(widths), max(latencies) * max(widths), 5)[1:-1]
        for cost in costs:
            y = cost / x_dense
            mask = (y >= min(widths)) & (y <= max(widths))
            ax.plot(x_dense[mask], y[mask], color="white", alpha=.45, lw=.8, ls="--")
        ax.set_title(LABELS[condition], fontsize=10)
        ax.set_xticks(latencies); ax.set_yticks(widths)
        ax.set_xlabel("Total latency T (steps)")
    axes[0].set_ylabel("Total physical hidden neurons")
    assert image is not None
    label = "Worst-query balanced accuracy" if metric == "worst_query_balanced_accuracy" else "Exact-trial accuracy"
    fig.colorbar(image, ax=axes, shrink=.88, label=label)
    fig.suptitle(f"K={K} fixed mixed-operation workload | seed 307 | LR=0.01", fontsize=12)
    fig.savefig(path, dpi=180); plt.close(fig)


def factor_decomposition(matrix: np.ndarray) -> dict[str, Any]:
    grand = float(matrix.mean())
    mean_hidden = matrix.mean(axis=1)
    mean_T = matrix.mean(axis=0)
    hidden_range = float(mean_hidden.max() - mean_hidden.min())
    T_range = float(mean_T.max() - mean_T.min())
    ss_hidden = float(matrix.shape[1] * np.square(mean_hidden - grand).sum())
    ss_T = float(matrix.shape[0] * np.square(mean_T - grand).sum())
    residual = matrix - mean_hidden[:, None] - mean_T[None, :] + grand
    ss_interaction = float(np.square(residual).sum())
    total = ss_hidden + ss_T + ss_interaction
    hidden_fraction = ss_hidden / total if total > 0 else None
    T_fraction = ss_T / total if total > 0 else None
    interaction_fraction = ss_interaction / total if total > 0 else None
    range_gate = hidden_range >= 2.0 * T_range
    ss_gate = hidden_fraction is not None and hidden_fraction >= 0.80
    return {
        "hidden_marginal_accuracy_range": hidden_range,
        "T_marginal_accuracy_range": T_range,
        "grid_mean_accuracy": grand,
        "grid_has_nonzero_variation": bool(total > 0),
        "hidden_to_T_range_ratio": hidden_range / T_range if T_range > 0 else None,
        "SS_hidden": ss_hidden, "SS_T": ss_T, "SS_interaction": ss_interaction,
        "SS_hidden_fraction": hidden_fraction,
        "SS_T_fraction": T_fraction,
        "SS_interaction_fraction": interaction_fraction,
        "hidden_range_at_least_twice_T": range_gate,
        "hidden_SS_fraction_at_least_80_percent": ss_gate,
        "hidden_more_valuable_language_allowed": bool(range_gate and ss_gate),
        "interpretation": (
            "width_more_sensitive_in_this_exploratory_preview"
            if range_gate and ss_gate else "interaction_or_insufficient_evidence"
        ),
    }


def _factor_summary(
    rows: list[dict[str, Any]], output: Path, K_values: tuple[int, ...]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for K in K_values:
        for condition in CONDITIONS:
            _, _, matrix = _matrix(rows, K, condition, "worst_query_balanced_accuracy")
            records.append({"K": K, "condition": condition, **factor_decomposition(matrix)})
    x = np.arange(len(records)); width = .36
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True)
    axes[0].bar(x - width / 2, [r["hidden_marginal_accuracy_range"] for r in records], width, label="Hidden marginal range", color="#4C78A8")
    axes[0].bar(x + width / 2, [r["T_marginal_accuracy_range"] for r in records], width, label="T marginal range", color="#F28E2B")
    axes[0].set_ylabel("Accuracy range"); axes[0].legend(frameon=False); axes[0].grid(axis="y", alpha=.25)
    if not any(r["grid_has_nonzero_variation"] for r in records):
        axes[0].set_ylim(0, .05)
    for index, record in enumerate(records):
        if not record["grid_has_nonzero_variation"]:
            axes[0].text(
                index, .012, f"constant\nacc={record['grid_mean_accuracy']:.2f}",
                ha="center", va="center", fontsize=9,
            )
    bottom = np.zeros(len(records))
    for key, label, color in (
        ("SS_hidden_fraction", "Hidden", "#4C78A8"),
        ("SS_T_fraction", "T", "#F28E2B"),
        ("SS_interaction_fraction", "Interaction", "#B07AA1"),
    ):
        values = np.asarray([
            0.0 if r[key] is None else r[key] for r in records
        ])
        axes[1].bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[1].axhline(.8, color="black", lw=1, ls="--", label="80% hidden-SS gate")
    axes[1].set_ylim(0, 1.04); axes[1].set_ylabel("Two-way grid SS fraction")
    axes[1].set_xticks(x, [f"K{r['K']}\n{r['condition'].replace('shared_temporal_', '').replace('spatial_independent_d0', 'spatial')}" for r in records])
    for index, record in enumerate(records):
        if not record["grid_has_nonzero_variation"]:
            axes[1].text(
                index, .42, "SS undefined\n(no variation)",
                ha="center", va="center", fontsize=9,
            )
    axes[1].legend(ncol=4, frameon=False); axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("Pre-registered descriptive factor comparison (single seed; no inference)")
    fig.savefig(output, dpi=180); plt.close(fig)
    return records


def _pareto(rows: list[dict[str, Any]], path: Path, K_values: tuple[int, ...]) -> None:
    fig, axes = plt.subplots(
        1, len(K_values), figsize=(6 * len(K_values), 4.7), sharey=True,
        constrained_layout=True, squeeze=False,
    )
    for ax, K in zip(axes[0], K_values):
        for condition in CONDITIONS:
            subset = [r for r in rows if r["K"] == K and r["condition"] == condition]
            x = [r["total_hidden_neurons"] * r["T"] for r in subset]
            y = [r["worst_query_balanced_accuracy"] for r in subset]
            ax.scatter(x, y, label=LABELS[condition], color=COLORS[condition], alpha=.8, s=45)
        ax.set_title(f"K={K}"); ax.set_xlabel(r"Neuron-update proxy $N_{hid}T$")
        ax.grid(alpha=.25)
    axes[0, 0].set_ylabel("Worst-query balanced accuracy")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Accuracy versus neuron-update proxy (not hardware energy)")
    fig.savefig(path, dpi=180); plt.close(fig)


def _categorical_heatmaps(
    rows: list[dict[str, Any]], value_key: str, path: Path, title: str,
    K_values: tuple[int, ...],
) -> None:
    fig, axes = plt.subplots(
        len(K_values), 3, figsize=(16, 4 * len(K_values)),
        constrained_layout=True, squeeze=False,
    )
    image = None
    global_values = np.asarray(
        [value for row in rows for value in row[value_key]], dtype=float
    )
    common_vmin, common_vmax = (
        (0.5, 1.0) if "accuracy" in value_key
        else (0.0, max(1.0, float(global_values.max())))
    )
    for row_idx, K in enumerate(K_values):
        for col_idx, condition in enumerate(CONDITIONS):
            ax = axes[row_idx, col_idx]
            subset = sorted(
                [r for r in rows if r["K"] == K and r["condition"] == condition],
                key=lambda r: (r["total_hidden_neurons"], r["T"]),
            )
            values = np.asarray([r[value_key] for r in subset], dtype=float)
            image = ax.imshow(
                values, aspect="auto", cmap="viridis",
                vmin=common_vmin, vmax=common_vmax,
            )
            ax.set_xticks(range(K), [f"Q{q}" for q in range(K)])
            ax.set_yticks(range(len(subset)), [f"N{r['total_hidden_neurons']}/T{r['T']}" for r in subset], fontsize=6)
            ax.set_title(f"K={K} | {LABELS[condition]}", fontsize=9)
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=.7)
    fig.suptitle(title)
    fig.savefig(path, dpi=180); plt.close(fig)


def _routing_heatmaps(
    rows: list[dict[str, Any]], path: Path, K_values: tuple[int, ...]
) -> None:
    fig, axes = plt.subplots(
        len(K_values), 2, figsize=(12, 4 * len(K_values)),
        constrained_layout=True, squeeze=False,
    )
    image = None
    for row_idx, K in enumerate(K_values):
        for col_idx, condition in enumerate(("shared_temporal_oracle", "shared_temporal_wad")):
            ax = axes[row_idx, col_idx]
            subset = sorted([r for r in rows if r["K"] == K and r["condition"] == condition], key=lambda r: (r["total_hidden_neurons"], r["T"]))
            values = np.asarray([r["delay_query_mean_steps"] for r in subset], dtype=float)
            vmax = max(float(values.max()), max(r["output_window_len"] * (K - 1) for r in subset))
            image = ax.imshow(values, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks(range(K), [f"Q{q}" for q in range(K)])
            ax.set_yticks(range(len(subset)), [f"N{r['total_hidden_neurons']}/T{r['T']}" for r in subset], fontsize=6)
            ax.set_title(f"K={K} | {LABELS[condition]}", fontsize=9)
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=.72, label="Mean input→hidden delay (steps)")
    fig.suptitle("Delay–query correspondence; inspect distributions in each cell NPZ/panel")
    fig.savefig(path, dpi=180); plt.close(fig)


def _resource_figure(rows: list[dict[str, Any]], path: Path) -> None:
    metrics = (
        ("resource_dense_synapse_macs_per_trial", "Dense synaptic MACs / trial"),
        ("resource_mean_synaptic_events_total", "Measured synaptic events / trial"),
        ("resource_delay_value_storage_elements", "Delay-value storage (elements)"),
        ("resource_delay_buffer_elements_per_sample", "Delay buffer / sample (elements)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, metrics):
        for condition in CONDITIONS:
            subset = [r for r in rows if r["condition"] == condition]
            ax.scatter([r["total_hidden_neurons"] * r["T"] for r in subset], [r[metric] for r in subset], label=LABELS[condition], color=COLORS[condition], alpha=.72, s=34)
        ax.set_xlabel(r"$N_{hid}T$ proxy"); ax.set_ylabel(label); ax.grid(alpha=.25)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Resource vector — no hardware-energy scalarization")
    fig.savefig(path, dpi=180); plt.close(fig)


def _k8_exclusion_audit(protocol: dict[str, Any], output: Path) -> None:
    root = BASE / protocol["execution"]["formal_root"]
    cells = []
    for config_path in root.glob("*/K8/*/config.json"):
        run_dir = config_path.parent
        cells.append({
            "run_dir": str(run_dir.relative_to(BASE)),
            "run_complete": (run_dir / "run_complete.json").exists(),
            "interruption_audit": (run_dir / "interruption_audit.json").exists(),
            "included_in_K5_gate": False,
        })
    _write_json(output / "k8_exclusion_audit.json", {
        "protocol_id": PROTOCOL,
        "reason": "researcher_requested_K5_first_gate_after_K8_had_started",
        "completed_K8_cells_excluded": sum(cell["run_complete"] for cell in cells),
        "interrupted_K8_cells_excluded": sum(cell["interruption_audit"] for cell in cells),
        "cells": cells,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("k5", "full"), default="full")
    args = parser.parse_args()
    K_values = (5,) if args.stage == "k5" else (5, 8)
    protocol, rows = load_rows(K_values)
    output = BASE / protocol["execution"]["generated_root"]
    if args.stage == "k5":
        output = output / "K5_gate"
    output.mkdir(parents=True, exist_ok=True)
    for K in K_values:
        _surface_figure(rows, K, "worst_query_balanced_accuracy", output / f"K{K}_T_total_hidden_plane.png")
        _surface_figure(rows, K, "exact_trial_accuracy", output / f"K{K}_exact_trial_plane.png")
    factors = _factor_summary(rows, output / "factor_value_summary.png", K_values)
    _pareto(rows, output / "accuracy_cost_pareto.png", K_values)
    _categorical_heatmaps(rows, "per_query_balanced_accuracy", output / "per_query_accuracy_heatmap.png", "Per-query balanced accuracy across the frozen grid", K_values)
    _categorical_heatmaps(rows, "per_output_window_hidden_spikes", output / "hidden_window_activity.png", "Mean hidden spikes in each declared output window", K_values)
    _routing_heatmaps(rows, output / "routing_heatmaps.png", K_values)
    _resource_figure(rows, output / "resource_scaling.png")
    flat_rows = []
    for row in rows:
        flat_rows.append({key: (json.dumps(value) if isinstance(value, list) else value) for key, value in row.items()})
    _write_csv(output / "surface_results.csv", flat_rows)
    _write_json(output / "factor_analysis.json", {
        "protocol_id": PROTOCOL,
        "primary_metric": "worst_query_balanced_accuracy",
        "single_seed": True, "K_values": list(K_values),
        "inferential_statistics_permitted": False,
        "definitions": {
            "hidden_marginal_range": "range of T-averaged accuracy across hidden widths",
            "T_marginal_range": "range of hidden-averaged accuracy across T",
            "two_way_SS": "descriptive balanced-grid sums of squares without replication",
        },
        "decision_rule": "hidden range >= 2*T range AND hidden SS fraction >= 0.80",
        "analyses": factors,
    })
    if args.stage == "k5":
        _k8_exclusion_audit(protocol, output)
    figures = [path.name for path in output.glob("*.png")]
    _write_json(output / "aggregate_complete.json", {
        "protocol_id": PROTOCOL, "stage": args.stage,
        "cells": len(rows), "K_values": list(K_values),
        "figures": sorted(figures),
        "test_split_opened": False,
    })
    print(json.dumps({"protocol": PROTOCOL, "stage": args.stage, "cells": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()

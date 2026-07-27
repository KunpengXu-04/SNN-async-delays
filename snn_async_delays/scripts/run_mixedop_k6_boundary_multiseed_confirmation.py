"""Run the frozen fresh-seed confirmation of the K=6 boundary region."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

import scripts.run_mixedop_k6_centroid_supervised_surface as parent


BASE = parent.BASE
PROTOCOL_ID = "mixedop_k6_boundary_multiseed_confirmation_v1"
CONFIG_PATH = BASE / "configs" / f"{PROTOCOL_ID}.yaml"
PARENT_CONFIG_PATH = BASE / "configs" / "mixedop_k6_centroid_supervised_surface_v1.yaml"


def load_confirmation() -> dict[str, Any]:
    confirmation = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if confirmation.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("confirmation protocol id mismatch")
    observed_hash = hashlib.sha256(PARENT_CONFIG_PATH.read_bytes()).hexdigest().upper()
    if observed_hash != str(confirmation["parent_config_sha256"]).upper():
        raise ValueError(
            f"parent config hash changed: expected {confirmation['parent_config_sha256']}, "
            f"observed {observed_hash}"
        )
    grid = confirmation["frozen_grid"]
    if [10 + 6 * w for w in grid["output_window_lengths"]] != grid["total_latency_steps"]:
        raise ValueError("confirmation w/T mapping changed")
    seeds = list(map(int, grid["fresh_seeds"]))
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("exactly five unique fresh seeds are required")
    if {grid["excluded_parent_seed"], grid["excluded_preflight_seed"]} & set(seeds):
        raise ValueError("confirmation seeds overlap parent evidence")
    return confirmation


def materialize_protocol(confirmation: dict[str, Any]) -> dict[str, Any]:
    protocol = deepcopy(parent.load_protocol())
    grid = confirmation["frozen_grid"]
    protocol.update({
        "protocol_id": PROTOCOL_ID,
        "status": confirmation["status"],
        "study_class": confirmation["study_class"],
        "claim_boundary": confirmation["claim_boundary"],
    })
    protocol["surface"].update({
        "total_hidden_neurons": list(grid["total_hidden_neurons"]),
        "output_window_lengths": list(grid["output_window_lengths"]),
        "total_latency_steps": list(grid["total_latency_steps"]),
        "grid_points": 40,
    })
    protocol["required_cell_artifacts"] = list(confirmation["required_cell_artifacts"])
    protocol["required_aggregate_outputs"] = list(confirmation["required_aggregate_outputs"])
    protocol["execution"]["formal_root"] = confirmation["execution"]["formal_root"]
    protocol["execution"]["generated_root"] = confirmation["execution"]["generated_root"]
    protocol["authorization"]["formal_surface_launch"] = bool(
        confirmation["authorization"]["confirmation_launch"]
    )
    protocol["authorization"]["sealed_test"] = False
    return protocol


def specs(confirmation: dict[str, Any]) -> list[dict[str, Any]]:
    grid = confirmation["frozen_grid"]
    updates = int(confirmation["inherited_recipe"]["optimizer_updates"])
    return [
        parent._spec(n, w, seed, "confirmation", updates)
        for seed, n, w in product(
            grid["fresh_seeds"], grid["total_hidden_neurons"],
            grid["output_window_lengths"],
        )
    ]


def run_dir(confirmation: dict[str, Any], spec: dict[str, Any]) -> Path:
    return parent.run_dir(materialize_protocol(confirmation), spec)


def run_cell(
    confirmation: dict[str, Any], spec: dict[str, Any], device: str,
    dry_run: bool = False,
) -> Path:
    return parent.run_cell(materialize_protocol(confirmation), spec, device, dry_run)


def _cell_rows(confirmation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs(confirmation):
        directory = run_dir(confirmation, spec)
        result = parent._read_json(directory / "validation_results.json")
        activity = result.get(
            "per_output_window_hidden_activity_fraction",
            result.get("per_query_hidden_window_activity_fraction"),
        )
        row: dict[str, Any] = {
            "seed": int(spec["seed"]), "N_hidden": int(spec["total_hidden"]),
            "w": int(spec["output_window_len"]), "T": 10 + 6 * int(spec["output_window_len"]),
            "worst_bacc": float(result["worst_query_balanced_accuracy"]),
            "exact_trial": float(result["exact_trial_accuracy"]),
            "mechanism_valid": bool(result["mechanism_valid"]),
            "pass_90": bool(result["mechanism_valid_90_pass"]),
            "max_centroid_error": float(result["arrival_centroid_max_abs_error_steps"]),
            "minimum_window_activity": float(min(activity)),
            "selected_update": int(result["selected_update"]),
        }
        for q, op in enumerate(parent.OPS):
            row[f"bacc_q{q}_{op}"] = float(result["per_query_balanced_accuracy"][q])
            row[f"activity_q{q}"] = float(activity[q])
        rows.append(row)
    return rows


def observed_n90(rows: list[dict[str, Any]], seed: int, T: int) -> int | None:
    passing = sorted(
        row["N_hidden"] for row in rows
        if row["seed"] == seed and row["T"] == T and row["pass_90"]
    )
    return int(passing[0]) if passing else None


def consensus_n90(
    rows: list[dict[str, Any]], seeds: list[int], T: int, required: int
) -> int | None:
    widths = sorted({int(row["N_hidden"]) for row in rows})
    for width in widths:
        count = sum(
            any(
                row["seed"] == seed and row["T"] == T
                and row["N_hidden"] == width and row["pass_90"]
                for row in rows
            )
            for seed in seeds
        )
        if count >= required:
            return width
    return None


def observed_increases(values: list[int | None]) -> tuple[int, int]:
    increases = 0
    censored = 0
    for left, right in zip(values[:-1], values[1:]):
        if left is None or right is None:
            censored += 1
        elif right > left:
            increases += 1
    return increases, censored


def reversal_present(
    rows: list[dict[str, Any]], seed: int, T: int, lower: int, higher: int
) -> bool:
    lookup = {
        (row["seed"], row["T"], row["N_hidden"]): bool(row["pass_90"])
        for row in rows
    }
    return bool(lookup[(seed, T, lower)] and not lookup[(seed, T, higher)])


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _edges(values: list[int]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    midpoint = (values[:-1] + values[1:]) / 2
    return np.r_[
        values[0] - (midpoint[0] - values[0]), midpoint,
        values[-1] + (values[-1] - midpoint[-1]),
    ]


def _aggregate_plane(
    matrix: np.ndarray, widths: list[int], times: list[int], output: Path,
    title: str, color_label: str, vmin: float = 0.0, vmax: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    mesh = ax.pcolormesh(
        _edges(times), _edges(widths), matrix, shading="flat", cmap="viridis",
        vmin=vmin, vmax=vmax,
    )
    fig.colorbar(mesh, ax=ax, label=color_label)
    for i, width in enumerate(widths):
        for j, T in enumerate(times):
            color = "white" if matrix[i, j] < (vmin + vmax) / 2 else "black"
            ax.text(T, width, f"{matrix[i,j]:.2f}", ha="center", va="center", color=color)
    ax.set(
        xlabel="Simulation duration T (steps)", ylabel="Total hidden neurons N_hid",
        title=title,
    )
    ax.set_xticks(times); ax.set_yticks(widths)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def summarize(confirmation: dict[str, Any]) -> dict[str, Any]:
    rows = _cell_rows(confirmation)
    grid = confirmation["frozen_grid"]
    seeds = list(map(int, grid["fresh_seeds"]))
    widths = list(map(int, grid["total_hidden_neurons"]))
    times = list(map(int, grid["total_latency_steps"]))
    output = BASE / confirmation["execution"]["generated_root"]
    output.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []
    seed_curves: dict[int, list[int | None]] = {}
    seed_nonmonotonic: dict[int, bool] = {}
    for seed in seeds:
        curve = [observed_n90(rows, seed, T) for T in times]
        seed_curves[seed] = curve
        increases, censored = observed_increases(curve)
        seed_nonmonotonic[seed] = increases > 0
        for T, value in zip(times, curve):
            per_seed_rows.append({
                "seed": seed, "T": T, "N90": value,
                "display": ">5" if value is None else str(value),
                "censoring": "right" if value is None else "none",
                "seed_curve_observed_increases": increases,
                "seed_curve_censored_adjacent_comparisons": censored,
            })

    robust = [consensus_n90(rows, seeds, T, 4) for T in times]
    strict = [consensus_n90(rows, seeds, T, 5) for T in times]
    parent_curve = list(map(int, confirmation["boundary_estimands"]["parent_seed_boundary_reported_separately"]))
    robust_increases, robust_censored = observed_increases(robust)

    reversal_rows = []
    reversal_decisions = {}
    threshold = int(confirmation["reversal_reproduced_if_fresh_seeds_minimum"])
    for item in confirmation["registered_reversals"]:
        flags = {
            seed: reversal_present(rows, seed, int(item["T"]), int(item["lower_N"]), int(item["higher_N"]))
            for seed in seeds
        }
        count = sum(flags.values())
        reversal_decisions[item["id"]] = {
            "fresh_seed_flags": flags, "fresh_seeds_reproducing": count,
            "wilson_95_interval": _wilson(count, len(seeds)),
            "registered_reproduction_threshold": threshold,
            "reproduced": count >= threshold,
        }
        reversal_rows.extend({
            "reversal_id": item["id"], "seed": seed, "present": flag,
            "T": item["T"], "lower_N": item["lower_N"], "higher_N": item["higher_N"],
        } for seed, flag in flags.items())

    numeric_ranges = []
    for index, T in enumerate(times):
        values = [seed_curves[seed][index] for seed in seeds]
        numeric = [value for value in values if value is not None]
        numeric_ranges.append({
            "T": T, "values": values,
            "range": max(numeric) - min(numeric) if len(numeric) == len(values) else None,
            "has_censoring": len(numeric) != len(values),
        })
    stable = bool(
        all(value is not None for value in robust)
        and all(item["range"] is not None and item["range"] <= 1 for item in numeric_ranges)
    )
    nonmono_count = sum(seed_nonmonotonic.values())
    nonmono_threshold = int(
        confirmation["nonmonotonicity_decision"]["confirmed_if_fresh_seeds_nonmonotonic_minimum"]
    )

    mean_matrix = np.asarray([
        [np.mean([row["worst_bacc"] for row in rows if row["N_hidden"] == n and row["T"] == T])
         for T in times] for n in widths
    ])
    pass_matrix = np.asarray([
        [np.mean([row["pass_90"] for row in rows if row["N_hidden"] == n and row["T"] == T])
         for T in times] for n in widths
    ])
    _aggregate_plane(
        mean_matrix, widths, times, output / "K6_fresh_seed_mean_worst_bacc.png",
        "Five-fresh-seed mean worst-query balanced accuracy", "mean worst BAcc",
        vmin=float(mean_matrix.min()), vmax=1.0,
    )
    _aggregate_plane(
        pass_matrix, widths, times, output / "K6_fresh_seed_pass_fraction.png",
        "Five-fresh-seed joint pass fraction", "pass fraction",
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for seed in seeds:
        ax.plot(times, [np.nan if value is None else value for value in seed_curves[seed]], "o-", label=str(seed))
    ax.set(xlabel="T (steps)", ylabel="Observed N90", title="Fresh-seed raw N90 curves")
    ax.set_xticks(times); ax.set_yticks(widths); ax.legend(title="seed", ncol=3)
    fig.tight_layout(); fig.savefig(output / "K6_per_seed_N90.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(times, robust, "o-", label="4/5 robust")
    ax.plot(times, strict, "s--", label="5/5 strict")
    ax.plot(times, parent_curve, "x:", label="parent seed 3907 (not pooled)")
    ax.set(xlabel="T (steps)", ylabel="N90", title="Robust, strict and parent boundaries")
    ax.set_xticks(times); ax.set_yticks(widths); ax.legend()
    fig.tight_layout(); fig.savefig(output / "K6_robust_strict_parent_N90.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ids = list(reversal_decisions)
    counts = [reversal_decisions[key]["fresh_seeds_reproducing"] for key in ids]
    ax.bar(ids, counts); ax.axhline(threshold, color="red", linestyle="--", label="registered threshold")
    ax.set(ylabel="fresh seeds reproducing reversal", ylim=(0, 5.5), title="Registered reversal reproduction")
    ax.tick_params(axis="x", rotation=15); ax.legend(); fig.tight_layout()
    fig.savefig(output / "K6_reversal_reproduction.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    box_values = [[seed_curves[seed][i] for seed in seeds if seed_curves[seed][i] is not None] for i in range(len(times))]
    ax.boxplot(box_values, positions=times, widths=[max(2, T * .025) for T in times], manage_ticks=False)
    ax.set(xlabel="T (steps)", ylabel="Fresh-seed N90", title="Boundary variability across fresh seeds")
    ax.set_xticks(times); ax.set_yticks(widths); fig.tight_layout()
    fig.savefig(output / "K6_boundary_variability.png", dpi=180); plt.close(fig)

    parent.core._write_csv(output / "confirmation_cells.csv", rows)
    parent.core._write_csv(output / "per_seed_N90.csv", per_seed_rows)
    parent.core._write_csv(output / "reversal_seed_flags.csv", reversal_rows)
    decision = {
        "protocol_id": PROTOCOL_ID, "cells_expected": 200, "cells_audited": len(rows),
        "all_cells_complete": len(rows) == 200,
        "mechanism_invalid_cells": sum(not row["mechanism_valid"] for row in rows),
        "fresh_seeds": seeds, "parent_seed_excluded_from_confirmation_statistics": True,
        "per_seed_N90": seed_curves,
        "robust_4_of_5_N90": robust, "strict_5_of_5_N90": strict,
        "parent_seed_3907_N90_separate_reference": parent_curve,
        "per_T_seed_ranges": numeric_ranges,
        "boundary_stable_by_registered_rule": stable,
        "robust_boundary_matches_parent_at_all_T": robust == parent_curve,
        "per_seed_nonmonotonic": seed_nonmonotonic,
        "fresh_seeds_nonmonotonic": nonmono_count,
        "nonmonotonicity_confirmation_threshold": nonmono_threshold,
        "nonmonotonicity_confirmed": nonmono_count >= nonmono_threshold,
        "robust_boundary_observed_increases": robust_increases,
        "robust_boundary_censored_adjacent_comparisons": robust_censored,
        "reversal_decisions": reversal_decisions,
        "sealed_test_opened": False,
        "autonomous_WAD_claim_authorized": False,
        "spatial_temporal_pareto_claim_authorized": False,
    }
    parent.core._write_json(output / "boundary_confirmation_decision.json", decision)
    return decision


def preconditions(confirmation: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    if not confirmation["authorization"]["confirmation_launch"]:
        raise RuntimeError("confirmation launch is locked")
    if confirmation["authorization"]["sealed_test"]:
        raise RuntimeError("sealed test must remain closed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    confirmation = load_confirmation()
    preconditions(confirmation, args.dry_run)
    cells = specs(confirmation)
    directories = [run_cell(confirmation, spec, args.device, args.dry_run) for spec in cells]
    if args.dry_run:
        print(json.dumps({"protocol_id": PROTOCOL_ID, "cells": len(cells), "paths": [str(path) for path in directories]}, indent=2))
        return
    print(json.dumps(summarize(confirmation), indent=2))


if __name__ == "__main__":
    main()

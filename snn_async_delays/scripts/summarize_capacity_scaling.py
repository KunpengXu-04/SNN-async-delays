"""Create raw spatial/temporal capacity staircases and preregistered fits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt

from scripts.run_spatial_temporal_capacity_slayer import BASE, load_protocol
from utils.capacity_scaling import fit_power_law, reliability_pass, validate_result_record


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in root.rglob("validation_results.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        validate_result_record(row)
        if row.get("test_split_opened", False):
            raise ValueError(f"sealed-test result found in exploratory scaling: {path}")
        rows.append({**row, "artifact_path": str(path)})
    return rows


def aggregate_cells(
    rows: list[dict[str, Any]], *, resource_key: str, required_seed_passes: int,
    expected_seed_count: int | None = None,
    seed_pass: Callable[[dict[str, Any]], bool] = reliability_pass,
) -> list[dict[str, Any]]:
    expected_seed_count = (
        int(required_seed_passes) if expected_seed_count is None
        else int(expected_seed_count)
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row[resource_key], row["K"])].append(row)
    output = []
    for (condition, resource, K), members in sorted(groups.items()):
        passing = sum(seed_pass(row) for row in members)
        output.append({
            "condition": condition, resource_key: resource, "K": K,
            "seeds_observed": len(members), "seeds_passing": passing,
            "required_seed_passes": required_seed_passes,
            "expected_seed_count": expected_seed_count,
            "pass_reliability": bool(
                len(members) == expected_seed_count and passing >= required_seed_passes
            ),
            "minimum_worst_query_balanced_accuracy": min(
                float(row["worst_query_balanced_accuracy"]) for row in members
            ),
            "minimum_exact_trial_accuracy": min(
                float(row["exact_trial_accuracy"]) for row in members
            ),
        })
    return output


def capacity_rows(
    cells: list[dict[str, Any]], *, resource_key: str,
) -> list[dict[str, Any]]:
    conditions = sorted({row["condition"] for row in cells})
    output = []
    for condition in conditions:
        resources = sorted({row[resource_key] for row in cells if row["condition"] == condition})
        for resource in resources:
            tested = [
                int(row["K"]) for row in cells
                if row["condition"] == condition and row[resource_key] == resource
            ]
            passing = [
                int(row["K"]) for row in cells
                if row["condition"] == condition
                and row[resource_key] == resource and row["pass_reliability"]
            ]
            observed = max(passing) if passing else None
            top_censored = bool(observed is not None and observed == max(tested))
            output.append({
                "condition": condition, resource_key: resource,
                "capacity_observed": observed,
                "capacity": None if top_censored else observed,
                "censoring": (
                    "above_tested_K" if top_censored
                    else "below_tested_K" if observed is None else "none"
                ),
            })
    return output


def inverse_rows(
    cells: list[dict[str, Any]], *, resource_key: str,
) -> list[dict[str, Any]]:
    output = []
    for condition in sorted({row["condition"] for row in cells}):
        for K in sorted({int(row["K"]) for row in cells if row["condition"] == condition}):
            passing = [
                row[resource_key] for row in cells
                if row["condition"] == condition and int(row["K"]) == K
                and row["pass_reliability"]
            ]
            output.append({
                "condition": condition, "K": K,
                f"minimum_passing_{resource_key}": min(passing) if passing else None,
                "right_censored": not bool(passing),
            })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def worst_query_pass(row: dict[str, Any]) -> bool:
    """Pass predicate for the diagnostic single-query endpoint only."""
    return float(row["worst_query_balanced_accuracy"]) >= 0.90


def summarize(
    stage: str, *, reliability_endpoint: str = "joint", artifact_suffix: str = "",
) -> dict[str, Any]:
    if reliability_endpoint not in {"joint", "worst_query_only"}:
        raise ValueError(f"unknown reliability endpoint: {reliability_endpoint}")
    protocol = load_protocol()
    resource_key = "N_hidden_total" if stage == "spatial" else "output_budget_B"
    input_root = BASE / protocol["execution"]["exploratory_root"] / stage
    output = BASE / protocol["execution"]["generated_root"] / stage
    output.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(input_root)
    required = int(protocol["scaling_analysis"].get(
        f"{stage}_required_seed_passes_per_cell",
        protocol["scaling_analysis"]["required_seed_passes_per_cell"],
    ))
    expected = int(protocol["scaling_analysis"].get(
        f"{stage}_expected_seed_count_per_cell",
        protocol["scaling_analysis"]["expected_seed_count_per_cell"],
    ))
    seed_pass = reliability_pass if reliability_endpoint == "joint" else worst_query_pass
    cells = aggregate_cells(
        rows, resource_key=resource_key, required_seed_passes=required,
        expected_seed_count=expected, seed_pass=seed_pass,
    )
    capacities = capacity_rows(cells, resource_key=resource_key)
    inverse = inverse_rows(cells, resource_key=resource_key)
    fits = {}
    for condition in sorted({row["condition"] for row in capacities}):
        selected = [row for row in capacities if row["condition"] == condition]
        fits[condition] = fit_power_law(
            [row[resource_key] for row in selected],
            [row["capacity"] for row in selected],
            minimum_points=int(
                protocol["scaling_analysis"]["minimum_uncensored_points_for_fit"]
            ), seed=5209,
        )
    summary = {
        "protocol_id": protocol["protocol_id"], "stage": stage,
        "resource_key": resource_key, "raw_cells": cells,
        "raw_capacity_staircases": capacities, "inverse_required_resource": inverse,
        "power_law_fits": fits,
        "reliability_endpoint": reliability_endpoint,
        "smoothing_used": False, "censored_values_imputed": False,
    }
    endpoint_suffix = "" if reliability_endpoint == "joint" else "_worst_query_only"
    suffix = endpoint_suffix + artifact_suffix
    (output / f"summary{suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_csv(output / f"cell_seed_gates{suffix}.csv", cells)
    _write_csv(output / f"raw_capacity{suffix}.csv", capacities)
    _write_csv(output / f"inverse_required_resource{suffix}.csv", inverse)

    fig, ax = plt.subplots(figsize=(8, 5))
    for condition in sorted({row["condition"] for row in capacities}):
        selected = [row for row in capacities if row["condition"] == condition]
        x = [row[resource_key] for row in selected if row["capacity_observed"] is not None]
        y = [row["capacity_observed"] for row in selected if row["capacity_observed"] is not None]
        ax.step(x, y, where="post", marker="o", label=condition)
    ax.set(
        xlabel="Total hidden neurons N" if stage == "spatial" else "Output-time budget B",
        ylabel="Raw capacity (max K)",
        title=(
            f"{stage.capitalize()} capacity staircase ({reliability_endpoint}; no smoothing)"
        ),
    )
    if capacities:
        ax.legend(fontsize=8)
    fig.tight_layout()
    figure_name = f"raw_capacity_staircase{suffix}.png"
    fig.savefig(output / figure_name, dpi=180)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["spatial", "temporal"])
    parser.add_argument(
        "--reliability-endpoint", choices=["joint", "worst_query_only"], default="joint",
    )
    parser.add_argument(
        "--artifact-suffix", default="",
        help="Append a version label to generated files without overwriting an earlier summary.",
    )
    arguments = parser.parse_args()
    print(json.dumps(
        summarize(
            arguments.stage, reliability_endpoint=arguments.reliability_endpoint,
            artifact_suffix=arguments.artifact_suffix,
        ), indent=2
    ))

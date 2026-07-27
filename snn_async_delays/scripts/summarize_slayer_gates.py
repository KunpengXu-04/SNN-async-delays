"""Registered validation-only decisions for SLAYER calibration, S1 and S2."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_spatial_temporal_capacity_slayer import BASE, load_protocol


def _results(protocol: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    root = BASE / protocol["execution"]["exploratory_root"] / stage
    rows = []
    for path in root.rglob("validation_results.json"):
        rows.append({
            **json.loads(path.read_text(encoding="utf-8")),
            "_artifact_path": str(path),
        })
    return rows


def summarize_calibration(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = _results(protocol, "s1_calibration")
    expected = len(protocol["gates"]["s1"]["calibration_seeds"])
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        config_path = Path(row["_artifact_path"]).parent / "resolved_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config is None:
            # LR values are encoded in the immutable parent directory name.
            artifact = Path(row["_artifact_path"])
            name = artifact.parent.name if artifact else ""
            lrw = float(name.split("_lrw", 1)[1].split("_lrd", 1)[0])
            lrd = float(name.split("_lrd", 1)[1])
        else:
            lrw, lrd = config["weight_learning_rate"], config["delay_learning_rate"]
        grouped[(float(lrw), float(lrd))].append(row)
    candidates = []
    for (lrw, lrd), members in grouped.items():
        candidates.append({
            "weight_learning_rate": lrw, "delay_learning_rate": lrd,
            "seeds_observed": len(members),
            "passes": sum(bool(row["pass_reliability"]) for row in members),
            "mean_worst_bacc": float(np.mean([
                row["worst_query_balanced_accuracy"] for row in members
            ])),
            "mean_exact_trial": float(np.mean([
                row["exact_trial_accuracy"] for row in members
            ])),
        })
    complete = bool(candidates) and all(row["seeds_observed"] == expected for row in candidates)
    selected = max(
        candidates,
        key=lambda row: (
            row["passes"], row["mean_worst_bacc"], row["mean_exact_trial"],
            -row["weight_learning_rate"], -row["delay_learning_rate"],
        ),
    ) if complete else None
    decision = {
        "protocol_id": protocol["protocol_id"], "stage": "s1_calibration",
        "complete": complete, "passed": complete and selected is not None,
        "selection_uses_validation_only": True, "candidates": candidates,
        "selected_recipe": selected,
    }
    return _write(protocol, "s1_calibration_decision.json", decision)


def summarize_s1(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = _results(protocol, "s1_confirmation")
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    fixed = by_condition["slayer_fixed_interface"]
    task = by_condition["task_only_slayer"]
    fixed_passes = sum(row["pass_reliability"] for row in fixed)
    task_passes = sum(row["pass_reliability"] for row in task)
    calibration_path = (
        BASE / protocol["execution"]["generated_root"]
        / "s1_calibration_decision.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    expected = len(protocol["gates"]["s1"]["confirmation_seeds"])
    fixed_required = int(protocol["gates"]["s1"]["fixed_interface_required_passes"])
    task_required = int(protocol["gates"]["s1"]["task_only_slayer_required_passes"])
    complete = len(fixed) == expected and len(task) == expected
    passed = bool(
        complete and fixed_passes >= fixed_required and task_passes >= task_required
    )
    decision = {
        "protocol_id": protocol["protocol_id"], "stage": "s1_confirmation",
        "complete": complete, "fixed_interface_passes": fixed_passes,
        "task_only_slayer_passes": task_passes, "passed": passed,
        "selected_recipe": calibration["selected_recipe"],
        "s2_authorized_by_results": passed,
    }
    return _write(protocol, "s1_confirmation_decision.json", decision)


def summarize_s2(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row for row in _results(protocol, "s2")
        if row["condition"] == "task_only_slayer"
    ]
    reliable = sum(row["pass_reliability"] for row in rows)
    active = sum(
        min(row["per_window_activity"]["active_fraction"]) > 0.0 for row in rows
    )
    degrading = []
    for row in rows:
        variants = row["delay_intervention_metrics"].get("variants", {})
        degrading.append(max(
            [float(value["worst_bacc_drop"]) for value in variants.values()] or [0.0]
        ))
    required = int(protocol["gates"]["s2"]["intervention_required_degrading_seeds"])
    threshold = float(protocol["gates"]["s2"]["intervention_mean_degradation_minimum"])
    expected = len(protocol["gates"]["s2"]["confirmation_seeds"])
    required_reliable = int(protocol["gates"]["s2"]["task_only_slayer_required_passes"])
    complete = len(rows) == expected
    passed = bool(
        complete and reliable >= required_reliable and active >= required_reliable
        and sum(value > 0 for value in degrading) >= required
        and float(np.mean(degrading)) >= threshold
    )
    decision = {
        "protocol_id": protocol["protocol_id"], "stage": "s2",
        "complete": complete, "reliability_passes": reliable,
        "seeds_with_activity_in_every_window": active,
        "per_seed_best_intervention_worst_bacc_drop": degrading,
        "mean_best_intervention_worst_bacc_drop": (
            float(np.mean(degrading)) if degrading else None
        ),
        "passed": passed, "learned_delay_scaling_authorized_by_results": passed,
    }
    return _write(protocol, "s2_decision.json", decision)


def _write(protocol: dict[str, Any], name: str, decision: dict[str, Any]) -> dict[str, Any]:
    output = BASE / protocol["execution"]["generated_root"]
    output.mkdir(parents=True, exist_ok=True)
    (output / name).write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["s1_calibration", "s1", "s2"])
    args = parser.parse_args(); protocol = load_protocol()
    function = {
        "s1_calibration": summarize_calibration,
        "s1": summarize_s1, "s2": summarize_s2,
    }[args.stage]
    print(json.dumps(function(protocol), indent=2))

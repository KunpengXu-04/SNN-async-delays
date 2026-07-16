"""Apply the non-accuracy viability gates for temporal preflight v1."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "runs/exploratory/simultaneous_temporal_viability_preflight_v1"
OUT = BASE / "docs/generated/simultaneous_temporal_viability_preflight_v1"


def finite(value) -> bool:
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return True


def main() -> None:
    prereg = yaml.safe_load(
        (BASE / "configs/simultaneous_temporal_viability_preflight_v1.yaml")
        .read_text(encoding="utf-8")
    )
    rows = []
    for path in sorted(ROOT.rglob("validation_results.json")):
        run = path.parent
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        result = json.loads(path.read_text(encoding="utf-8"))
        with (run / "train_log.csv").open(encoding="utf-8", newline="") as handle:
            logs = list(csv.DictReader(handle))
        output_grad_fraction = sum(
            float(row.get("output_weight_grad_norm", 0)) > 0 for row in logs
        ) / len(logs)
        delay_grad_fraction = sum(
            float(row.get("delay_grad_norm", 0)) > 0 for row in logs
        ) / len(logs)
        final = logs[-1]
        rows.append({
            "condition": cfg["name"],
            "finite": finite(result) and all(
                finite(float(value)) for row in logs for value in row.values()
            ),
            "hidden_activity_fraction": result["per_query_hidden_window_activity_fraction"],
            "hidden_spikes": result["per_query_hidden_window_spikes"],
            "output_silent_rate": result["output_silent_rate"],
            "output_tie_rate": result["output_tie_rate"],
            "output_gradient_epoch_fraction": output_grad_fraction,
            "delay_gradient_epoch_fraction": delay_grad_fraction,
            "final_delay_movement": float(final["mean_abs_delay_movement"]),
            "final_delay_saturation_fraction": float(final["delay_saturation_fraction"]),
            "worst_balanced_descriptive": result["worst_query_balanced_accuracy"],
            "exact_trial_descriptive": result["exact_trial_accuracy"],
        })

    expected = set(prereg["conditions"])
    found = {row["condition"] for row in rows}
    if found != expected:
        raise SystemExit(f"Incomplete preflight: found {sorted(found)}, expected {sorted(expected)}")
    by_name = {row["condition"]: row for row in rows}
    gates = prereg["locked_viability_gates"]
    checks = {
        "all_cells_complete_and_finite": all(row["finite"] for row in rows),
        "scaffold_window_support": min(by_name["temporal_scaffold"]["hidden_activity_fraction"])
            >= gates["scaffold_min_hidden_activity_fraction_each_window"],
        "scaffold_output_excitable": by_name["temporal_scaffold"]["output_silent_rate"]
            < gates["scaffold_max_output_silent_rate"],
        "fixed_full_window_support": min(by_name["fixed_full_support"]["hidden_activity_fraction"])
            >= gates["fixed_full_min_hidden_activity_fraction_each_window"],
        "fixed_full_output_gradient": by_name["fixed_full_support"]["output_gradient_epoch_fraction"]
            >= gates["fixed_full_min_output_gradient_epoch_fraction"],
        "wad_window_support": min(by_name["w_and_d"]["hidden_activity_fraction"])
            >= gates["wad_min_hidden_activity_fraction_each_window"],
        "wad_output_gradient": by_name["w_and_d"]["output_gradient_epoch_fraction"]
            >= gates["wad_min_output_gradient_epoch_fraction"],
        "wad_delay_gradient": by_name["w_and_d"]["delay_gradient_epoch_fraction"]
            >= gates["wad_min_delay_gradient_epoch_fraction"],
        "wad_delay_movement": by_name["w_and_d"]["final_delay_movement"]
            >= gates["wad_min_final_delay_movement"],
        "wad_delay_not_saturated": by_name["w_and_d"]["final_delay_saturation_fraction"]
            < gates["wad_max_final_delay_saturation_fraction"],
    }
    decision = {"complete": True, "cells": rows, "checks": checks,
                "preflight_passed": all(checks.values()),
                "accuracy_used_for_decision": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

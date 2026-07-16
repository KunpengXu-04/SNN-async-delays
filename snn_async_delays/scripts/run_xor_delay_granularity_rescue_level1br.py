"""Run the preregistered dimension-aware XOR delay rescue (Level 1B-R v1)."""

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
import torch
import yaml

from scripts import run_xor_delay_granularity_level1b as level1b


BASE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "xor_delay_granularity_rescue_level1br_v1"
CONFIG_PATH = BASE / "configs" / f"{PROTOCOL_ID}.yaml"
RUN_ROOT = BASE / "runs" / "exploratory" / PROTOCOL_ID
SMOKE_ROOT = BASE / "runs" / "smoke" / PROTOCOL_ID
SUMMARY_ROOT = BASE / "docs" / "generated" / PROTOCOL_ID


def load_protocol(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"unexpected protocol id: {protocol.get('protocol_id')!r}")
    return protocol


def expected_r1_cells(protocol: dict[str, Any]) -> int:
    stage = protocol["stage_r1_dimension_normalization"]
    return (
        len(stage["conditions"])
        * len(protocol["optimization"]["learned_delay_initial_raw_values"])
        * len(protocol["optimization"]["r1_calibration_seeds"])
    )


def r1_specs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for condition in protocol["stage_r1_dimension_normalization"]["conditions"]:
        for initial_raw in protocol["optimization"]["learned_delay_initial_raw_values"]:
            for seed in protocol["optimization"]["r1_calibration_seeds"]:
                specs.append({
                    "stage": "r1",
                    "condition": str(condition["name"]),
                    "encoding": "single_event",
                    "granularity": str(condition["granularity"]),
                    "delay_tying": str(condition["tying"]),
                    "independent_delay_parameters": int(condition["independent_delay_parameters"]),
                    "normalization_factor": float(condition["normalization_factor"]),
                    "arrival_auxiliary_weight": float(condition["effective_lambda"]),
                    "base_arrival_lambda": float(protocol["losses"]["per_parameter_arrival_centroid"]["base_lambda"]),
                    "arrival_condition": str(condition["name"]),
                    "seed": int(seed),
                    "learned_delay": True,
                    "fixed_delay_steps": None,
                    "target_delay_steps": float(protocol["timing"]["target_delay_steps"]),
                    "initial_raw": float(initial_raw),
                    "weight_learning_rate": float(protocol["optimization"]["weight_learning_rate"]),
                    "delay_learning_rate": float(protocol["optimization"]["delay_learning_rate"]),
                    "full_batch_updates": int(protocol["optimization"]["full_batch_updates"]),
                    "selection_role": "r1_normalization",
                })
    return specs


def _granularity(protocol: dict[str, Any], name: str) -> dict[str, Any]:
    return next(
        item for item in protocol["model"]["delay_granularities"]
        if str(item["name"]) == name
    )


def _dimension_matched_spec_values(
    protocol: dict[str, Any], granularity_name: str
) -> tuple[str, int, float, float]:
    granularity = _granularity(protocol, granularity_name)
    count = int(granularity["independent_delay_parameters"])
    base_lambda = float(protocol["losses"]["per_parameter_arrival_centroid"]["base_lambda"])
    return str(granularity["tying"]), count, float(count), base_lambda * count


def r2_specs(
    protocol: dict[str, Any], r1_decision: dict[str, Any]
) -> list[dict[str, Any]]:
    if not bool(r1_decision.get("r2_authorized")):
        raise RuntimeError("R2 is locked because R1 found no budget-eligible granularity")
    eligible = [str(value) for value in r1_decision.get("r2_eligible_granularities", [])]
    specs: list[dict[str, Any]] = []
    for name in eligible:
        tying, count, factor, effective_lambda = _dimension_matched_spec_values(protocol, name)
        for intervention in protocol["stage_r2_lr_budget_calibration"]["interventions_priority_order"]:
            for initial_raw in protocol["optimization"]["learned_delay_initial_raw_values"]:
                for seed in protocol["optimization"]["r2_budget_calibration_seeds"]:
                    specs.append({
                        "stage": "r2",
                        "condition": str(intervention["name"]),
                        "encoding": "single_event",
                        "granularity": name,
                        "delay_tying": tying,
                        "independent_delay_parameters": count,
                        "normalization_factor": factor,
                        "arrival_auxiliary_weight": effective_lambda,
                        "base_arrival_lambda": float(protocol["losses"]["per_parameter_arrival_centroid"]["base_lambda"]),
                        "arrival_condition": "coordinate_matched",
                        "seed": int(seed),
                        "learned_delay": True,
                        "fixed_delay_steps": None,
                        "target_delay_steps": float(protocol["timing"]["target_delay_steps"]),
                        "initial_raw": float(initial_raw),
                        "weight_learning_rate": float(protocol["optimization"]["weight_learning_rate"]),
                        "delay_learning_rate": float(intervention["delay_learning_rate"]),
                        "full_batch_updates": int(intervention["full_batch_updates"]),
                        "selection_role": "r2_budget_calibration",
                    })
    return specs


def _provisional_recipes(
    r1_decision: dict[str, Any], r2_decision: dict[str, Any] | None
) -> list[dict[str, Any]]:
    recipes = [dict(value) for value in r1_decision.get("provisional_recipes", [])]
    if r2_decision is not None:
        recipes.extend(dict(value) for value in r2_decision.get("provisional_recipes", []))
    unique: dict[str, dict[str, Any]] = {}
    for recipe in recipes:
        unique[str(recipe["granularity"])] = recipe
    return list(unique.values())


def r3_specs(
    protocol: dict[str, Any],
    r1_decision: dict[str, Any],
    r2_decision: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not bool(r1_decision.get("global_anchor_pass")):
        raise RuntimeError("R3 is locked because the R1 global anchor failed")
    if r1_decision.get("r2_eligible_granularities") and r2_decision is None:
        raise RuntimeError("R3 must wait for the complete conditional R2 decision")
    recipes = _provisional_recipes(r1_decision, r2_decision)
    if not recipes:
        raise RuntimeError("R3 is locked because no higher-dimensional recipe is provisional")
    anchor = {
        "granularity": "global",
        "source": "r1_global_anchor",
        "condition": "global_anchor",
        "delay_learning_rate": float(protocol["optimization"]["delay_learning_rate"]),
        "full_batch_updates": int(protocol["optimization"]["full_batch_updates"]),
    }
    specs: list[dict[str, Any]] = []
    for recipe in [anchor, *recipes]:
        name = str(recipe["granularity"])
        tying, count, factor, effective_lambda = _dimension_matched_spec_values(protocol, name)
        condition = "global_anchor" if name == "global" else f"confirm_{name}"
        for initial_raw in protocol["optimization"]["learned_delay_initial_raw_values"]:
            for seed in protocol["optimization"]["r3_sealed_confirmation_seeds"]:
                specs.append({
                    "stage": "r3",
                    "condition": condition,
                    "encoding": "single_event",
                    "granularity": name,
                    "delay_tying": tying,
                    "independent_delay_parameters": count,
                    "normalization_factor": factor,
                    "arrival_auxiliary_weight": effective_lambda,
                    "base_arrival_lambda": float(protocol["losses"]["per_parameter_arrival_centroid"]["base_lambda"]),
                    "arrival_condition": "sealed_confirmation",
                    "seed": int(seed),
                    "learned_delay": True,
                    "fixed_delay_steps": None,
                    "target_delay_steps": float(protocol["timing"]["target_delay_steps"]),
                    "initial_raw": float(initial_raw),
                    "weight_learning_rate": float(protocol["optimization"]["weight_learning_rate"]),
                    "delay_learning_rate": float(recipe["delay_learning_rate"]),
                    "full_batch_updates": int(recipe["full_batch_updates"]),
                    "selection_role": "r3_sealed_confirmation",
                    "recipe_source": str(recipe["source"]),
                })
    return specs


def _token(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def cell_directory(root: Path, spec: dict[str, Any]) -> Path:
    base = root / str(spec["stage"])
    if spec["stage"] == "r1":
        base = base / str(spec["condition"])
    elif spec["stage"] == "r2":
        base = base / str(spec["granularity"]) / str(spec["condition"])
    else:
        base = base / str(spec["condition"]) / f"source_{spec['recipe_source']}"
    return base / f"init_{_token(spec['initial_raw'])}" / f"seed_{spec['seed']}"


def _gradient_summary(array: np.ndarray | None) -> dict[str, float]:
    values = np.asarray([] if array is None else array, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"mean_abs": 0.0, "norm": 0.0}
    return {"mean_abs": float(np.abs(values).mean()), "norm": float(np.linalg.norm(values))}


def run_cell(
    protocol: dict[str, Any], spec: dict[str, Any], *, root: Path, device: str
) -> dict[str, Any]:
    directory = cell_directory(root, spec)
    metrics_path = directory / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    cell_config = {
        "protocol_id": PROTOCOL_ID,
        **spec,
        "operation": "XOR",
        "K": 1,
        "hidden_neurons": int(protocol["model"]["hidden_neurons"]),
        "input_events_per_trial": int(protocol["encodings"]["single_event"]["events_per_trial"]),
        "target_output_step": int(protocol["timing"]["output_target_step"]),
        "checkpoint_selection": "final_only",
        "test_split_opened": False,
    }
    level1b._strict_write_json(directory / "config.json", cell_config)
    model, result = level1b.train_cell(protocol, spec, device=device)
    record = result["final_record"]
    interface = result["final_interface"]
    task_gradient = _gradient_summary(result["initial_gradients"].get("task"))
    arrival_gradient = _gradient_summary(result["initial_gradients"].get("arrival"))
    total_gradient = _gradient_summary(result["initial_gradients"].get("total"))
    effective_lambda = float(spec["arrival_auxiliary_weight"])
    clip_min = float(np.min(result["history"]["clip_coefficient"]))
    metrics = {
        **cell_config,
        **{key: value for key, value in interface.items() if key not in {"predictions", "labels", "valid_pattern_mask"}},
        "initial_delay_mean_steps": float(np.mean(record["initial_independent_delays"])),
        "final_delay_mean_steps": float(np.mean(record["final_independent_delays"])),
        "final_delay_min_steps": float(np.min(record["final_independent_delays"])),
        "final_delay_max_steps": float(np.max(record["final_independent_delays"])),
        "final_delay_max_error_steps": float(result["final_delay_max_error_steps"]),
        "final_delay_fraction_within_tolerance": float(result["final_delay_fraction_within_tolerance"]),
        "initial_task_gradient_mean_abs": task_gradient["mean_abs"],
        "initial_unweighted_arrival_gradient_mean_abs": arrival_gradient["mean_abs"],
        "initial_weighted_arrival_gradient_mean_abs": effective_lambda * arrival_gradient["mean_abs"],
        "initial_total_gradient_mean_abs": total_gradient["mean_abs"],
        "initial_task_gradient_norm": task_gradient["norm"],
        "initial_unweighted_arrival_gradient_norm": arrival_gradient["norm"],
        "initial_weighted_arrival_gradient_norm": effective_lambda * arrival_gradient["norm"],
        "initial_total_gradient_norm": total_gradient["norm"],
        "initial_total_gradient_correct_coordinate_fraction": float(result["initial_gradient_stats"]["direction_fraction"]),
        "initial_total_gradient_nonzero_coordinate_fraction": float(result["initial_gradient_stats"]["nonzero_fraction"]),
        "initial_task_arrival_gradient_conflict_fraction": float(result["initial_task_arrival_gradient_conflict_fraction"]),
        "minimum_clip_coefficient": clip_min,
        "gradient_clipping_flag": bool(clip_min < 0.999),
        "interface_pass": bool(result["interface_pass"]),
        "learned_delay_pass": bool(result["learned_delay_pass"]),
        "complete": True,
    }
    truth = {
        "protocol_id": PROTOCOL_ID,
        "evaluation_split": "exhaustive_truth_table_training_domain",
        "patterns": level1b._truth_records(record),
        "predictions": interface["predictions"],
        "labels": interface["labels"],
        "valid_pattern_mask": interface["valid_pattern_mask"],
        "exact_truth_table_completion": bool(interface["exact_truth_table_completion"]),
        "exact_interface_completion": bool(interface["exact_interface_completion"]),
        "test_split_opened": False,
    }
    torch.save(model.state_dict(), directory / "final_model.pt")
    level1b._strict_write_json(
        directory / "training_log.json",
        [
            {key: float(values[index]) for key, values in result["history"].items()}
            for index in range(len(result["history"]["step"]))
        ],
    )
    level1b._strict_write_json(directory / "exhaustive_truth_table_results.json", truth)
    level1b._strict_write_json(directory / "resource_ledger.json", level1b.resource_ledger(model, spec, record))
    level1b._strict_write_json(metrics_path, metrics)
    plots = directory / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    gradient_arrays = {
        f"initial_{name}_gradient": np.asarray(value if value is not None else [], dtype=np.float64)
        for name, value in result["initial_gradients"].items()
    }
    np.savez_compressed(
        plots / "diagnostic_data.npz",
        **result["history"],
        **record,
        **gradient_arrays,
        normalization_factor=np.asarray(float(spec["normalization_factor"])),
        effective_arrival_lambda=np.asarray(effective_lambda),
    )
    level1b.save_diagnostic_panel(protocol, spec, result, plots / "diagnostic_panel.png")
    return metrics


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_result(rows: list[dict[str, Any]], condition: str, expected: int) -> dict[str, Any]:
    selected = [row for row in rows if str(row["condition"]) == condition]
    return {
        "condition": condition,
        "granularity": selected[0]["granularity"] if selected else None,
        "complete_cells": len(selected),
        "passing_cells": sum(bool(row["learned_delay_pass"]) for row in selected),
        "interface_passing_cells": sum(bool(row["interface_pass"]) for row in selected),
        "all_coordinate_direction_cells": sum(
            float(row["initial_total_gradient_correct_coordinate_fraction"]) == 1.0
            and float(row["initial_total_gradient_nonzero_coordinate_fraction"]) == 1.0
            for row in selected
        ),
        "full_delay_coverage_cells": sum(
            float(row["final_delay_fraction_within_tolerance"]) == 1.0 for row in selected
        ),
        "gradient_clipping_flagged_cells": sum(bool(row["gradient_clipping_flag"]) for row in selected),
        "candidate_pass": len(selected) == expected and all(bool(row["learned_delay_pass"]) for row in selected),
    }


def _r1_plot(candidates: list[dict[str, Any]], output: Path) -> None:
    labels = [str(item["condition"]).replace("_", "\n") for item in candidates]
    x = np.arange(len(labels))
    width = .25
    fig, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    axis.bar(x - width, [item["passing_cells"] for item in candidates], width, label="full gate")
    axis.bar(x, [item["all_coordinate_direction_cells"] for item in candidates], width, label="initial direction")
    axis.bar(x + width, [item["full_delay_coverage_cells"] for item in candidates], width, label="final coverage")
    axis.axhline(10, linestyle="--", color="tab:red", label="10/10 gate")
    axis.set_xticks(x, labels)
    axis.set(ylabel="passing cells", ylim=(0, 10.7), title="Level 1B-R R1 dimension-normalization gates")
    axis.legend(frameon=False, ncol=4)
    axis.grid(axis="y", alpha=.2)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def aggregate_r1(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = SUMMARY_ROOT / "r1"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output / "cells.csv")
    condition_names = [
        str(item["name"]) for item in protocol["stage_r1_dimension_normalization"]["conditions"]
    ]
    candidates = [_candidate_result(rows, name, 10) for name in condition_names]
    by_name = {item["condition"]: item for item in candidates}
    global_pass = bool(by_name["global_anchor"]["candidate_pass"])
    provisional: list[dict[str, Any]] = []
    eligible: list[str] = []
    redesign: list[str] = []
    for granularity, condition in (
        ("per_hidden_neuron", "per_hidden_coordinate_matched"),
        ("per_synapse", "per_synapse_coordinate_matched"),
    ):
        candidate = by_name[condition]
        if candidate["candidate_pass"]:
            provisional.append({
                "granularity": granularity,
                "source": "r1_dimension_matched",
                "condition": condition,
                "delay_learning_rate": float(protocol["optimization"]["delay_learning_rate"]),
                "full_batch_updates": int(protocol["optimization"]["full_batch_updates"]),
            })
        elif candidate["all_coordinate_direction_cells"] == 10:
            eligible.append(granularity)
        else:
            redesign.append(granularity)
    decision = {
        "protocol_id": PROTOCOL_ID,
        "stage": "r1",
        "expected_cells": expected_r1_cells(protocol),
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == expected_r1_cells(protocol),
        "candidate_results": candidates,
        "global_anchor_pass": global_pass,
        "provisional_recipes": provisional if global_pass else [],
        "r2_eligible_granularities": eligible if global_pass else [],
        "loss_redesign_required_granularities": redesign,
        "r2_authorized": bool(global_pass and eligible),
        "r3_direct_authorized": bool(global_pass and provisional and not eligible),
        "test_split_opened": False,
        "microburst_authorized": False,
        "K_greater_than_one_authorized": False,
    }
    level1b._strict_write_json(output / "decision.json", decision)
    _r1_plot(candidates, output / "r1_gate_summary.png")
    return decision


def aggregate_r2(
    protocol: dict[str, Any], rows: list[dict[str, Any]], r1_decision: dict[str, Any]
) -> dict[str, Any]:
    output = SUMMARY_ROOT / "r2"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output / "cells.csv")
    candidates: list[dict[str, Any]] = []
    provisional: list[dict[str, Any]] = []
    interventions = protocol["stage_r2_lr_budget_calibration"]["interventions_priority_order"]
    for granularity in r1_decision.get("r2_eligible_granularities", []):
        selected_recipe = None
        for intervention in interventions:
            condition = str(intervention["name"])
            subset = [row for row in rows if row["granularity"] == granularity and row["condition"] == condition]
            result = {
                **_candidate_result(subset, condition, 6),
                "granularity": granularity,
                "delay_learning_rate": float(intervention["delay_learning_rate"]),
                "full_batch_updates": int(intervention["full_batch_updates"]),
            }
            candidates.append(result)
            if selected_recipe is None and result["candidate_pass"]:
                selected_recipe = {
                    "granularity": granularity,
                    "source": f"r2_{condition}",
                    "condition": condition,
                    "delay_learning_rate": float(intervention["delay_learning_rate"]),
                    "full_batch_updates": int(intervention["full_batch_updates"]),
                }
        if selected_recipe is not None:
            provisional.append(selected_recipe)
    all_provisional = _provisional_recipes(r1_decision, {"provisional_recipes": provisional})
    decision = {
        "protocol_id": PROTOCOL_ID,
        "stage": "r2",
        "expected_cells": len(r1_decision.get("r2_eligible_granularities", [])) * 18,
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == len(r1_decision.get("r2_eligible_granularities", [])) * 18,
        "candidate_results": candidates,
        "provisional_recipes": provisional,
        "all_provisional_recipes": all_provisional,
        "r3_authorized": bool(r1_decision.get("global_anchor_pass") and all_provisional),
        "test_split_opened": False,
    }
    level1b._strict_write_json(output / "decision.json", decision)
    return decision


def aggregate_r3(
    protocol: dict[str, Any], rows: list[dict[str, Any]], recipes: list[dict[str, Any]]
) -> dict[str, Any]:
    output = SUMMARY_ROOT / "r3"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output / "cells.csv")
    conditions = ["global_anchor", *[f"confirm_{item['granularity']}" for item in recipes]]
    candidates = [_candidate_result(rows, condition, 10) for condition in conditions]
    global_pass = next(item for item in candidates if item["condition"] == "global_anchor")["candidate_pass"]
    higher = [item for item in candidates if item["condition"] != "global_anchor"]
    priorities = list(protocol["stage_r3_sealed_confirmation"]["final_selection_priority"])
    passing_names = [str(item["granularity"]) for item in higher if item["candidate_pass"]]
    selected = next((name for name in priorities if name in passing_names), None)
    decision = {
        "protocol_id": PROTOCOL_ID,
        "stage": "r3",
        "expected_cells": len(conditions) * 10,
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == len(conditions) * 10,
        "candidate_results": candidates,
        "global_anchor_confirmation_pass": bool(global_pass),
        "passing_higher_dimensional_granularities": passing_names,
        "selected_rescued_granularity": selected,
        "level1br_pass": bool(global_pass and selected is not None),
        "microburst_authorized": False,
        "K_greater_than_one_authorized": False,
        "test_split_opened": False,
    }
    level1b._strict_write_json(output / "decision.json", decision)
    return decision


def _read_decision(path: Path, error: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(error)
    return json.loads(path.read_text(encoding="utf-8"))


def run_stage(
    protocol: dict[str, Any], *, stage: str, root: Path, device: str, smoke: bool
) -> dict[str, Any]:
    if stage == "r1":
        specs = r1_specs(protocol)
        if smoke:
            specs = [specs[0], specs[-1]]
        rows = [run_cell(protocol, spec, root=root, device=device) for spec in specs]
        return {"smoke_cells": len(rows), "cells": rows} if smoke else aggregate_r1(protocol, rows)
    if smoke:
        r1_decision = {
            "global_anchor_pass": True,
            "r2_authorized": True,
            "r2_eligible_granularities": ["per_hidden_neuron", "per_synapse"],
            "provisional_recipes": [],
        }
    else:
        r1_decision = _read_decision(
            SUMMARY_ROOT / "r1" / "decision.json", "R2/R3 requires the complete R1 decision"
        )
    if stage == "r2":
        specs = r2_specs(protocol, r1_decision)
        if smoke:
            specs = [specs[0], specs[-1]]
        rows = [run_cell(protocol, spec, root=root, device=device) for spec in specs]
        return {"smoke_cells": len(rows), "cells": rows} if smoke else aggregate_r2(protocol, rows, r1_decision)
    if smoke:
        r1_decision = {
            "global_anchor_pass": True,
            "r2_eligible_granularities": [],
            "provisional_recipes": [{
                "granularity": "per_hidden_neuron",
                "source": "synthetic_smoke",
                "condition": "per_hidden_coordinate_matched",
                "delay_learning_rate": 0.01,
                "full_batch_updates": 500,
            }],
        }
        r2_decision = None
    elif r1_decision.get("r2_eligible_granularities"):
        r2_decision = _read_decision(
            SUMMARY_ROOT / "r2" / "decision.json", "R3 must wait for the complete conditional R2 decision"
        )
    else:
        r2_decision = None
    recipes = _provisional_recipes(r1_decision, r2_decision)
    specs = r3_specs(protocol, r1_decision, r2_decision)
    if smoke:
        specs = [specs[0], specs[-1]]
    rows = [run_cell(protocol, spec, root=root, device=device) for spec in specs]
    return {"smoke_cells": len(rows), "cells": rows} if smoke else aggregate_r3(protocol, rows, recipes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["r1", "r2", "r3"], default="r1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    declared_r1 = int(protocol["stage_r1_dimension_normalization"]["grid"]["deterministic_cells"])
    if expected_r1_cells(protocol) != declared_r1:
        raise SystemExit("R1 declared cell count does not match the generated grid")
    if not args.smoke and not args.dry_run:
        status_key = {
            "r1": "stage_r1_dimension_normalization",
            "r2": "stage_r2_lr_budget_calibration",
            "r3": "stage_r3_sealed_confirmation",
        }[args.stage]
        if protocol[status_key]["status"] != "preregistered_ready":
            raise SystemExit(f"{args.stage.upper()} is not launch-ready")
    if args.dry_run:
        formal_cells: int | str = {
            "r1": expected_r1_cells(protocol),
            "r2": f"0-{int(protocol['stage_r2_lr_budget_calibration']['maximum_conditional_cells'])} conditional",
            "r3": (
                f"{int(protocol['stage_r3_sealed_confirmation']['minimum_conditional_cells'])}-"
                f"{int(protocol['stage_r3_sealed_confirmation']['maximum_conditional_cells'])} conditional"
            ),
        }[args.stage]
        print(json.dumps({
            "protocol_id": PROTOCOL_ID,
            "stage": args.stage,
            "formal_cells": formal_cells,
            "test_split_opened": False,
            "microburst_authorized": False,
        }, indent=2))
        return
    result = run_stage(
        protocol,
        stage=args.stage,
        root=SMOKE_ROOT if args.smoke else RUN_ROOT,
        device=args.device,
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

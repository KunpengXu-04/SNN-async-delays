"""Run the preregistered local timing-scaffold withdrawal audit."""

from __future__ import annotations

import argparse
import copy
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
from scripts import run_xor_delay_granularity_rescue_level1br as rescue


BASE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "xor_task_derived_timing_withdrawal_v1"
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


def level1b_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Expose the frozen withdrawal config through the validated Level-1B API."""
    adapted = copy.deepcopy(protocol)
    adapted["encodings"] = {"consecutive_microburst": copy.deepcopy(protocol["encoding"])}
    adapted["losses"] = {
        "filtered_hard_spike": {
            "weight": float(protocol["losses"]["task"]["weight"]),
            "filter_tau_steps": float(protocol["losses"]["task"]["filter_tau_steps"]),
        },
        "per_parameter_arrival_centroid": {
            "base_lambda": float(protocol["losses"]["oracle_foundation_and_positive_control"]["base_lambda"]),
            "target_delay_steps": float(protocol["timing"]["target_delay_steps"]),
        },
    }
    adapted["per_cell_gates"]["learned_delay"] = {
        "max_independent_delay_error_steps": float(protocol["per_cell_gates"]["delay"]["max_independent_delay_error_steps"]),
        "fraction_independent_delays_within_tolerance": float(protocol["per_cell_gates"]["delay"]["fraction_independent_delays_within_tolerance"]),
    }
    return adapted


def expected_cells(protocol: dict[str, Any]) -> dict[str, int]:
    return {
        "w0": int(protocol["stage_w0_foundation"]["cells"]),
        "w1": int(protocol["stage_w1_damage_controls"]["cells"]),
        "w2": int(protocol["stage_w2_withdrawal_and_recovery"]["cells"]),
    }


def foundation_specs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stage": "stage_w0_foundation",
            "condition": "per_hidden_oracle_foundation",
            "encoding": "consecutive_microburst",
            "granularity": "per_hidden_neuron",
            "delay_tying": "post_neuron",
            "independent_delay_parameters": 16,
            "normalization_factor": 16.0,
            "arrival_auxiliary_weight": 0.16,
            "base_arrival_lambda": 0.01,
            "arrival_condition": "dimension_matched_foundation",
            "seed": int(seed),
            "learned_delay": True,
            "fixed_delay_steps": None,
            "target_delay_steps": 4.0,
            "initial_raw": float(protocol["optimization"]["foundation_initial_raw"]),
            "weight_learning_rate": float(protocol["optimization"]["weight_learning_rate"]),
            "delay_learning_rate": float(protocol["optimization"]["delay_learning_rate"]),
            "full_batch_updates": int(protocol["optimization"]["full_batch_updates"]),
            "selection_role": "independent_teacher_built_foundation",
        }
        for seed in protocol["optimization"]["formal_seeds"]
    ]


def branch_specs(protocol: dict[str, Any], *, stage: str) -> list[dict[str, Any]]:
    seeds = [int(seed) for seed in protocol["optimization"]["formal_seeds"]]
    perturbations = [float(value) for value in protocol["timing"]["perturbation_delay_steps"]]
    if stage == "w1":
        return [
            _branch_spec(protocol, seed, "no_update_damage", delay, updates=0)
            for delay in perturbations for seed in seeds
        ]
    specs = [_branch_spec(protocol, seed, "retention_task_delay_only", None) for seed in seeds]
    for delay in perturbations:
        for seed in seeds:
            for condition in ("oracle_delay_only", "task_delay_only", "task_joint", "task_weight_only"):
                specs.append(_branch_spec(protocol, seed, condition, delay))
    return specs


def _branch_spec(
    protocol: dict[str, Any], seed: int, condition: str, perturbation: float | None,
    *, updates: int | None = None,
) -> dict[str, Any]:
    trainable = {
        "no_update_damage": set(),
        "retention_task_delay_only": {"input_hidden_delays"},
        "oracle_delay_only": {"input_hidden_delays"},
        "task_delay_only": {"input_hidden_delays"},
        "task_joint": {"input_hidden_weights", "hidden_output_weights", "input_hidden_delays"},
        "task_weight_only": {"input_hidden_weights", "hidden_output_weights"},
    }[condition]
    oracle = condition == "oracle_delay_only"
    return {
        "stage": "stage_w1_damage_controls" if condition == "no_update_damage" else "stage_w2_withdrawal_recovery",
        "condition": condition,
        "encoding": "consecutive_microburst",
        "granularity": "per_hidden_neuron",
        "delay_tying": "post_neuron",
        "independent_delay_parameters": 16,
        "normalization_factor": 16.0 if oracle else 0.0,
        "arrival_auxiliary_weight": 0.16 if oracle else 0.0,
        "task_loss_weight": 0.0 if oracle else 1.0,
        "seed": seed,
        "learned_delay": True,
        "fixed_delay_steps": None,
        "target_delay_steps": 4.0,
        "initial_raw": float(protocol["optimization"]["foundation_initial_raw"]),
        "functional_delay_override": perturbation,
        "trainable_components": trainable,
        "weight_learning_rate": float(protocol["optimization"]["weight_learning_rate"]),
        "delay_learning_rate": float(protocol["optimization"]["delay_learning_rate"]),
        "full_batch_updates": int(protocol["optimization"]["full_batch_updates"] if updates is None else updates),
        "selection_role": condition,
    }


def _token(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:g}".replace("-", "m").replace(".", "p")


def cell_directory(root: Path, spec: dict[str, Any]) -> Path:
    return (
        root / str(spec["stage"]) / str(spec["condition"])
        / f"perturb_{_token(spec.get('functional_delay_override'))}"
        / f"seed_{spec['seed']}"
    )


def foundation_directory(root: Path, spec: dict[str, Any]) -> Path:
    return root / str(spec["stage"]) / str(spec["condition"]) / f"seed_{spec['seed']}"


def foundation_checkpoint(root: Path, seed: int) -> Path:
    return root / "stage_w0_foundation" / "per_hidden_oracle_foundation" / f"seed_{seed}" / "final_model.pt"


def _strict_json(path: Path, value: Any) -> None:
    level1b._strict_write_json(path, value)


def run_foundation(
    protocol: dict[str, Any], spec: dict[str, Any], *, root: Path, device: str
) -> dict[str, Any]:
    return rescue.run_cell(
        level1b_protocol(protocol), spec, root=root, device=device,
        protocol_id=PROTOCOL_ID, directory_builder=foundation_directory,
    )


def run_branch(
    protocol: dict[str, Any], spec: dict[str, Any], *, root: Path, device: str
) -> dict[str, Any]:
    directory = cell_directory(root, spec)
    metrics_path = directory / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = foundation_checkpoint(root, int(spec["seed"]))
    if not checkpoint.exists():
        raise RuntimeError(f"missing matching-seed W0 checkpoint: {checkpoint}")
    directory.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    adapted = level1b_protocol(protocol)
    model, result = level1b.train_cell(
        adapted, spec, device=device, initial_state_dict=state,
        functional_delay_override=spec.get("functional_delay_override"),
        trainable_components=set(spec["trainable_components"]),
        task_loss_weight=float(spec["task_loss_weight"]),
    )
    record = result["final_record"]
    interface = result["final_interface"]
    tolerance = float(protocol["per_cell_gates"]["delay"]["max_independent_delay_error_steps"])
    initial_delays = np.asarray(record["initial_independent_delays"])
    final_delays = np.asarray(record["final_independent_delays"])
    final_error = np.abs(final_delays - float(protocol["timing"]["target_delay_steps"]))
    initial_stats = result["initial_gradient_stats"]
    condition = str(spec["condition"])
    interface_pass = bool(result["interface_pass"])
    delay_pass = bool(final_error.max() <= tolerance and np.mean(final_error <= tolerance) == 1.0)
    gradient_pass = bool(initial_stats["direction_fraction"] == 1.0 and initial_stats["nonzero_fraction"] == 1.0)
    if condition == "no_update_damage":
        full_pass = not interface_pass
    elif condition == "task_weight_only":
        full_pass = interface_pass
    elif condition == "retention_task_delay_only":
        full_pass = interface_pass and delay_pass
    else:
        full_pass = interface_pass and delay_pass
    config = {
        "protocol_id": PROTOCOL_ID,
        **{key: (sorted(value) if isinstance(value, set) else value) for key, value in spec.items()},
        "parent_checkpoint": str(checkpoint.relative_to(BASE)),
        "optimizer_state_inherited": False,
        "test_split_opened": False,
    }
    metrics = {
        **config,
        **{key: value for key, value in interface.items() if key not in {"predictions", "labels", "valid_pattern_mask"}},
        "initial_interface_exact_patterns": int(result["history"]["exact_interface_patterns"][0]),
        "initial_delay_mean_steps": float(initial_delays.mean()),
        "final_delay_mean_steps": float(final_delays.mean()),
        "final_delay_min_steps": float(final_delays.min()),
        "final_delay_max_steps": float(final_delays.max()),
        "final_delay_max_error_steps": float(final_error.max()),
        "final_delay_fraction_within_tolerance": float(np.mean(final_error <= tolerance)),
        "initial_task_gradient_correct_coordinate_fraction": float(initial_stats["direction_fraction"]),
        "initial_task_gradient_nonzero_coordinate_fraction": float(initial_stats["nonzero_fraction"]),
        "interface_pass": interface_pass,
        "delay_pass": delay_pass,
        "initial_gradient_pass": gradient_pass,
        "full_pass": full_pass,
        "complete": True,
    }
    _strict_json(directory / "config.json", config)
    _strict_json(metrics_path, metrics)
    _strict_json(
        directory / "branch_provenance.json",
        {"parent_checkpoint": config["parent_checkpoint"], "perturbation_delay_steps": spec.get("functional_delay_override"),
         "trainable_components": config["trainable_components"], "arrival_auxiliary_weight": spec["arrival_auxiliary_weight"],
         "task_loss_weight": spec["task_loss_weight"], "fresh_optimizer_state": True},
    )
    _strict_json(
        directory / "training_log.json",
        [{key: float(values[index]) for key, values in result["history"].items()}
         for index in range(len(result["history"]["step"]))],
    )
    truth = {
        "evaluation_split": "exhaustive_truth_table_training_domain",
        "predictions": interface["predictions"], "labels": interface["labels"],
        "valid_pattern_mask": interface["valid_pattern_mask"],
        "exact_truth_table_completion": bool(interface["exact_truth_table_completion"]),
        "exact_interface_completion": bool(interface["exact_interface_completion"]),
        "test_split_opened": False,
    }
    _strict_json(directory / "exhaustive_truth_table_results.json", truth)
    _strict_json(directory / "resource_ledger.json", level1b.resource_ledger(model, spec, record))
    torch.save(model.state_dict(), directory / "final_model.pt")
    plots = directory / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    gradients = {f"initial_{name}_gradient": np.asarray(value if value is not None else [], dtype=np.float64)
                 for name, value in result["initial_gradients"].items()}
    np.savez_compressed(plots / "diagnostic_data.npz", **result["history"], **record, **gradients)
    level1b.save_diagnostic_panel(adapted, spec, result, plots / "diagnostic_panel.png")
    return metrics


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(protocol: dict[str, Any], stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = SUMMARY_ROOT / f"stage_{stage}"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output / "cells.csv")
    if stage == "w0":
        gate = len(rows) == 5 and all(bool(row["learned_delay_pass"]) for row in rows)
        decision = {"protocol_id": PROTOCOL_ID, "stage": stage, "complete_cells": len(rows),
                    "foundation_passing_cells": sum(bool(row["learned_delay_pass"]) for row in rows),
                    "foundation_gate_pass": gate, "w1_authorized": gate, "w2_authorized": False}
    elif stage == "w1":
        gate = len(rows) == 10 and all(bool(row["full_pass"]) for row in rows)
        decision = {"protocol_id": PROTOCOL_ID, "stage": stage, "complete_cells": len(rows),
                    "damaged_cells": sum(bool(row["full_pass"]) for row in rows),
                    "damage_gate_pass": gate, "w2_authorized": gate}
    else:
        names = ["retention_task_delay_only", "oracle_delay_only", "task_delay_only", "task_joint", "task_weight_only"]
        candidates = []
        for name in names:
            selected = [row for row in rows if row["condition"] == name]
            candidates.append({"condition": name, "complete_cells": len(selected),
                               "passing_cells": sum(bool(row["full_pass"]) for row in selected),
                               "all_coordinate_initial_direction_cells": sum(
                                   bool(row["initial_gradient_pass"]) for row in selected
                               ),
                               "candidate_pass": len(selected) == (5 if name.startswith("retention") else 10)
                               and all(bool(row["full_pass"]) for row in selected)})
        by_name = {item["condition"]: item for item in candidates}
        directional = []
        for name in names:
            perturbations = [None] if name == "retention_task_delay_only" else [3.0, 5.0]
            for perturbation in perturbations:
                selected = [
                    row for row in rows
                    if row["condition"] == name
                    and row.get("functional_delay_override") == perturbation
                ]
                directional.append({
                    "condition": name,
                    "perturbation_delay_steps": perturbation,
                    "complete_cells": len(selected),
                    "interface_passing_cells": sum(bool(row["interface_pass"]) for row in selected),
                    "delay_passing_cells": sum(bool(row["delay_pass"]) for row in selected),
                    "full_passing_cells": sum(bool(row["full_pass"]) for row in selected),
                    "mean_exact_target_spike_train_matches": (
                        float(np.mean([row["exact_target_spike_train_matches"] for row in selected]))
                        if selected else None
                    ),
                    "mean_correct_target_time_rate": (
                        float(np.mean([row["correct_target_time_rate"] for row in selected]))
                        if selected else None
                    ),
                    "mean_final_delay_fraction_within_tolerance": (
                        float(np.mean([row["final_delay_fraction_within_tolerance"] for row in selected]))
                        if selected else None
                    ),
                })
        protocol_pass = bool(by_name["retention_task_delay_only"]["candidate_pass"]
                             and by_name["oracle_delay_only"]["candidate_pass"]
                             and by_name["task_delay_only"]["candidate_pass"])
        decision = {"protocol_id": PROTOCOL_ID, "stage": stage, "complete_cells": len(rows),
                    "candidate_results": candidates, "directional_results": directional,
                    "local_restoration_protocol_pass": protocol_pass,
                    "global_from_scratch_discovery_supported": False, "K_greater_than_one_authorized": False}
        _summary_plot(candidates, output / "withdrawal_recovery_summary.png")
        _directional_summary_plot(directional, output / "withdrawal_directional_endpoints.png")
    _strict_json(output / "decision.json", decision)
    return decision


def _summary_plot(candidates: list[dict[str, Any]], output: Path) -> None:
    labels = [item["condition"].replace("_", "\n") for item in candidates]
    values = [item["passing_cells"] for item in candidates]
    totals = [item["complete_cells"] for item in candidates]
    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    bars = axis.bar(range(len(labels)), values)
    for bar, value, total in zip(bars, values, totals):
        axis.text(bar.get_x() + bar.get_width() / 2, value + .15, f"{value}/{total}", ha="center")
    axis.set_xticks(range(len(labels)), labels)
    axis.set(title="Scaffold withdrawal and local restoration gates", ylabel="passing cells", ylim=(0, 11))
    axis.grid(axis="y", alpha=.2)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _directional_summary_plot(rows: list[dict[str, Any]], output: Path) -> None:
    labels = []
    interface = []
    delay = []
    for row in rows:
        suffix = "d4" if row["perturbation_delay_steps"] is None else f"d{int(row['perturbation_delay_steps'])}"
        short = {
            "retention_task_delay_only": "retention",
            "oracle_delay_only": "oracle",
            "task_delay_only": "task-delay",
            "task_joint": "task-joint",
            "task_weight_only": "weight-only",
        }[row["condition"]]
        labels.append(f"{short}\n{suffix}")
        interface.append(row["interface_passing_cells"])
        delay.append(row["delay_passing_cells"])
    x = np.arange(len(labels))
    width = 0.38
    fig, axis = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
    axis.bar(x - width / 2, interface, width, label="exact interface")
    axis.bar(x + width / 2, delay, width, label="all 16 delays within 0.1")
    axis.axhline(5, linestyle="--", color="tab:red", label="5/5 directional gate")
    axis.set_xticks(x, labels)
    axis.set(
        title="W2 directional functional and parameter endpoints",
        ylabel="passing cells",
        ylim=(0, 5.6),
    )
    axis.grid(axis="y", alpha=.2)
    axis.legend(frameon=False, ncol=3)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _decision(stage: str, root: Path) -> dict[str, Any]:
    path = SUMMARY_ROOT / f"stage_{stage}" / "decision.json"
    if not path.exists():
        raise RuntimeError(f"missing formal {stage} decision: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_stage(protocol: dict[str, Any], stage: str, *, root: Path, device: str, smoke: bool) -> dict[str, Any]:
    if stage == "w0":
        specs = foundation_specs(protocol)
        if smoke:
            specs = [{**specs[0], "seed": int(protocol["execution_policy"]["smoke_seeds"][0]), "full_batch_updates": 2}]
        rows = [run_foundation(protocol, spec, root=root, device=device) for spec in specs]
    else:
        if not smoke:
            prior = _decision("w0" if stage == "w1" else "w1", root)
            gate = prior["w1_authorized"] if stage == "w1" else prior["w2_authorized"]
            if not gate:
                raise RuntimeError(f"{stage} is locked by the prior formal decision")
        specs = branch_specs(protocol, stage=stage)
        if smoke:
            smoke_seed = int(protocol["execution_policy"]["smoke_seeds"][0])
            specs = [{**spec, "seed": smoke_seed, "full_batch_updates": min(2, int(spec["full_batch_updates"]))}
                     for index, spec in enumerate(specs) if index == 0 or (stage == "w2" and spec["condition"] in
                     {"retention_task_delay_only", "oracle_delay_only", "task_delay_only", "task_joint", "task_weight_only"})]
            unique = {}
            for spec in specs:
                unique.setdefault((spec["condition"], spec.get("functional_delay_override")), spec)
            specs = list(unique.values())
        rows = [run_branch(protocol, spec, root=root, device=device) for spec in specs]
    return {"smoke_cells": len(rows), "cells": rows} if smoke else aggregate(protocol, stage, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["w0", "w1", "w2"], default="w0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    counts = expected_cells(protocol)
    generated = {"w0": len(foundation_specs(protocol)), "w1": len(branch_specs(protocol, stage="w1")),
                 "w2": len(branch_specs(protocol, stage="w2"))}
    if counts != generated:
        raise SystemExit(f"declared cell counts do not match grids: {counts} vs {generated}")
    if args.dry_run:
        if args.stage == "w0":
            authorized = bool(protocol["authorization"]["w0_launch"])
        elif args.stage == "w1":
            path = SUMMARY_ROOT / "stage_w0" / "decision.json"
            authorized = path.exists() and bool(
                json.loads(path.read_text(encoding="utf-8")).get("w1_authorized")
            )
        else:
            path = SUMMARY_ROOT / "stage_w1" / "decision.json"
            authorized = path.exists() and bool(
                json.loads(path.read_text(encoding="utf-8")).get("w2_authorized")
            )
        print(json.dumps({"protocol_id": PROTOCOL_ID, "stage": args.stage,
                          "formal_cells": counts[args.stage], "authorized": authorized,
                          "stage_locked": not authorized,
                          "test_split_opened": False, "K_greater_than_one_authorized": False}, indent=2))
        return
    result = run_stage(protocol, args.stage, root=SMOKE_ROOT if args.smoke else RUN_ROOT,
                       device=args.device, smoke=args.smoke)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

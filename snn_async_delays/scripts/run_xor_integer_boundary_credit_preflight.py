"""Run the preregistered P0 delay-backward estimator screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from scripts import run_xor_delay_granularity_level1b as level1b
from scripts import run_xor_task_derived_timing_withdrawal as withdrawal


BASE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "xor_integer_boundary_credit_preflight_v1"
CONFIG = BASE / "configs" / f"{PROTOCOL_ID}.yaml"
RUN_ROOT = BASE / "runs" / "exploratory" / PROTOCOL_ID / "stage_p0_gradient_screen"
SUMMARY_ROOT = BASE / "docs" / "generated" / PROTOCOL_ID / "stage_p0_gradient_screen"
P1_RUN_ROOT = BASE / "runs" / "exploratory" / PROTOCOL_ID / "stage_p1_conditional_recovery"
P1_SUMMARY_ROOT = BASE / "docs" / "generated" / PROTOCOL_ID / "stage_p1_conditional_recovery"
W0_ROOT = (
    BASE / "runs" / "exploratory" / "xor_task_derived_timing_withdrawal_v1"
    / "stage_w0_foundation" / "per_hidden_oracle_foundation"
)


def load_config() -> dict[str, Any]:
    with CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"unexpected protocol id: {config.get('protocol_id')!r}")
    return config


def _hash_record(record: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("input", "hidden", "output", "output_pre_reset", "output_current"):
        array = np.ascontiguousarray(record[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _initial_state(
    adapted: dict[str, Any], spec: dict[str, Any], *, seed: int, device: str
) -> dict[str, torch.Tensor]:
    model = level1b.build_model(adapted, spec)
    destination = model.state_dict()
    parent_path = W0_ROOT / f"seed_{seed}" / "final_model.pt"
    if not parent_path.exists():
        raise FileNotFoundError(f"missing W0 checkpoint: {parent_path}")
    parent = torch.load(parent_path, map_location=device, weights_only=True)
    destination.update({
        key: value for key, value in parent.items()
        if key in destination and destination[key].shape == value.shape and "delay_raw" not in key
    })
    return destination


def run_probes(config: dict[str, Any], *, device: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    parent = withdrawal.load_protocol()
    adapted = withdrawal.level1b_protocol(parent)
    rows: list[dict[str, Any]] = []
    gradient_arrays: list[np.ndarray] = []
    target = float(config["scope"]["target_delay_steps"])

    for estimator in config["estimators"]:
        for granularity in config["delay_granularities"]:
            for delay in config["scope"]["perturbations_steps"]:
                for seed in config["p0_gradient_screen"]["seeds"]:
                    spec = withdrawal._branch_spec(
                        parent, int(seed), "task_delay_only", float(delay), updates=0
                    )
                    spec.update({
                        "granularity": str(granularity["id"]),
                        "delay_tying": str(granularity["delay_tying"]),
                        "independent_delay_parameters": int(granularity["independent_delay_parameters"]),
                        "delay_gradient_mode": str(estimator["delay_gradient_mode"]),
                        "delay_gradient_sigma": float(estimator["delay_gradient_sigma"]),
                    })
                    state = _initial_state(adapted, spec, seed=int(seed), device=device)
                    _, result = level1b.train_cell(
                        adapted,
                        spec,
                        device=device,
                        initial_state_dict=state,
                        functional_delay_override=float(delay),
                        trainable_components={"input_hidden_delays"},
                        task_loss_weight=1.0,
                    )
                    gradient = np.asarray(result["initial_gradients"]["task"], dtype=np.float64).reshape(-1)
                    direction = gradient * (float(delay) - target) > 0.0
                    nonzero = np.abs(gradient) > 1e-10
                    gradient_arrays.append(gradient)
                    rows.append({
                        "estimator_id": str(estimator["id"]),
                        "delay_gradient_mode": str(estimator["delay_gradient_mode"]),
                        "delay_gradient_sigma": float(estimator["delay_gradient_sigma"]),
                        "intervention_rank": int(estimator["intervention_rank"]),
                        "granularity": str(granularity["id"]),
                        "delay_tying": str(granularity["delay_tying"]),
                        "independent_delay_parameters": int(granularity["independent_delay_parameters"]),
                        "perturbation_delay_steps": float(delay),
                        "seed": int(seed),
                        "raw_gradient_mean": float(gradient.mean()),
                        "raw_gradient_min": float(gradient.min()),
                        "raw_gradient_max": float(gradient.max()),
                        "mean_gradient_target_directed": bool(gradient.mean() * (float(delay) - target) > 0.0),
                        "target_directed_coordinate_fraction": float(direction.mean()),
                        "nonzero_coordinate_fraction": float(nonzero.mean()),
                        "gradient_norm": float(np.linalg.norm(gradient)),
                        "exact_interface_patterns": int(result["history"]["exact_interface_patterns"][0]),
                        "forward_record_sha256": _hash_record(result["final_record"]),
                        "updates": 0,
                        "test_split_opened": False,
                    })
    width = max(array.size for array in gradient_arrays)
    padded = np.full((len(gradient_arrays), width), np.nan, dtype=np.float64)
    for index, array in enumerate(gradient_arrays):
        padded[index, :array.size] = array
    return rows, padded


def summarize(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, str, float], set[str]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["granularity"]), float(row["perturbation_delay_steps"]))
        groups.setdefault(key, set()).add(str(row["forward_record_sha256"]))
    mismatches = [
        {"seed": key[0], "granularity": key[1], "delay": key[2], "hash_count": len(hashes)}
        for key, hashes in groups.items() if len(hashes) != 1
    ]

    estimator_results = []
    legacy_d5 = np.mean([
        float(row["target_directed_coordinate_fraction"])
        for row in rows
        if row["estimator_id"] == "legacy_right_linear"
        and row["granularity"] == "per_hidden"
        and float(row["perturbation_delay_steps"]) == 5.0
    ])
    for estimator in config["estimators"]:
        estimator_id = str(estimator["id"])
        selected = [row for row in rows if row["estimator_id"] == estimator_id]
        global_rows = [row for row in selected if row["granularity"] == "global"]
        per_hidden_rows = [row for row in selected if row["granularity"] == "per_hidden"]
        d5_per_hidden = [
            row for row in per_hidden_rows if float(row["perturbation_delay_steps"]) == 5.0
        ]
        checks = {
            "global_mean_direction_all": all(bool(row["mean_gradient_target_directed"]) for row in global_rows),
            "global_nonzero_all": all(float(row["nonzero_coordinate_fraction"]) == 1.0 for row in global_rows),
            "per_hidden_mean_direction_all": all(bool(row["mean_gradient_target_directed"]) for row in per_hidden_rows),
            "per_hidden_nonzero_min_ge_0p75": min(
                float(row["nonzero_coordinate_fraction"]) for row in per_hidden_rows
            ) >= 0.75,
            "d5_direction_fraction_above_legacy": bool(np.mean([
                float(row["target_directed_coordinate_fraction"]) for row in d5_per_hidden
            ]) > legacy_d5),
        }
        eligible = bool(
            estimator_id != "legacy_right_linear"
            and not mismatches
            and all(checks.values())
        )
        estimator_results.append({
            "estimator_id": estimator_id,
            "delay_gradient_mode": estimator["delay_gradient_mode"],
            "delay_gradient_sigma": float(estimator["delay_gradient_sigma"]),
            "intervention_rank": int(estimator["intervention_rank"]),
            "global_target_directed_probes": int(sum(bool(row["mean_gradient_target_directed"]) for row in global_rows)),
            "global_probes": len(global_rows),
            "per_hidden_mean_target_directed_probes": int(sum(bool(row["mean_gradient_target_directed"]) for row in per_hidden_rows)),
            "per_hidden_probes": len(per_hidden_rows),
            "per_hidden_min_nonzero_coordinate_fraction": float(min(float(row["nonzero_coordinate_fraction"]) for row in per_hidden_rows)),
            "d3_per_hidden_mean_direction_fraction": float(np.mean([
                float(row["target_directed_coordinate_fraction"]) for row in per_hidden_rows
                if float(row["perturbation_delay_steps"]) == 3.0
            ])),
            "d5_per_hidden_mean_direction_fraction": float(np.mean([
                float(row["target_directed_coordinate_fraction"]) for row in d5_per_hidden
            ])),
            "eligibility_checks": checks,
            "eligible": eligible,
        })
    eligible = sorted(
        (result for result in estimator_results if result["eligible"]),
        key=lambda result: int(result["intervention_rank"]),
    )
    winner = None if not eligible else str(eligible[0]["estimator_id"])
    return {
        "protocol_id": PROTOCOL_ID,
        "stage": "stage_p0_gradient_screen",
        "complete": True,
        "probe_count": len(rows),
        "expected_probe_count": int(config["p0_gradient_screen"]["probes"]),
        "forward_equivalence_pass": not mismatches,
        "forward_equivalence_mismatches": mismatches,
        "legacy_d5_per_hidden_mean_direction_fraction": float(legacy_d5),
        "estimator_results": estimator_results,
        "selected_estimator_id": winner,
        "p1_unlocked_by_rule": winner is not None,
        "test_split_opened": False,
        "claim_boundary": "exploratory_mechanism_and_engineering_selection_only",
    }


def write_artifacts(
    config: dict[str, Any], rows: list[dict[str, Any]], gradients: np.ndarray,
    summary: dict[str, Any]
) -> None:
    completion_marker = RUN_ROOT / "run_complete.json"
    if completion_marker.exists():
        raise FileExistsError("immutable completed P0 output already exists; refusing overwrite")
    # A directory without the completion marker is an interrupted transaction,
    # not a formal result.  Reuse it so failed artifact serialization is safely
    # recoverable without weakening immutability after completion.
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG, RUN_ROOT / "preregistered_protocol.yaml")
    with (RUN_ROOT / "gradient_probes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (RUN_ROOT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _selected_estimator(config: dict[str, Any]) -> dict[str, Any]:
    selected_id = str(config["observed_stage_decisions"]["selected_estimator_id"])
    matches = [item for item in config["estimators"] if str(item["id"]) == selected_id]
    if len(matches) != 1:
        raise RuntimeError(f"selected estimator {selected_id!r} is not unique")
    return matches[0]


def _p1_cell_directory(delay: float, seed: int) -> Path:
    token = f"{delay:g}".replace(".", "p")
    return P1_RUN_ROOT / f"perturb_{token}" / f"seed_{seed}"


def run_p1_cell(
    config: dict[str, Any], *, delay: float, seed: int, device: str
) -> dict[str, Any]:
    directory = _p1_cell_directory(delay, seed)
    completion_marker = directory / "run_complete.json"
    metrics_path = directory / "metrics.json"
    if completion_marker.exists() and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if directory.exists():
        raise RuntimeError(f"incomplete P1 cell exists and requires manual audit: {directory}")

    parent = withdrawal.load_protocol()
    adapted = withdrawal.level1b_protocol(parent)
    estimator = _selected_estimator(config)
    spec = withdrawal._branch_spec(
        parent, int(seed), "task_delay_only", float(delay),
        updates=int(config["p1_conditional_recovery"]["full_batch_updates"]),
    )
    spec.update({
        "stage": "stage_p1_conditional_recovery",
        "condition": "task_delay_only_repaired_backward",
        "selection_role": "conditional_recovery_after_registered_p0_selection",
        "granularity": "per_hidden_neuron",
        "delay_tying": "post_neuron",
        "independent_delay_parameters": 16,
        "delay_gradient_mode": str(estimator["delay_gradient_mode"]),
        "delay_gradient_sigma": float(estimator["delay_gradient_sigma"]),
    })
    state = _initial_state(adapted, spec, seed=int(seed), device=device)
    model, result = level1b.train_cell(
        adapted,
        spec,
        device=device,
        initial_state_dict=state,
        functional_delay_override=float(delay),
        trainable_components={"input_hidden_delays"},
        task_loss_weight=1.0,
    )
    record = result["final_record"]
    interface = result["final_interface"]
    final_delays = np.asarray(record["final_independent_delays"], dtype=np.float64).reshape(-1)
    target = float(config["scope"]["target_delay_steps"])
    error = np.abs(final_delays - target)
    tolerance = 0.1
    functional_pass = bool(result["interface_pass"])
    schedule_pass = bool(error.max() <= tolerance and np.mean(error <= tolerance) == 1.0)
    initial_task_gradient = np.asarray(result["initial_gradients"]["task"], dtype=np.float64).reshape(-1)
    metrics = {
        "protocol_id": PROTOCOL_ID,
        "stage": "stage_p1_conditional_recovery",
        "condition": spec["condition"],
        "selected_estimator_id": str(estimator["id"]),
        "delay_gradient_mode": str(estimator["delay_gradient_mode"]),
        "delay_gradient_sigma": float(estimator["delay_gradient_sigma"]),
        "perturbation_delay_steps": float(delay),
        "seed": int(seed),
        "full_batch_updates": int(spec["full_batch_updates"]),
        "initial_task_gradient_target_directed_coordinate_fraction": float(
            np.mean(initial_task_gradient * (float(delay) - target) > 0.0)
        ),
        "initial_task_gradient_mean_target_directed": bool(
            initial_task_gradient.mean() * (float(delay) - target) > 0.0
        ),
        "initial_task_gradient_nonzero_coordinate_fraction": float(
            np.mean(np.abs(initial_task_gradient) > 1e-10)
        ),
        "initial_exact_interface_patterns": int(result["history"]["exact_interface_patterns"][0]),
        "final_exact_interface_patterns": int(result["history"]["exact_interface_patterns"][-1]),
        "balanced_accuracy": float(interface["balanced_accuracy"]),
        "exact_truth_table_completion": bool(interface["exact_truth_table_completion"]),
        "exact_interface_completion": bool(interface["exact_interface_completion"]),
        "silent_rate": float(interface["silent_rate"]),
        "collision_rate": float(interface["collision_rate"]),
        "correct_target_time_rate": float(interface["correct_target_time_rate"]),
        "mean_output_spikes_per_trial": float(interface["mean_output_spikes_per_trial"]),
        "hidden_active_pattern_count": int(interface["hidden_active_pattern_count"]),
        "final_delay_mean_steps": float(final_delays.mean()),
        "final_delay_min_steps": float(final_delays.min()),
        "final_delay_max_steps": float(final_delays.max()),
        "final_delay_max_error_steps": float(error.max()),
        "final_delay_fraction_within_0p1": float(np.mean(error <= tolerance)),
        "functional_recovery_pass": functional_pass,
        "schedule_identification_pass": schedule_pass,
        "joint_full_pass": bool(functional_pass and schedule_pass),
        "test_split_opened": False,
        "complete": True,
    }

    directory.mkdir(parents=True)
    shutil.copy2(CONFIG, directory / "protocol_snapshot.yaml")
    level1b._strict_write_json(directory / "config.json", {
        **{key: (sorted(value) if isinstance(value, set) else value) for key, value in spec.items()},
        "protocol_id": PROTOCOL_ID,
        "parent_checkpoint": str((W0_ROOT / f"seed_{seed}" / "final_model.pt").relative_to(BASE)),
        "optimizer_state_inherited": False,
        "test_split_opened": False,
    })
    level1b._strict_write_json(metrics_path, metrics)
    level1b._strict_write_json(directory / "branch_provenance.json", {
        "parent_checkpoint": str((W0_ROOT / f"seed_{seed}" / "final_model.pt").relative_to(BASE)),
        "p0_selection_summary": str((RUN_ROOT / "summary.json").relative_to(BASE)),
        "selected_estimator_id": str(estimator["id"]),
        "perturbation_delay_steps": float(delay),
        "trainable_components": ["input_hidden_delays"],
        "weights_frozen": True,
        "arrival_teacher_weight": 0.0,
        "task_loss_weight": 1.0,
        "fresh_optimizer_state": True,
    })
    level1b._strict_write_json(
        directory / "training_log.json",
        [
            {key: float(values[index]) for key, values in result["history"].items()}
            for index in range(len(result["history"]["step"]))
        ],
    )
    level1b._strict_write_json(directory / "exhaustive_truth_table_results.json", {
        "evaluation_split": "exhaustive_truth_table_training_domain",
        "predictions": interface["predictions"],
        "labels": interface["labels"],
        "valid_pattern_mask": interface["valid_pattern_mask"],
        "exact_truth_table_completion": bool(interface["exact_truth_table_completion"]),
        "exact_interface_completion": bool(interface["exact_interface_completion"]),
        "test_split_opened": False,
    })
    level1b._strict_write_json(
        directory / "resource_ledger.json",
        level1b.resource_ledger(model, spec, record),
    )
    torch.save(model.state_dict(), directory / "final_model.pt")
    plots = directory / "plots"
    plots.mkdir()
    gradients = {
        f"initial_{name}_gradient": np.asarray(value if value is not None else [], dtype=np.float64)
        for name, value in result["initial_gradients"].items()
    }
    np.savez_compressed(plots / "diagnostic_data.npz", **result["history"], **record, **gradients)
    level1b.save_diagnostic_panel(adapted, spec, result, plots / "diagnostic_panel.png")
    with completion_marker.open("x", encoding="utf-8") as handle:
        json.dump({"protocol_id": PROTOCOL_ID, "stage": metrics["stage"], "complete": True}, handle, indent=2)
        handle.write("\n")
    return metrics


def run_p1(config: dict[str, Any], *, device: str) -> dict[str, Any]:
    if not bool(config["authorization"]["p1_launch"]):
        raise RuntimeError("P1 is not authorized")
    p0_summary = json.loads((RUN_ROOT / "summary.json").read_text(encoding="utf-8"))
    selected_id = str(config["observed_stage_decisions"]["selected_estimator_id"])
    if not bool(p0_summary["p1_unlocked_by_rule"]) or p0_summary["selected_estimator_id"] != selected_id:
        raise RuntimeError("P0 artifact does not authorize the configured P1 estimator")

    rows = [
        run_p1_cell(config, delay=float(delay), seed=int(seed), device=device)
        for delay in config["scope"]["perturbations_steps"]
        for seed in config["p0_gradient_screen"]["seeds"]
    ]
    functional = int(sum(bool(row["functional_recovery_pass"]) for row in rows))
    schedule = int(sum(bool(row["schedule_identification_pass"]) for row in rows))
    joint = int(sum(bool(row["joint_full_pass"]) for row in rows))
    summary = {
        "protocol_id": PROTOCOL_ID,
        "stage": "stage_p1_conditional_recovery",
        "complete": True,
        "selected_estimator_id": selected_id,
        "cells": len(rows),
        "functional_recovery_cells": functional,
        "schedule_identification_cells": schedule,
        "joint_full_pass_cells": joint,
        "functional_gate_pass": functional == len(rows),
        "schedule_gate_pass": schedule == len(rows),
        "protocol_repair_pass": joint == len(rows),
        "by_perturbation": {
            f"d{delay:g}": {
                "cells": int(sum(float(row["perturbation_delay_steps"]) == float(delay) for row in rows)),
                "functional_recovery_cells": int(sum(
                    float(row["perturbation_delay_steps"]) == float(delay)
                    and bool(row["functional_recovery_pass"]) for row in rows
                )),
                "schedule_identification_cells": int(sum(
                    float(row["perturbation_delay_steps"]) == float(delay)
                    and bool(row["schedule_identification_pass"]) for row in rows
                )),
            }
            for delay in config["scope"]["perturbations_steps"]
        },
        "test_split_opened": False,
    }
    P1_SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    level1b._strict_write_json(P1_RUN_ROOT / "summary.json", summary)
    level1b._strict_write_json(P1_SUMMARY_ROOT / "summary.json", summary)
    with (P1_RUN_ROOT / "cell_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(P1_RUN_ROOT / "cell_metrics.csv", P1_SUMMARY_ROOT / "cell_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for index, delay in enumerate(config["scope"]["perturbations_steps"]):
        selected = [row for row in rows if float(row["perturbation_delay_steps"]) == float(delay)]
        axes[0].scatter(
            [float(delay)] * len(selected),
            [float(row["final_delay_mean_steps"]) for row in selected],
            label=f"d={float(delay):g}", alpha=.8,
        )
    axes[0].axhline(4.0, color="tab:green", linestyle="--", label="oracle d=4")
    axes[0].set(xlabel="Initial perturbation (steps)", ylabel="Final mean delay (steps)", title="Delay endpoint")
    axes[0].legend(frameon=False)
    labels = ["functional", "schedule", "joint"]
    counts = [functional, schedule, joint]
    axes[1].bar(labels, counts, color=["tab:blue", "tab:orange", "tab:purple"])
    axes[1].axhline(len(rows), color="black", linestyle="--", linewidth=1)
    axes[1].set(ylim=(0, len(rows) + .5), ylabel=f"Passing cells / {len(rows)}", title="Registered endpoints")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("XOR repaired-backward P1 recovery")
    fig.savefig(P1_RUN_ROOT / "diagnostic_panel.png", dpi=180, facecolor="white")
    fig.savefig(P1_SUMMARY_ROOT / "diagnostic_panel.png", dpi=180, facecolor="white")
    plt.close(fig)
    return summary
    np.savez_compressed(
        RUN_ROOT / "diagnostic_data.npz",
        raw_gradients=gradients,
        estimator_id=np.asarray([row["estimator_id"] for row in rows]),
        granularity=np.asarray([row["granularity"] for row in rows]),
        perturbation_delay_steps=np.asarray([row["perturbation_delay_steps"] for row in rows]),
        seed=np.asarray([row["seed"] for row in rows]),
        direction_fraction=np.asarray([row["target_directed_coordinate_fraction"] for row in rows]),
        nonzero_fraction=np.asarray([row["nonzero_coordinate_fraction"] for row in rows]),
    )
    shutil.copy2(RUN_ROOT / "gradient_probes.csv", SUMMARY_ROOT / "gradient_probes.csv")
    shutil.copy2(RUN_ROOT / "summary.json", SUMMARY_ROOT / "summary.json")

    estimator_ids = [str(item["id"]) for item in config["estimators"]]
    x = np.arange(len(estimator_ids))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    width = 0.36
    for offset, delay in ((-width / 2, 3.0), (width / 2, 5.0)):
        values = [
            np.mean([
                float(row["target_directed_coordinate_fraction"])
                for row in rows
                if row["estimator_id"] == estimator_id
                and row["granularity"] == "per_hidden"
                and float(row["perturbation_delay_steps"]) == delay
            ])
            for estimator_id in estimator_ids
        ]
        axes[0].bar(x + offset, values, width=width, label=f"d={delay:g}")
    axes[0].axhline(.75, color="black", linestyle="--", linewidth=1, label="registered 0.75")
    axes[0].set(ylabel="Target-directed coordinate fraction", title="Per-hidden task gradient")
    axes[0].set_xticks(x, estimator_ids, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False)

    for estimator_id in estimator_ids:
        means = []
        for delay in (3.0, 5.0):
            means.append(np.mean([
                float(row["raw_gradient_mean"])
                for row in rows
                if row["estimator_id"] == estimator_id
                and row["granularity"] == "global"
                and float(row["perturbation_delay_steps"]) == delay
            ]))
        axes[1].plot((3.0, 5.0), means, marker="o", label=estimator_id)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(4, color="tab:green", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Functional delay (steps)", ylabel="Mean raw task gradient", title="Global tied-delay direction")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("XOR integer-boundary credit preflight P0")
    fig.savefig(RUN_ROOT / "diagnostic_panel.png", dpi=180, facecolor="white")
    fig.savefig(SUMMARY_ROOT / "diagnostic_panel.png", dpi=180, facecolor="white")
    plt.close(fig)
    with completion_marker.open("x", encoding="utf-8") as handle:
        json.dump({"protocol_id": PROTOCOL_ID, "stage": summary["stage"], "complete": True}, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("p0", "p1"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config()
    if args.stage == "p0":
        if not bool(config["authorization"]["p0_launch"]):
            raise RuntimeError("P0 is not authorized")
        rows, gradients = run_probes(config, device=args.device)
        summary = summarize(config, rows)
        write_artifacts(config, rows, gradients, summary)
    else:
        summary = run_p1(config, device=args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

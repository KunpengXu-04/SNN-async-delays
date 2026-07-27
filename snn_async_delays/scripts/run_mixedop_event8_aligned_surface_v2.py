"""Run the preregistered event8-aligned K=5 landmark/surface preview."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import scripts.run_mixedop_spatial_temporal_surface_preview as core
from scripts.run_mixedop_temporal_wad_scaffold_withdrawal import (
    compose_delay_gradient,
    lambda_at_update,
    physical_query_delays,
)
from train.eval import evaluate_simultaneous, save_eval_results
from train.trainer import build_optimizer, window_class_balanced_bce
from utils.seed import set_seed
from utils.viz import save_run_diagnostic_plots


BASE = core.BASE
PROTOCOL = "mixedop_event8_aligned_surface_v2"
CONFIG_PATH = BASE / "configs" / f"{PROTOCOL}.yaml"
SURFACE_CONDITIONS = (
    "spatial_independent_d0",
    "shared_d0",
    "shared_temporal_oracle",
    "shared_temporal_task_only",
    "shared_temporal_annealed",
)


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL:
        raise ValueError("protocol id mismatch")
    windows = protocol["surface"]["output_window_lengths"]
    if min(windows) < int(protocol["encoding"]["rate_steps"]):
        raise ValueError("every output window must contain the event8 packet")
    if tuple(protocol["surface"]["conditions"]) != SURFACE_CONDITIONS:
        raise ValueError("surface conditions changed after preregistration")
    return protocol


def _stage_protocol(protocol: dict[str, Any], stage: str) -> dict[str, Any]:
    value = deepcopy(protocol)
    if stage == "full":
        value["execution"]["formal_root"] = value["execution"]["full_root"]
    return value


def _core_condition(surface_condition: str) -> tuple[str, str | None]:
    if surface_condition == "shared_temporal_task_only":
        return "shared_temporal_wad", surface_condition
    if surface_condition == "shared_temporal_annealed":
        return "shared_temporal_wad", surface_condition
    return surface_condition, None


def _spec(
    point: dict[str, Any], condition: str, seed: int, stage: str,
    updates: int,
) -> dict[str, Any]:
    core_condition, variant = _core_condition(condition)
    return {
        "K": 5, "condition": core_condition,
        "surface_condition": condition,
        "path_variant": variant,
        "training_arm": condition,
        "total_hidden": int(point["total_hidden"]),
        "output_window_len": int(point["output_window_len"]),
        "point_label": str(point["label"]),
        "seed": int(seed), "updates": int(updates),
        "stage": "smoke" if stage == "smoke" else stage,
        "event_budget_label": "event8",
        "r_on_hz": 990.0,
        "routing_loss_kind": "arrival_centroid_huber",
        "routing_loss_weight": 0.0,
        "delay_credit_mode": "joint",
        "save_schedule_checkpoint": condition == "shared_temporal_task_only",
        "event8_aligned": True,
        "primary_surface_metric": "worst_query_balanced_accuracy",
    }


def specs(protocol: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "smoke":
        block = protocol["smoke"]
        point = block["point"]
        return [
            _spec(
                point, condition, block["seed"], stage,
                block["curriculum_scaffold_updates"]
                + block["curriculum_withdrawal_updates"]
                if condition == "shared_temporal_annealed"
                else block["ordinary_condition_updates"],
            )
            for condition in block["conditions"]
        ]
    if stage == "landmark":
        block = protocol["landmark_pilot"]
        return [
            _spec(
                point, condition, block["seed"], stage,
                protocol["curriculum_assisted"]["scaffold_updates"]
                + protocol["curriculum_assisted"]["withdrawal_updates"]
                if condition == "shared_temporal_annealed"
                else protocol["optimization"]["optimizer_updates"],
            )
            for point, condition in product(block["points"], SURFACE_CONDITIONS)
        ]
    if stage == "full":
        seed = int(protocol["full_sweep"]["seed"])
        landmark_labels = {
            (int(point["total_hidden"]), int(point["output_window_len"])): str(point["label"])
            for point in protocol["landmark_pilot"]["points"]
        }
        points = [
            {"label": landmark_labels.get((hidden, window), f"N{hidden}_w{window}"),
             "total_hidden": hidden,
             "output_window_len": window}
            for hidden, window in product(
                protocol["surface"]["total_hidden_neurons"],
                protocol["surface"]["output_window_lengths"],
            )
        ]
        cells = []
        for point, condition in product(points, SURFACE_CONDITIONS):
            point_stage = (
                "landmark"
                if (int(point["total_hidden"]), int(point["output_window_len"]))
                in landmark_labels
                else "full"
            )
            cells.append(_spec(
                point, condition, seed, point_stage,
                protocol["curriculum_assisted"]["scaffold_updates"]
                + protocol["curriculum_assisted"]["withdrawal_updates"]
                if condition == "shared_temporal_annealed"
                else protocol["optimization"]["optimizer_updates"],
            ))
        return cells
    raise ValueError(stage)


def build_config(protocol: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    cfg = core.build_config(_stage_protocol(protocol, "full" if spec["stage"] == "full" else spec["stage"]), spec)
    cfg.update({
        "surface_condition": spec["surface_condition"],
        "routing_loss_kind": "arrival_centroid_huber",
        "event8_aligned": True,
        "primary_surface_metric": "worst_query_balanced_accuracy",
    })
    return cfg


def run_dir(protocol: dict[str, Any], spec: dict[str, Any]) -> Path:
    stage_protocol = _stage_protocol(protocol, "full" if spec["stage"] == "full" else spec["stage"])
    return core._run_directory(stage_protocol, core.build_config(stage_protocol, spec))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optimizer(model: torch.nn.Module, cfg: dict[str, Any]):
    return build_optimizer(model, cfg)


def _delay_gradient(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    if not values or values[0] is None:
        return torch.zeros_like(parameters[0])
    return values[0].detach().clone()


def _joint_gate(protocol: dict[str, Any], metrics: dict[str, Any]) -> bool:
    gate = protocol["pilot_decision"]["curriculum_gate_each_passing_point"]
    return bool(
        metrics["worst_query_balanced_accuracy"] >= float(gate["worst_query_balanced_accuracy_minimum"])
        and metrics["exact_trial_accuracy"] >= float(gate["exact_trial_accuracy_minimum"])
        and min(metrics["per_output_window_hidden_activity_fraction"])
        >= float(gate["each_window_activity_fraction_minimum"])
        and metrics["delay_query_schedule_max_abs_error_steps"]
        <= float(gate["maximum_integer_oracle_schedule_error_steps"])
    )


def _evaluate(model, loader, cfg, device, encoder, collect=False):
    metrics, arrays = core.validation_snapshot(
        model, loader, cfg, device, encoder, collect=collect
    )
    metrics.update(core._delay_diagnostics(model, cfg))
    return metrics, arrays


def _curriculum_updates(protocol: dict[str, Any], stage: str) -> tuple[int, int]:
    if stage == "smoke":
        return (
            int(protocol["smoke"]["curriculum_scaffold_updates"]),
            int(protocol["smoke"]["curriculum_withdrawal_updates"]),
        )
    return (
        int(protocol["curriculum_assisted"]["scaffold_updates"]),
        int(protocol["curriculum_assisted"]["withdrawal_updates"]),
    )


def run_curriculum_cell(
    protocol: dict[str, Any], spec: dict[str, Any], device: str,
    dry_run: bool = False,
) -> Path:
    directory = run_dir(protocol, spec)
    if dry_run:
        return directory
    stage_protocol = _stage_protocol(protocol, "full" if spec["stage"] == "full" else spec["stage"])
    if core._complete(directory, stage_protocol):
        return directory
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"incomplete existing curriculum cell: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    cfg = build_config(protocol, spec)
    core._write_json(directory / "config.json", cfg)
    started = time.time()
    set_seed(cfg["seed"])
    model = core.build_model(cfg).to(device)
    train_loader, validation_loader = core.loaders(cfg)
    encoder = core.encode_fn(cfg)
    delay_parameters = list(model.delay_params())
    if len(delay_parameters) != 1 or delay_parameters[0].numel() != 5:
        raise RuntimeError("curriculum requires exactly five query-tied delay parameters")
    optimized_parameters = [p for p in model.parameters() if p.requires_grad]
    scaffold_updates, withdrawal_updates = _curriculum_updates(protocol, spec["stage"])
    update_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    viz_rows: list[dict[str, Any]] = []

    optimizer = _optimizer(model, cfg)
    initial, _ = _evaluate(model, validation_loader, cfg, device, encoder, False)
    validation_rows.append({"phase": "scaffold", "update": 0, "global_update": 0, **initial})
    best_accuracy = (float(initial["worst_query_balanced_accuracy"]), float(initial["exact_trial_accuracy"]))
    best_schedule = (-float(initial["delay_query_schedule_max_abs_error_steps"]), *best_accuracy)
    best_accuracy_update = best_schedule_update = 0
    torch.save(model.state_dict(), directory / "scaffold_accuracy_best_model.pt")
    torch.save(model.state_dict(), directory / "scaffold_schedule_best_model.pt")
    iterator = iter(train_loader)

    def train_step(phase: str, update: int, global_update: int, routing_lambda: float, route_only: bool):
        nonlocal iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        model.train()
        _, _, _, labels, spikes = core._batch_input(batch, cfg, device, encoder)
        optimizer.zero_grad(set_to_none=True)
        logits, info = model(spikes)
        task = window_class_balanced_bce(logits, labels)
        route = core.routing_alignment_loss(spikes, model, cfg)
        task_grad = _delay_gradient(task, delay_parameters)
        route_grad = _delay_gradient(route, delay_parameters)
        task.backward()
        combined = (
            route_grad if route_only
            else compose_delay_gradient("annealed_joint", task_grad, route_grad, routing_lambda)
        )
        delay_parameters[0].grad = combined.clone()
        finite = core._all_finite(p.grad for p in optimized_parameters if p.grad is not None)
        norm = torch.nn.utils.clip_grad_norm_(optimized_parameters, cfg["grad_clip"])
        optimizer.step()
        delays = physical_query_delays(model)
        row = {
            "phase": phase, "update": update, "global_update": global_update,
            "lambda_routing": routing_lambda,
            "task_loss": float(task.detach().item()),
            "routing_loss": float(route.detach().item()),
            "train_pooled_accuracy": float(((logits > 0).float() == labels).float().mean().item()),
            "mean_hidden_spikes": float(info["total_hidden_spikes"].detach().mean().item()),
            "task_delay_gradient_norm": float(task_grad.norm().item()),
            "routing_delay_gradient_norm": float(route_grad.norm().item()),
            "combined_delay_gradient_norm": float(combined.norm().item()),
            "global_grad_norm_before_clip": float(norm.item()),
            "loss_finite": bool(torch.isfinite(task).item() and torch.isfinite(route).item()),
            "gradients_finite": finite,
            "parameters_finite": core._all_finite(p.detach() for p in model.parameters()),
            "delays_finite": bool(np.isfinite(delays).all()),
            "delays_legal": bool(min(delays) >= 0 and max(delays) <= cfg["d_max"]),
        }
        for q, value in enumerate(delays):
            row[f"delay_q{q}_steps"] = value
            row[f"task_delay_grad_q{q}"] = float(task_grad.reshape(-1)[q].item())
            row[f"routing_delay_grad_q{q}"] = float(route_grad.reshape(-1)[q].item())
        update_rows.append(row)
        return row

    for update in range(1, scaffold_updates + 1):
        row = train_step("scaffold", update, update, 1.0, True)
        if update % cfg["validation_interval_updates"] == 0 or update == scaffold_updates:
            metrics, _ = _evaluate(model, validation_loader, cfg, device, encoder, False)
            validation_rows.append({"phase": "scaffold", "update": update, "global_update": update, **metrics})
            score = (float(metrics["worst_query_balanced_accuracy"]), float(metrics["exact_trial_accuracy"]))
            schedule = (-float(metrics["delay_query_schedule_max_abs_error_steps"]), *score)
            if score > best_accuracy:
                best_accuracy, best_accuracy_update = score, update
                torch.save(model.state_dict(), directory / "scaffold_accuracy_best_model.pt")
            if schedule > best_schedule:
                best_schedule, best_schedule_update = schedule, update
                torch.save(model.state_dict(), directory / "scaffold_schedule_best_model.pt")
            viz_rows.append({
                "epoch": update, "train_loss": row["task_loss"], "val_loss": metrics["loss"],
                "train_acc": row["train_pooled_accuracy"], "val_acc": metrics["pooled_accuracy"],
                "val_worst_query_balanced_accuracy": metrics["worst_query_balanced_accuracy"],
                "val_exact_trial_accuracy": metrics["exact_trial_accuracy"],
                "weight_grad_norm": row["global_grad_norm_before_clip"],
                "delay_grad_norm": row["combined_delay_gradient_norm"],
            })

    source_name = "scaffold_accuracy_best_model.pt"
    model.load_state_dict(torch.load(directory / source_name, map_location=device, weights_only=True))
    source_metrics, _ = _evaluate(model, validation_loader, cfg, device, encoder, True)
    source_update = best_accuracy_update
    scaffold_pass = _joint_gate(protocol, source_metrics)
    if not scaffold_pass:
        source_name = "scaffold_schedule_best_model.pt"
        source_update = best_schedule_update
        model.load_state_dict(torch.load(directory / source_name, map_location=device, weights_only=True))
        source_metrics, _ = _evaluate(model, validation_loader, cfg, device, encoder, True)
    torch.save(model.state_dict(), directory / "scaffold_source_model.pt")
    core._write_json(directory / "scaffold_source_results.json", {
        **source_metrics, "source_checkpoint": source_name,
        "source_update": source_update, "joint_gate_passed": scaffold_pass,
    })

    optimizer = _optimizer(model, cfg)
    iterator = iter(train_loader)
    torch.save(model.state_dict(), directory / "best_model.pt")
    best_withdrawal = (
        float(source_metrics["worst_query_balanced_accuracy"]),
        float(source_metrics["exact_trial_accuracy"]),
    )
    core._write_json(directory / "phase_provenance.json", {
        "scaffold_source_checkpoint": source_name,
        "scaffold_source_update": source_update,
        "scaffold_source_joint_gate_passed": scaffold_pass,
        "optimizer_state_reset_before_withdrawal": True,
        "scaffold_updates": scaffold_updates,
        "withdrawal_updates": withdrawal_updates,
    })

    for update in range(1, withdrawal_updates + 1):
        lam = lambda_at_update("annealed_joint", update)
        global_update = scaffold_updates + update
        row = train_step("withdrawal", update, global_update, lam, False)
        if update % cfg["validation_interval_updates"] == 0 or update == withdrawal_updates:
            metrics, _ = _evaluate(model, validation_loader, cfg, device, encoder, False)
            validation_rows.append({"phase": "withdrawal", "update": update, "global_update": global_update, **metrics})
            score = (float(metrics["worst_query_balanced_accuracy"]), float(metrics["exact_trial_accuracy"]))
            if score > best_withdrawal:
                best_withdrawal = score
                torch.save(model.state_dict(), directory / "best_model.pt")
            viz_rows.append({
                "epoch": global_update, "train_loss": row["task_loss"], "val_loss": metrics["loss"],
                "train_acc": row["train_pooled_accuracy"], "val_acc": metrics["pooled_accuracy"],
                "val_worst_query_balanced_accuracy": metrics["worst_query_balanced_accuracy"],
                "val_exact_trial_accuracy": metrics["exact_trial_accuracy"],
                "weight_grad_norm": row["global_grad_norm_before_clip"],
                "delay_grad_norm": row["combined_delay_gradient_norm"],
            })

    torch.save(model.state_dict(), directory / "last_model.pt")
    final_metrics, arrays = _evaluate(model, validation_loader, cfg, device, encoder, True)
    assert arrays is not None
    with core._fixed_encoding_rng(cfg, device):
        detailed = evaluate_simultaneous(
            model, validation_loader, cfg, device, encode_fn=encoder,
            return_trial_records=False,
        )
    results = {
        **detailed, **final_metrics,
        "protocol_id": PROTOCOL, "surface_condition": "shared_temporal_annealed",
        "point_label": spec["point_label"], "K": 5,
        "total_hidden_neurons": cfg["surface_total_hidden"],
        "total_latency_steps": cfg["T"], "output_window_len": cfg["output_window_len"],
        "selected_checkpoint": "last_model.pt", "selected_update": withdrawal_updates,
        "scaffold_source_joint_passed": scaffold_pass,
        "training_optimizer_updates_total": scaffold_updates + withdrawal_updates,
        "neuron_update_proxy_N_times_T": cfg["surface_total_hidden"] * cfg["T"],
        "test_split_opened": False,
        "claim_status": "invalid_smoke" if spec["stage"] == "smoke" else "exploratory_single_seed",
        "wall_time_seconds": time.time() - started,
    }
    save_eval_results(results, str(directory / "validation_results.json"))
    np.savez_compressed(directory / "validation_predictions.npz", **arrays)
    core._write_json(directory / "resource_ledger.json", results["resource_ledger"])
    core._write_csv(directory / "update_log.csv", update_rows)
    core._write_csv(directory / "validation_log.csv", validation_rows)
    save_run_diagnostic_plots(
        model, cfg, viz_rows, results, str(directory), 5, "mixed", device,
        seed=cfg["seed"] + 10000, dataset_override=validation_loader.dataset[0],
    )
    missing = [
        item for item in protocol["required_cell_artifacts"]
        if item != "run_complete.json" and not (directory / item).exists()
    ]
    if missing:
        raise RuntimeError(f"curriculum artifacts missing: {missing}")
    core._write_json(directory / "run_complete.json", {
        "protocol_id": PROTOCOL, "stage": spec["stage"], "completed": True,
        "required_artifacts_complete": True, "test_split_opened": False,
        "wall_time_seconds": results["wall_time_seconds"],
    })
    return directory


def run_cell(protocol: dict[str, Any], spec: dict[str, Any], device: str) -> Path:
    if spec["surface_condition"] == "shared_temporal_annealed":
        return run_curriculum_cell(protocol, spec, device)
    return core.run_cell(_stage_protocol(protocol, "full" if spec["stage"] == "full" else spec["stage"]), spec, device)


def _cell_row(protocol: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    directory = run_dir(protocol, spec)
    stage_protocol = _stage_protocol(protocol, "full" if spec["stage"] == "full" else spec["stage"])
    complete = core._complete(directory, stage_protocol)
    result = _read_json(directory / "validation_results.json") if complete else {}
    logs = _read_csv(directory / "update_log.csv") if complete else []
    row = {
        "stage": spec["stage"], "point_label": spec["point_label"],
        "condition": spec["surface_condition"], "seed": spec["seed"],
        "total_hidden_neurons": spec["total_hidden"],
        "output_window_len": spec["output_window_len"],
        "total_latency_steps": 10 + 5 * int(spec["output_window_len"]),
        "artifacts_complete": complete,
        "finite_and_legal": bool(logs) and all(
            item.get("loss_finite") == "True" and item.get("gradients_finite") == "True"
            and item.get("parameters_finite") == "True" and item.get("delays_finite") == "True"
            and item.get("delays_legal") == "True" for item in logs
        ),
    }
    if result:
        row.update({
            "worst_query_balanced_accuracy": result["worst_query_balanced_accuracy"],
            "exact_trial_accuracy": result["exact_trial_accuracy"],
            "minimum_window_activity_fraction": min(result["per_output_window_hidden_activity_fraction"]),
            "maximum_schedule_error_steps": result["delay_query_schedule_max_abs_error_steps"],
            "delay_query_means": result["delay_query_mean_steps"],
            "neuron_update_proxy_N_times_T": int(spec["total_hidden"]) * (10 + 5 * int(spec["output_window_len"])),
            "scaffold_source_joint_passed": result.get("scaffold_source_joint_passed"),
        })
    row["technical_valid"] = bool(row["artifacts_complete"] and row["finite_and_legal"])
    return row


def _plot_landmarks(rows: list[dict[str, Any]], output: Path) -> None:
    conditions = list(SURFACE_CONDITIONS)
    points = ["lowN_lowT", "lowN_highT", "highN_lowT", "highN_highT", "geometric_center"]
    for metric, filename, vmin, vmax in (
        ("worst_query_balanced_accuracy", "landmark_worst_bacc.png", 0.5, 1.0),
        ("exact_trial_accuracy", "landmark_exact_trial.png", 0.0, 1.0),
        ("maximum_schedule_error_steps", "landmark_schedule_error.png", 0.0, 4.0),
    ):
        matrix = np.full((len(conditions), len(points)), np.nan)
        for row in rows:
            if (
                metric == "maximum_schedule_error_steps"
                and row["condition"] in {"spatial_independent_d0", "shared_d0"}
            ):
                continue
            matrix[conditions.index(row["condition"]), points.index(row["point_label"])] = float(row[metric])
        fig, ax = plt.subplots(figsize=(10, 5))
        image = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isnan(value):
                    label, color = "n/a", "black"
                else:
                    label = f"{value:.3f}"
                    normalized = (min(max(value, vmin), vmax) - vmin) / max(vmax - vmin, 1e-12)
                    color = "black" if normalized >= 0.62 else "white"
                ax.text(j, i, label, ha="center", va="center", color=color)
        ax.set_xticks(range(len(points)), points, rotation=25, ha="right")
        ax.set_yticks(range(len(conditions)), conditions)
        ax.set_title(metric.replace("_", " "))
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        ax.scatter(
            [row["neuron_update_proxy_N_times_T"] for row in selected],
            [row["worst_query_balanced_accuracy"] for row in selected],
            label=condition,
        )
    ax.set_xlabel("N_hidden × T (neuron-update proxy)")
    ax.set_ylabel("Worst-query balanced accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "landmark_accuracy_cost.png", dpi=180)
    plt.close(fig)


def _plot_full_surface(
    protocol: dict[str, Any], rows: list[dict[str, Any]], output: Path
) -> None:
    hidden_values = list(protocol["surface"]["total_hidden_neurons"])
    window_values = list(protocol["surface"]["output_window_lengths"])
    for metric, filename, vmin, vmax in (
        ("worst_query_balanced_accuracy", "surface_worst_bacc.png", 0.5, 1.0),
        ("exact_trial_accuracy", "surface_exact_trial.png", 0.0, 1.0),
        ("maximum_schedule_error_steps", "surface_schedule_error.png", 0.0, 4.0),
    ):
        fig, axes = plt.subplots(1, len(SURFACE_CONDITIONS), figsize=(22, 4.5))
        image = None
        for ax, condition in zip(axes, SURFACE_CONDITIONS):
            matrix = np.full((len(hidden_values), len(window_values)), np.nan)
            for row in rows:
                if row["condition"] != condition:
                    continue
                i = hidden_values.index(int(row["total_hidden_neurons"]))
                j = window_values.index(int(row["output_window_len"]))
                matrix[i, j] = float(row[metric])
            image = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_xticks(range(len(window_values)), window_values)
            ax.set_yticks(range(len(hidden_values)), hidden_values)
            ax.set_xlabel("output-window length w")
            ax.set_ylabel("total hidden neurons")
            ax.set_title(condition)
        if image is not None:
            fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
        fig.suptitle(metric.replace("_", " "))
        fig.subplots_adjust(left=0.05, right=0.97, bottom=0.15, top=0.82, wspace=0.35)
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)


def summarize(protocol: dict[str, Any], stage: str) -> dict[str, Any]:
    rows = [_cell_row(protocol, spec) for spec in specs(protocol, stage)]
    output = BASE / protocol["execution"]["generated_root"]
    output.mkdir(parents=True, exist_ok=True)
    serializable = [
        {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
        for row in rows
    ]
    if stage == "smoke":
        by_condition = {row["condition"]: row for row in rows}
        task_spec = next(item for item in specs(protocol, stage) if item["surface_condition"] == "shared_temporal_task_only")
        task_logs = _read_csv(run_dir(protocol, task_spec) / "update_log.csv")
        curriculum_spec = next(item for item in specs(protocol, stage) if item["surface_condition"] == "shared_temporal_annealed")
        curriculum_dir = run_dir(protocol, curriculum_spec)
        curriculum_logs = _read_csv(curriculum_dir / "update_log.csv")
        curriculum_phases = {item["phase"] for item in curriculum_logs}
        passed = bool(rows) and all(row["technical_valid"] for row in rows)
        passed = passed and float(by_condition["shared_d0"]["maximum_schedule_error_steps"]) <= 1e-7
        passed = passed and float(by_condition["shared_temporal_oracle"]["maximum_schedule_error_steps"]) <= 1e-7
        passed = passed and any(float(item.get("delay_grad_norm", 0.0)) > 0 for item in task_logs)
        passed = passed and curriculum_phases == {"scaffold", "withdrawal"}
        passed = passed and _read_json(curriculum_dir / "phase_provenance.json").get(
            "optimizer_state_reset_before_withdrawal"
        ) is True
        decision = {"protocol_id": PROTOCOL, "stage": stage, "passed": passed, "rows": rows}
        core._write_csv(output / "smoke_cells.csv", serializable)
        core._write_json(output / "smoke_decision.json", decision)
        return decision

    if stage == "full":
        decision = {
            "protocol_id": PROTOCOL,
            "stage": stage,
            "cells_total_including_reused_landmarks": len(rows),
            "new_full_surface_cells": sum(row["stage"] == "full" for row in rows),
            "reused_landmark_cells": sum(row["stage"] == "landmark" for row in rows),
            "all_cells_technically_valid": all(row["technical_valid"] for row in rows),
            "single_seed_exploratory": True,
            "test_split_opened": False,
            "publication_claim_authorized": False,
        }
        core._write_csv(output / "surface_cells.csv", serializable)
        core._write_json(output / "surface_decision.json", decision)
        _plot_full_surface(protocol, rows, output)
        return decision

    oracle_gate = protocol["pilot_decision"]["oracle_gate_each_required_point"]
    required_oracles = []
    for label in protocol["pilot_decision"]["oracle_required_points"]:
        row = next(item for item in rows if item["point_label"] == label and item["condition"] == "shared_temporal_oracle")
        required_oracles.append(bool(
            row["technical_valid"]
            and row["worst_query_balanced_accuracy"] >= float(oracle_gate["worst_query_balanced_accuracy_minimum"])
            and row["exact_trial_accuracy"] >= float(oracle_gate["exact_trial_accuracy_minimum"])
            and row["minimum_window_activity_fraction"] >= float(oracle_gate["each_window_activity_fraction_minimum"])
            and row["maximum_schedule_error_steps"] <= 1e-7
        ))
    curriculum_gate = protocol["pilot_decision"]["curriculum_gate_each_passing_point"]
    curriculum_pass: dict[str, bool] = {}
    for row in [item for item in rows if item["condition"] == "shared_temporal_annealed"]:
        curriculum_pass[row["point_label"]] = bool(
            row["technical_valid"] and row["scaffold_source_joint_passed"] is True
            and row["worst_query_balanced_accuracy"] >= float(curriculum_gate["worst_query_balanced_accuracy_minimum"])
            and row["exact_trial_accuracy"] >= float(curriculum_gate["exact_trial_accuracy_minimum"])
            and row["minimum_window_activity_fraction"] >= float(curriculum_gate["each_window_activity_fraction_minimum"])
            and row["maximum_schedule_error_steps"] <= float(curriculum_gate["maximum_integer_oracle_schedule_error_steps"])
        )
    high_pass = curriculum_pass.get("highN_lowT", False) or curriculum_pass.get("highN_highT", False)
    sweep_gate = bool(
        all(row["technical_valid"] for row in rows)
        and all(required_oracles)
        and curriculum_pass.get("geometric_center", False)
        and high_pass
        and sum(curriculum_pass.values()) >= int(
            protocol["pilot_decision"]["curriculum_sweep_gate"]["total_landmarks_passing_minimum"]
        )
    )
    decision = {
        "protocol_id": PROTOCOL, "stage": stage, "cells": len(rows),
        "required_oracle_points_passed": all(required_oracles),
        "curriculum_landmark_pass": curriculum_pass,
        "curriculum_landmarks_passing": sum(curriculum_pass.values()),
        "full_sweep_authorized_by_results": sweep_gate,
        "single_seed_exploratory": True, "test_split_opened": False,
    }
    core._write_csv(output / "landmark_cells.csv", serializable)
    core._write_json(output / "landmark_decision.json", decision)
    _plot_landmarks(rows, output)
    return decision


def preconditions(protocol: dict[str, Any], stage: str) -> None:
    key = {"smoke": "smoke_launch", "landmark": "landmark_launch", "full": "full_sweep_launch"}[stage]
    if protocol["authorization"].get(key) is not True:
        raise SystemExit(f"{stage} launch locked")
    output = BASE / protocol["execution"]["generated_root"]
    if stage == "landmark":
        decision = _read_json(output / "smoke_decision.json")
        if decision.get("passed") is not True:
            raise SystemExit("landmark requires passing smoke")
    if stage == "full":
        decision = _read_json(output / "landmark_decision.json")
        if decision.get("full_sweep_authorized_by_results") is not True:
            raise SystemExit("full sweep landmark gate failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "landmark", "full"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    cells = specs(protocol, args.stage)
    if args.dry_run:
        print(json.dumps({
            "protocol": PROTOCOL, "stage": args.stage, "cells": len(cells),
            "paths": [str(run_dir(protocol, spec).relative_to(BASE)) for spec in cells],
        }, indent=2))
        return
    preconditions(protocol, args.stage)
    for spec in cells:
        run_cell(protocol, spec, args.device)
    decision = summarize(protocol, args.stage)
    print(json.dumps({"stage": args.stage, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()

"""Run the preregistered mixed-op temporal WAD scaffold withdrawal protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import scripts.run_mixedop_spatial_temporal_surface_preview as core
from train.eval import evaluate_simultaneous, save_eval_results
from train.trainer import window_class_balanced_bce
from utils.seed import set_seed
from utils.viz import save_run_diagnostic_plots


BASE = core.BASE
PROTOCOL = "mixedop_temporal_wad_scaffold_withdrawal_v1"
CONFIG_PATH = BASE / "configs" / f"{PROTOCOL}.yaml"
PARENT_CONFIG_PATH = BASE / "configs" / "mixedop_temporal_wad_repair_v3.yaml"


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL:
        raise ValueError("protocol id mismatch")
    if protocol["authorization"].get("sealed_test") is not False:
        raise ValueError("sealed test must remain locked")
    if protocol["w1_retention_withdrawal"]["optimizer_updates"] != 200:
        raise ValueError("W1 update budget must remain 200")
    if protocol["w2_local_restoration"]["optimizer_updates"] != 200:
        raise ValueError("W2 update budget must remain 200")
    return protocol


def _core_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    parent = yaml.safe_load(PARENT_CONFIG_PATH.read_text(encoding="utf-8"))
    parent["protocol_id"] = PROTOCOL
    parent["study_class"] = protocol["study_class"]
    return parent


def base_spec(seed: int, stage: str, arm: str, updates: int) -> dict[str, Any]:
    return {
        "K": 5,
        "condition": "shared_temporal_wad",
        "total_hidden": 120,
        "output_window_len": 4,
        "seed": int(seed),
        "updates": int(updates),
        "stage": "smoke" if stage == "smoke" else stage,
        "training_arm": arm,
        "path_variant": arm,
        "event_budget_label": "event8",
        "r_on_hz": 990.0,
        "routing_loss_kind": "arrival_centroid_huber",
        "routing_loss_weight": 1.0,
        "delay_credit_mode": "joint",
        "save_schedule_checkpoint": False,
    }


def specs(protocol: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "smoke":
        block = protocol["technical_smoke"]
        return [
            {**base_spec(block["seed"], stage, arm, block["optimizer_updates"]),
             "source_kind": "parent_v3"}
            for arm in block["arms"]
        ]
    if stage == "w0":
        block = protocol["w0_scaffold_replication"]
        return [
            {**base_spec(seed, stage, "centroid_scaffold", block["optimizer_updates"]),
             "source_kind": "fresh"}
            for seed in block["seeds"]
        ]
    if stage == "w1":
        block = protocol["w1_retention_withdrawal"]
        return [
            {**base_spec(seed, stage, arm, block["optimizer_updates"]),
             "source_kind": "w0"}
            for arm, seed in product(block["arms"].keys(), block["seeds"])
        ]
    if stage == "w2":
        block = protocol["w2_local_restoration"]
        return [
            {**base_spec(seed, stage, arm, block["optimizer_updates"]),
             "source_kind": "w0", "perturbation": perturbation,
             "path_variant": f"{perturbation}/{arm}"}
            for perturbation, arm, seed in product(
                block["perturbations"].keys(), block["arms"], block["seeds"]
            )
        ]
    raise ValueError(f"unknown stage: {stage}")


def build_config(protocol: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    cfg = core.build_config(_core_protocol(protocol), spec)
    cfg.update({
        "withdrawal_stage": spec["stage"],
        "withdrawal_arm": spec["training_arm"],
        "source_kind": spec["source_kind"],
        "perturbation": spec.get("perturbation"),
        "routing_loss_kind": "arrival_centroid_huber",
        "routing_loss_weight": 1.0,
        "optimizer_state_reset": True,
        "primary_checkpoint": "last_model.pt",
    })
    return cfg


def run_dir(protocol: dict[str, Any], spec: dict[str, Any]) -> Path:
    stage = spec["stage"]
    root_key = {
        "smoke": "smoke_root", "w0": "w0_root", "w1": "w1_root", "w2": "w2_root",
    }[stage]
    root = BASE / protocol["execution"][root_key]
    if stage == "w2":
        root = root / str(spec["perturbation"]) / str(spec["training_arm"])
    else:
        root = root / str(spec["training_arm"])
    return root / f"K5_T30_N120_w4_seed{spec['seed']}"


def _required(protocol: dict[str, Any]) -> list[str]:
    return list(protocol["required_artifacts"])


def complete(protocol: dict[str, Any], directory: Path) -> bool:
    return all((directory / item).exists() for item in _required(protocol))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parent_v3_checkpoint() -> Path:
    return (
        BASE / "runs/exploratory/mixedop_temporal_wad_repair_v3/formal_recovery"
        / "shared_temporal_wad/arrival_centroid_huber/K5/T30_N120_w4_seed3557"
        / "best_model.pt"
    )


def _w0_directory(protocol: dict[str, Any], seed: int) -> Path:
    spec = base_spec(seed, "w0", "centroid_scaffold", 400)
    spec["source_kind"] = "fresh"
    return run_dir(protocol, spec)


def source_checkpoint(protocol: dict[str, Any], spec: dict[str, Any]) -> Path | None:
    if spec["source_kind"] == "fresh":
        return None
    if spec["source_kind"] == "parent_v3":
        return _parent_v3_checkpoint()
    if spec["source_kind"] == "w0":
        return _w0_directory(protocol, int(spec["seed"])) / "joint_source_model.pt"
    raise ValueError(spec["source_kind"])


def physical_query_delays(model: torch.nn.Module) -> list[float]:
    values = model.syn_ih.get_delays().detach()
    return [float(values[4 * q:4 * (q + 1)].mean().item()) for q in range(5)]


def set_physical_query_delays(model: torch.nn.Module, values: list[float]) -> None:
    if len(values) != 5 or model.syn_ih.delay_raw.numel() != 5:
        raise ValueError("physical delay setter requires five query-tied parameters")
    d_max = float(model.syn_ih.d_max)
    tensor = torch.as_tensor(
        values, device=model.syn_ih.delay_raw.device, dtype=model.syn_ih.delay_raw.dtype
    )
    if bool(((tensor <= 0) | (tensor >= d_max)).any().item()):
        raise ValueError("perturbed delays must remain strictly interior")
    raw = torch.logit(tensor / d_max).reshape_as(model.syn_ih.delay_raw)
    with torch.no_grad():
        model.syn_ih.delay_raw.copy_(raw)
    recovered = torch.tensor(physical_query_delays(model), device=tensor.device)
    if not torch.allclose(recovered, tensor.float(), atol=1e-5, rtol=0):
        raise RuntimeError("physical-to-raw delay conversion is not reversible")


def apply_perturbation(
    protocol: dict[str, Any], model: torch.nn.Module, name: str | None
) -> tuple[list[float], list[float]]:
    before = physical_query_delays(model)
    if name is None:
        return before, before.copy()
    block = protocol["w2_local_restoration"]["perturbations"][name]
    if name == "uniform_early_2_steps":
        after = [value - 2.0 for value in before]
    elif name == "compressed_spacing":
        offsets = [float(value) for value in block["offsets_steps_by_query"]]
        after = [value + offset for value, offset in zip(before, offsets)]
    else:
        raise ValueError(f"unknown perturbation: {name}")
    set_physical_query_delays(model, after)
    return before, physical_query_delays(model)


def lambda_at_update(arm: str, update: int) -> float:
    if arm in {"centroid_scaffold", "centroid_continue", "centroid_restore"}:
        return 1.0
    if arm == "annealed_joint":
        if update <= 100:
            return float(100 - update) / 99.0
        return 0.0
    return 0.0


def compose_delay_gradient(
    arm: str, task_gradient: torch.Tensor, routing_gradient: torch.Tensor,
    routing_lambda: float,
) -> torch.Tensor:
    if arm in {"centroid_scaffold", "centroid_continue", "centroid_restore"}:
        return routing_gradient
    if arm == "annealed_joint":
        return task_gradient + float(routing_lambda) * routing_gradient
    if arm in {"abrupt_task_only", "task_only_restore"}:
        return task_gradient
    if arm == "delay_frozen":
        return torch.zeros_like(task_gradient)
    raise ValueError(f"unknown withdrawal arm: {arm}")


def _optimizer(model: torch.nn.Module, cfg: dict[str, Any], arm: str):
    groups = []
    weights = list(model.weight_params())
    readout = list(model.readout_params())
    delays = list(model.delay_params())
    if weights:
        groups.append({"params": weights, "lr": cfg["lr_w"]})
    if readout:
        groups.append({"params": readout, "lr": cfg["lr_readout"]})
    if arm not in {"delay_frozen"} and delays:
        groups.append({"params": delays, "lr": cfg["lr_d"]})
    return torch.optim.Adam(groups), weights, readout, delays


def _vector(values: list[torch.Tensor | None], like: torch.Tensor) -> torch.Tensor:
    if not values or values[0] is None:
        return torch.zeros_like(like)
    return values[0].detach().clone()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a, b = left.reshape(-1).float(), right.reshape(-1).float()
    denom = float(a.norm().item() * b.norm().item())
    return float(torch.dot(a, b).item() / denom) if denom > 0 else 0.0


def _joint_gate(protocol: dict[str, Any], metrics: dict[str, Any]) -> bool:
    gate = protocol["w0_scaffold_replication"]["gate_each_seed"]
    return bool(
        metrics["worst_query_balanced_accuracy"] >= float(gate["worst_query_balanced_accuracy_minimum"])
        and metrics["exact_trial_accuracy"] >= float(gate["exact_trial_accuracy_minimum"])
        and min(metrics["per_output_window_hidden_activity_fraction"])
        >= float(gate["each_window_activity_fraction_minimum"])
        and metrics["delay_query_schedule_max_abs_error_steps"]
        <= float(gate["maximum_integer_oracle_schedule_error_steps"])
    )


def _evaluate(
    model: torch.nn.Module, loader, cfg: dict[str, Any], device: str, encoder,
    collect: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    metrics, arrays = core.validation_snapshot(
        model, loader, cfg, device, encoder, collect=collect
    )
    metrics.update(core._delay_diagnostics(model, cfg))
    return metrics, arrays


def _load_state(model: torch.nn.Module, path: Path, device: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)


def selected_update_from_validation_log(directory: Path, checkpoint: str) -> int:
    rows = _read_csv(directory / "validation_log.csv")
    if not rows:
        raise ValueError(f"empty validation log: {directory}")
    best_update = int(rows[0]["update"])
    if checkpoint == "descriptive_best_model.pt":
        best = (
            float(rows[0]["worst_query_balanced_accuracy"]),
            float(rows[0]["exact_trial_accuracy"]),
        )
        for row in rows[1:]:
            candidate = (
                float(row["worst_query_balanced_accuracy"]),
                float(row["exact_trial_accuracy"]),
            )
            if candidate > best:
                best, best_update = candidate, int(row["update"])
        return best_update
    if checkpoint == "schedule_best_model.pt":
        best = (
            -float(rows[0]["delay_query_schedule_max_abs_error_steps"]),
            float(rows[0]["worst_query_balanced_accuracy"]),
            float(rows[0]["exact_trial_accuracy"]),
        )
        for row in rows[1:]:
            candidate = (
                -float(row["delay_query_schedule_max_abs_error_steps"]),
                float(row["worst_query_balanced_accuracy"]),
                float(row["exact_trial_accuracy"]),
            )
            if candidate > best:
                best, best_update = candidate, int(row["update"])
        return best_update
    raise ValueError(f"unknown source checkpoint: {checkpoint}")


def _save_final_artifacts(
    protocol: dict[str, Any], directory: Path, model: torch.nn.Module,
    validation_loader, cfg: dict[str, Any], device: str, encoder,
    update_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]],
    viz_rows: list[dict[str, Any]], started: float, initial_error: float,
) -> dict[str, Any]:
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
        "protocol_id": PROTOCOL, "stage": cfg["withdrawal_stage"],
        "arm": cfg["withdrawal_arm"], "seed": cfg["seed"],
        "perturbation": cfg.get("perturbation"),
        "primary_checkpoint": "last_model.pt",
        "selected_update": cfg["optimizer_updates"],
        "initial_schedule_max_error_steps": initial_error,
        "schedule_error_reduction_fraction": (
            (initial_error - final_metrics["delay_query_schedule_max_abs_error_steps"]) / initial_error
            if initial_error > 0 else 0.0
        ),
        "evaluation_split": cfg["evaluation_split"],
        "test_split_opened": False,
        "claim_status": "invalid_smoke" if cfg["withdrawal_stage"] == "smoke" else "exploratory",
        "wall_time_seconds": time.time() - started,
    }
    save_eval_results(results, str(directory / "final_validation_results.json"))
    np.savez_compressed(directory / "final_validation_predictions.npz", **arrays)
    core._write_json(directory / "resource_ledger.json", results["resource_ledger"])
    core._write_csv(directory / "update_log.csv", update_rows)
    core._write_csv(directory / "validation_log.csv", validation_rows)
    save_run_diagnostic_plots(
        model, cfg, viz_rows, results, str(directory), cfg["K"], "mixed", device,
        seed=cfg["seed"] + 10000, dataset_override=validation_loader.dataset[0],
    )
    return results


def run_cell(
    protocol: dict[str, Any], spec: dict[str, Any], device: str, dry_run: bool = False
) -> Path:
    directory = run_dir(protocol, spec)
    if dry_run:
        return directory
    if complete(protocol, directory):
        return directory
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"incomplete existing cell requires audit: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    cfg = build_config(protocol, spec)
    core._write_json(directory / "config.json", cfg)
    started = time.time()
    set_seed(cfg["seed"])
    model = core.build_model(cfg).to(device)

    source = source_checkpoint(protocol, spec)
    source_hash = None
    if source is not None:
        _load_state(model, source, device)
        source_hash = _sha256(source)
    source_loaded_delays = physical_query_delays(model)
    before, after = apply_perturbation(protocol, model, spec.get("perturbation"))
    core._write_json(directory / "source_checkpoint_provenance.json", {
        "source_kind": spec["source_kind"],
        "source_path": str(source.resolve()) if source is not None else None,
        "source_sha256": source_hash,
        "strict_state_dict_load": source is not None,
        "optimizer_state_loaded": False,
        "source_loaded_delay_vector": source_loaded_delays,
    })
    target = [float(value) for value in protocol["fixed_interface"]["integer_oracle_schedule_steps"]]
    initial_error = float(np.max(np.abs(np.asarray(after) - np.asarray(target))))
    core._write_json(directory / "initial_and_perturbed_delay_vectors.json", {
        "source_delay_vector": before, "perturbation": spec.get("perturbation"),
        "training_initial_delay_vector": after,
        "declared_integer_oracle_schedule": target,
        "initial_max_schedule_error_steps": initial_error,
    })

    train_loader, validation_loader = core.loaders(cfg)
    encoder = core.encode_fn(cfg)
    optimizer, weight_parameters, readout_parameters, delay_parameters = _optimizer(
        model, cfg, spec["training_arm"]
    )
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    update_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    viz_rows: list[dict[str, Any]] = []
    best_accuracy_score: tuple[float, float] | None = None
    best_schedule_score: tuple[float, float, float] | None = None
    best_accuracy_update = 0
    best_schedule_update = 0

    initial_metrics, _ = _evaluate(model, validation_loader, cfg, device, encoder, False)
    validation_rows.append({"update": 0, **initial_metrics})
    best_accuracy_score = (
        float(initial_metrics["worst_query_balanced_accuracy"]),
        float(initial_metrics["exact_trial_accuracy"]),
    )
    best_schedule_score = (
        -float(initial_metrics["delay_query_schedule_max_abs_error_steps"]),
        *best_accuracy_score,
    )
    torch.save(model.state_dict(), directory / "descriptive_best_model.pt")
    torch.save(model.state_dict(), directory / "schedule_best_model.pt")
    viz_rows.append({
        "epoch": 0, "train_loss": initial_metrics["loss"],
        "val_loss": initial_metrics["loss"],
        "train_acc": initial_metrics["pooled_accuracy"],
        "val_acc": initial_metrics["pooled_accuracy"],
        "val_worst_query_balanced_accuracy": initial_metrics["worst_query_balanced_accuracy"],
        "val_exact_trial_accuracy": initial_metrics["exact_trial_accuracy"],
        "weight_grad_norm": 0.0, "delay_grad_norm": 0.0,
    })

    iterator = iter(train_loader)
    for update in range(1, cfg["optimizer_updates"] + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        model.train()
        _, _, _, labels, spikes = core._batch_input(batch, cfg, device, encoder)
        optimizer.zero_grad(set_to_none=True)
        logits, info = model(spikes)
        task_loss = window_class_balanced_bce(logits, labels)
        routing_loss = core.routing_alignment_loss(spikes, model, cfg)
        delay_parameter = delay_parameters[0]
        task_gradient = _vector(list(torch.autograd.grad(
            task_loss, delay_parameters, retain_graph=True, allow_unused=True
        )), delay_parameter)
        routing_gradient = _vector(list(torch.autograd.grad(
            routing_loss, delay_parameters, retain_graph=True, allow_unused=True
        )), delay_parameter)
        task_loss.backward()
        arm = spec["training_arm"]
        lam = lambda_at_update(arm, update)
        combined = compose_delay_gradient(
            arm, task_gradient, routing_gradient, lam
        )
        delay_parameter.grad = None if arm == "delay_frozen" else combined.clone()
        gradients_finite = core._all_finite(
            p.grad for p in optimized if p.grad is not None
        )
        global_norm = torch.nn.utils.clip_grad_norm_(
            optimized, cfg["grad_clip"], error_if_nonfinite=False
        )
        optimizer.step()
        delays = physical_query_delays(model)
        combined_values = combined.detach().reshape(-1)
        task_values = task_gradient.detach().reshape(-1)
        route_values = routing_gradient.detach().reshape(-1)
        row: dict[str, Any] = {
            "update": update, "lambda_routing": lam,
            "task_loss": float(task_loss.detach().item()),
            "routing_loss": float(routing_loss.detach().item()),
            "train_pooled_accuracy": float(
                ((logits > 0).float() == labels).float().mean().item()
            ),
            "mean_hidden_spikes": float(info["total_hidden_spikes"].detach().mean().item()),
            "task_delay_gradient_norm": float(task_values.norm().item()),
            "routing_delay_gradient_norm": float(route_values.norm().item()),
            "combined_delay_gradient_norm": float(combined_values.norm().item()),
            "gradient_cosine_task_vs_routing": _cosine(task_gradient, routing_gradient),
            "global_grad_norm_before_clip": float(global_norm.item()),
            "loss_finite": bool(torch.isfinite(task_loss).item() and torch.isfinite(routing_loss).item()),
            "gradients_finite": gradients_finite,
            "parameters_finite": core._all_finite(p.detach() for p in model.parameters()),
            "delays_finite": bool(np.isfinite(delays).all()),
            "delays_legal": bool(min(delays) >= 0 and max(delays) <= cfg["d_max"]),
        }
        for q in range(5):
            row[f"delay_q{q}_steps"] = delays[q]
            row[f"task_delay_grad_q{q}"] = float(task_values[q].item())
            row[f"routing_delay_grad_q{q}"] = float(route_values[q].item())
            row[f"combined_delay_grad_q{q}"] = float(combined_values[q].item())
        update_rows.append(row)

        if update % cfg["validation_interval_updates"] == 0 or update == cfg["optimizer_updates"]:
            metrics, _ = _evaluate(model, validation_loader, cfg, device, encoder, False)
            validation_rows.append({"update": update, **metrics})
            score = (
                float(metrics["worst_query_balanced_accuracy"]),
                float(metrics["exact_trial_accuracy"]),
            )
            schedule_score = (
                -float(metrics["delay_query_schedule_max_abs_error_steps"]), *score
            )
            if score > best_accuracy_score:
                best_accuracy_score = score
                best_accuracy_update = update
                torch.save(model.state_dict(), directory / "descriptive_best_model.pt")
            if schedule_score > best_schedule_score:
                best_schedule_score = schedule_score
                best_schedule_update = update
                torch.save(model.state_dict(), directory / "schedule_best_model.pt")
            viz_rows.append({
                "epoch": update, "train_loss": row["task_loss"],
                "val_loss": metrics["loss"], "train_acc": row["train_pooled_accuracy"],
                "val_acc": metrics["pooled_accuracy"],
                "val_worst_query_balanced_accuracy": metrics["worst_query_balanced_accuracy"],
                "val_exact_trial_accuracy": metrics["exact_trial_accuracy"],
                "weight_grad_norm": float(global_norm.item()),
                "delay_grad_norm": row["combined_delay_gradient_norm"],
            })

    results = _save_final_artifacts(
        protocol, directory, model, validation_loader, cfg, device, encoder,
        update_rows, validation_rows, viz_rows, started, initial_error,
    )

    if spec["stage"] == "w0":
        selected_name = None
        selected_metrics = None
        for name in ("descriptive_best_model.pt", "schedule_best_model.pt"):
            _load_state(model, directory / name, device)
            metrics, _ = _evaluate(model, validation_loader, cfg, device, encoder, True)
            if _joint_gate(protocol, metrics):
                selected_name, selected_metrics = name, metrics
                break
        if selected_name is not None:
            _load_state(model, directory / selected_name, device)
            torch.save(model.state_dict(), directory / "joint_source_model.pt")
            selected_update = (
                best_accuracy_update if selected_name == "descriptive_best_model.pt"
                else best_schedule_update
            )
            core._write_json(directory / "joint_source_validation_results.json", {
                **selected_metrics, "source_checkpoint": selected_name,
                "selected_update": selected_update,
            })
            core._write_json(directory / "joint_source_provenance.json", {
                "source_checkpoint": selected_name,
                "selected_update": selected_update,
                "joint_source_model_sha256": _sha256(directory / "joint_source_model.pt"),
                "selection_rule": "accuracy_first_then_schedule_if_jointly_passing",
                "reconstructed_posthoc": False,
            })

    missing = [
        item for item in _required(protocol)
        if item != "run_complete.json" and not (directory / item).exists()
    ]
    if missing:
        raise RuntimeError(f"required withdrawal artifacts missing: {missing}")
    core._write_json(directory / "run_complete.json", {
        "protocol_id": PROTOCOL, "stage": spec["stage"], "arm": spec["training_arm"],
        "seed": spec["seed"], "completed": True, "test_split_opened": False,
        "wall_time_seconds": results["wall_time_seconds"],
    })
    return directory


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _provenance_valid(path: Path) -> bool:
    if not path.exists():
        return False
    value = _read_json(path)
    if value.get("optimizer_state_loaded") is not False:
        return False
    if value.get("source_kind") == "fresh":
        return (
            value.get("source_path") is None
            and value.get("source_sha256") is None
            and value.get("strict_state_dict_load") is False
        )
    source_value = value.get("source_path")
    if not source_value or value.get("strict_state_dict_load") is not True:
        return False
    source = Path(source_value)
    return bool(
        source.exists() and value.get("source_sha256") == _sha256(source)
    )


def summarize(protocol: dict[str, Any], stage: str) -> dict[str, Any]:
    generated = BASE / protocol["execution"]["generated_root"]
    rows = []
    for spec in specs(protocol, stage):
        directory = run_dir(protocol, spec)
        result = _read_json(directory / "final_validation_results.json") if complete(protocol, directory) else {}
        logs = _read_csv(directory / "update_log.csv") if result else []
        row = {
            "stage": stage, "arm": spec["training_arm"], "seed": spec["seed"],
            "perturbation": spec.get("perturbation"), "artifacts_complete": bool(result),
            "provenance_valid": _provenance_valid(
                directory / "source_checkpoint_provenance.json"
            ),
            "finite_and_legal": bool(logs) and all(
                item["loss_finite"] == "True" and item["gradients_finite"] == "True"
                and item["parameters_finite"] == "True" and item["delays_finite"] == "True"
                and item["delays_legal"] == "True" for item in logs
            ),
        }
        row["technical_valid"] = bool(
            row["artifacts_complete"] and row["finite_and_legal"]
            and row["provenance_valid"]
        )
        if result:
            row.update({
                "worst_query_balanced_accuracy": result["worst_query_balanced_accuracy"],
                "exact_trial_accuracy": result["exact_trial_accuracy"],
                "minimum_window_activity_fraction": min(result["per_output_window_hidden_activity_fraction"]),
                "maximum_schedule_error_steps": result["delay_query_schedule_max_abs_error_steps"],
                "schedule_error_reduction_fraction": result["schedule_error_reduction_fraction"],
            })
        rows.append(row)

    if stage == "smoke":
        by_arm = {row["arm"]: row for row in rows}
        abrupt_logs = _read_csv(run_dir(protocol, next(
            spec for spec in specs(protocol, stage) if spec["training_arm"] == "abrupt_task_only"
        )) / "update_log.csv")
        anneal_logs = _read_csv(run_dir(protocol, next(
            spec for spec in specs(protocol, stage) if spec["training_arm"] == "annealed_joint"
        )) / "update_log.csv")
        passed = bool(rows) and all(
            row["technical_valid"] for row in rows
        ) and any(float(item["task_delay_gradient_norm"]) > 0 for item in abrupt_logs) \
          and all(math.isclose(
              float(item["lambda_routing"]), lambda_at_update("annealed_joint", int(item["update"])),
              abs_tol=1e-12,
          ) for item in anneal_logs)
        decision = {"stage": stage, "passed": passed, "rows": rows}
    elif stage == "w0":
        passed_by_seed = {}
        for spec, row in zip(specs(protocol, stage), rows):
            directory = run_dir(protocol, spec)
            result_path = directory / "joint_source_validation_results.json"
            source_joint_pass = False
            if result_path.exists() and (directory / "joint_source_model.pt").exists():
                source_result = _read_json(result_path)
                source_name = str(source_result["source_checkpoint"])
                selected_update = selected_update_from_validation_log(directory, source_name)
                source_result["selected_update"] = selected_update
                core._write_json(result_path, source_result)
                core._write_json(directory / "joint_source_provenance.json", {
                    "source_checkpoint": source_name,
                    "selected_update": selected_update,
                    "joint_source_model_sha256": _sha256(directory / "joint_source_model.pt"),
                    "selection_rule": "accuracy_first_then_schedule_if_jointly_passing",
                    "reconstructed_posthoc": True,
                    "reconstruction_source": "immutable_validation_log.csv",
                })
                source_joint_pass = _joint_gate(protocol, source_result)
                row.update({
                    "joint_source_checkpoint": source_name,
                    "joint_source_selected_update": selected_update,
                    "joint_source_worst_query_balanced_accuracy": source_result["worst_query_balanced_accuracy"],
                    "joint_source_exact_trial_accuracy": source_result["exact_trial_accuracy"],
                    "joint_source_minimum_window_activity_fraction": min(
                        source_result["per_output_window_hidden_activity_fraction"]
                    ),
                    "joint_source_maximum_schedule_error_steps": source_result["delay_query_schedule_max_abs_error_steps"],
                })
            passed_by_seed[str(spec["seed"])] = bool(
                row["technical_valid"] and (directory / "joint_source_model.pt").exists()
                and (directory / "joint_source_provenance.json").exists()
                and source_joint_pass
            )
        decision = {
            "stage": stage, "passed_by_seed": passed_by_seed,
            "passed": all(passed_by_seed.values()), "rows": rows,
        }
    elif stage == "w1":
        gate = protocol["w1_retention_withdrawal"]["gate_each_seed"]
        arm_seed_pass: dict[str, dict[str, bool]] = {}
        for arm in protocol["w1_retention_withdrawal"]["arms"]:
            arm_seed_pass[arm] = {}
            for row in [item for item in rows if item["arm"] == arm]:
                passed = bool(
                    row["technical_valid"]
                    and row["worst_query_balanced_accuracy"] >= float(gate["worst_query_balanced_accuracy_minimum"])
                    and row["exact_trial_accuracy"] >= float(gate["exact_trial_accuracy_minimum"])
                    and row["minimum_window_activity_fraction"] >= float(gate["each_window_activity_fraction_minimum"])
                    and row["maximum_schedule_error_steps"] <= float(gate["maximum_integer_oracle_schedule_error_steps"])
                )
                if arm == "annealed_joint":
                    spec = next(item for item in specs(protocol, stage) if item["training_arm"] == arm and item["seed"] == row["seed"])
                    late = [item for item in _read_csv(run_dir(protocol, spec) / "validation_log.csv") if int(item["update"]) >= 100]
                    passed = passed and all(float(item["delay_query_schedule_max_abs_error_steps"]) <= 1.5 for item in late)
                arm_seed_pass[arm][str(row["seed"])] = passed
        arm_pass = {arm: all(values.values()) for arm, values in arm_seed_pass.items()}
        decision = {
            "stage": stage, "arm_seed_pass": arm_seed_pass, "arm_pass": arm_pass,
            "withdrawal_retention_passed": bool(
                arm_pass.get("abrupt_task_only") or arm_pass.get("annealed_joint")
            ), "rows": rows,
        }
    else:
        gate = protocol["w2_local_restoration"]["gate_each_seed_and_perturbation"]
        pass_map: dict[str, dict[str, dict[str, bool]]] = {}
        for arm in protocol["w2_local_restoration"]["arms"]:
            pass_map[arm] = {}
            for perturbation in protocol["w2_local_restoration"]["perturbations"]:
                selected = [row for row in rows if row["arm"] == arm and row["perturbation"] == perturbation]
                pass_map[arm][perturbation] = {}
                for row in selected:
                    joint = bool(
                        row["technical_valid"]
                        and row["worst_query_balanced_accuracy"] >= float(gate["task_only_worst_query_balanced_accuracy_minimum"])
                        and row["exact_trial_accuracy"] >= float(gate["task_only_exact_trial_accuracy_minimum"])
                        and row["minimum_window_activity_fraction"] >= float(gate["task_only_each_window_activity_fraction_minimum"])
                        and row["maximum_schedule_error_steps"] <= float(gate["task_only_maximum_integer_oracle_schedule_error_steps"])
                    )
                    if arm == "task_only_restore":
                        joint = joint and row["schedule_error_reduction_fraction"] >= float(
                            gate["task_only_schedule_error_reduction_from_perturbed_initial_minimum_fraction"]
                        )
                    pass_map[arm][perturbation][str(row["seed"])] = joint
        task_pass = all(
            all(seed_values.values())
            for seed_values in pass_map["task_only_restore"].values()
        )
        centroid_pass = all(
            all(seed_values.values())
            for seed_values in pass_map["centroid_restore"].values()
        )
        decision = {
            "stage": stage, "pass_map": pass_map,
            "positive_centroid_control_passed": centroid_pass,
            "local_task_restoration_passed": bool(task_pass and centroid_pass),
            "rows": rows,
        }
    core._write_json(generated / f"{stage}_decision.json", decision)
    core._write_csv(generated / f"{stage}_cells.csv", rows)
    return decision


def preconditions(protocol: dict[str, Any], stage: str) -> None:
    auth_key = {
        "smoke": "implementation_smoke_launch", "w0": "w0_launch",
        "w1": "w1_launch", "w2": "w2_launch",
    }[stage]
    if protocol["authorization"].get(auth_key) is not True:
        raise SystemExit(f"{stage} launch is locked")
    generated = BASE / protocol["execution"]["generated_root"]
    prerequisite = {"w0": "smoke", "w1": "w0", "w2": "w1"}.get(stage)
    if prerequisite:
        path = generated / f"{prerequisite}_decision.json"
        if not path.exists():
            raise SystemExit(f"missing prerequisite decision: {path}")
        decision = _read_json(path)
        passed = (
            decision.get("passed") if prerequisite in {"smoke", "w0"}
            else decision.get("withdrawal_retention_passed")
        )
        if passed is not True:
            raise SystemExit(f"{stage} prerequisite did not pass")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "w0", "w1", "w2"), required=True)
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

"""Run interior-support and decoupled-credit temporal WAD repair v2."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
import yaml

import scripts.run_mixedop_spatial_temporal_surface_preview as core
from train.trainer import window_class_balanced_bce
from utils.seed import set_seed


BASE = core.BASE
PROTOCOL = "mixedop_temporal_wad_repair_v2"
CONFIG_PATH = BASE / "configs" / f"{PROTOCOL}.yaml"


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL:
        raise ValueError("protocol id mismatch")
    delay = protocol["model"]["temporal_delay"]
    if delay["oracle_schedule_steps"][-1] >= delay["delay_support_steps"][-1]:
        raise ValueError("all oracle targets must be interior to delay support")
    return protocol


def base_spec(seed: int, stage: str) -> dict[str, Any]:
    return {
        "K": 5,
        "condition": "shared_temporal_wad",
        "total_hidden": 120,
        "output_window_len": 4,
        "seed": int(seed),
        "updates": 0,
        "stage": stage,
        "event_budget_label": "event8",
        "r_on_hz": 990.0,
        "save_schedule_checkpoint": True,
        "schedule_gate_max_error_steps": 1.0,
    }


def gradient_preflight(protocol: dict[str, Any], device: str) -> dict[str, Any]:
    records = []
    gradients: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed in protocol["gradient_preflight"]["seeds"]:
        spec = base_spec(seed, "gradient_preflight")
        cfg = core.build_config(protocol, spec)
        set_seed(seed)
        model = core.build_model(cfg).to(device)
        train, _ = core.loaders(cfg)
        batch = next(iter(train))
        encoder = core.encode_fn(cfg)
        _, _, _, labels, spikes = core._batch_input(batch, cfg, device, encoder)
        logits, _ = model(spikes)
        task = window_class_balanced_bce(logits, labels)
        route = core.routing_alignment_loss(spikes, model, cfg)
        parameter = model.syn_ih.delay_raw
        g_task = torch.autograd.grad(task, parameter, retain_graph=True)[0]
        g_route = torch.autograd.grad(route, parameter)[0]
        task_values = g_task.detach().cpu().reshape(-1).numpy()
        route_values = g_route.detach().cpu().reshape(-1).numpy()
        gradients[int(seed)] = (task_values, route_values)
        records.append({
            "seed": seed,
            "task_loss": float(task.detach().item()),
            "routing_loss": float(route.detach().item()),
            "task_gradient": task_values.tolist(),
            "route_gradient": route_values.tolist(),
            "task_gradient_norm": float(np.linalg.norm(task_values)),
            "route_gradient_norm": float(np.linalg.norm(route_values)),
            "route_q1_to_q4_target_directed": bool(np.all(route_values[1:] < 0)),
            "initial_delay_means": core._delay_diagnostics(model, cfg)["delay_query_mean_steps"],
        })
    ratios = [
        row["task_gradient_norm"] / max(row["route_gradient_norm"], 1e-12)
        for row in records
    ]
    base_lambda = float(median(ratios))
    candidates = []
    for multiplier in protocol["gradient_preflight"]["joint_multipliers"]:
        value = base_lambda * float(multiplier)
        per_seed = {}
        for seed, (task_values, route_values) in gradients.items():
            combined = task_values + value * route_values
            per_seed[str(seed)] = {
                "combined_gradient": combined.tolist(),
                "q1_to_q4_target_directed": bool(np.all(combined[1:] < 0)),
            }
        candidates.append({
            "multiplier": float(multiplier),
            "lambda": value,
            "passed": all(item["q1_to_q4_target_directed"] for item in per_seed.values()),
            "per_seed": per_seed,
        })
    selected = next((item for item in candidates if item["passed"]), None)
    decoupled_pass = all(row["route_q1_to_q4_target_directed"] for row in records)
    decision = {
        "protocol_id": PROTOCOL,
        "stage": "gradient_preflight",
        "base_balancing_lambda": base_lambda,
        "joint_candidates": candidates,
        "selected_joint": selected,
        "decoupled_route_gradient_gate_pass": decoupled_pass,
        "training_authorized_by_results": bool(selected is not None or decoupled_pass),
        "records": records,
        "test_split_opened": False,
    }
    output = BASE / protocol["execution"]["generated_root"]
    core._write_json(output / "gradient_preflight_decision.json", decision)
    core._write_csv(output / "gradient_preflight_seeds.csv", [
        {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
        for row in records
    ])
    return decision


def _decision(protocol: dict[str, Any]) -> dict[str, Any]:
    return json.loads(
        (BASE / protocol["execution"]["generated_root"] / "gradient_preflight_decision.json").read_text(
            encoding="utf-8"
        )
    )


def specs(protocol: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    decision = _decision(protocol)
    selected = decision.get("selected_joint")
    arms = ["oracle", "task_only"]
    if selected is not None:
        arms.append("joint_calibrated")
    if decision.get("decoupled_route_gradient_gate_pass"):
        arms.append("routing_only_for_delays")
    if stage == "smoke":
        seeds = [protocol["smoke"]["seed"]]
        updates = int(protocol["smoke"]["optimizer_updates"])
    elif stage == "formal":
        seeds = protocol["formal_recovery"]["seeds"]
        updates = int(protocol["formal_recovery"]["optimizer_updates"])
    else:
        raise ValueError(stage)
    rows = []
    for arm, seed in product(arms, seeds):
        spec = base_spec(seed, "smoke" if stage == "smoke" else "formal_recovery")
        spec.update({
            "condition": "shared_temporal_oracle" if arm == "oracle" else "shared_temporal_wad",
            "training_arm": arm,
            "updates": updates,
            "routing_loss_weight": (
                float(selected["lambda"]) if arm == "joint_calibrated"
                else 1.0 if arm == "routing_only_for_delays" else 0.0
            ),
            "routing_loss_temperature": 0.75,
            "delay_credit_mode": (
                "routing_only_for_delays" if arm == "routing_only_for_delays" else "joint"
            ),
        })
        rows.append(spec)
    return rows


def run_dir(protocol: dict[str, Any], spec: dict[str, Any]) -> Path:
    return core._run_directory(protocol, core.build_config(protocol, spec))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_smoke(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for spec in specs(protocol, "smoke"):
        directory = run_dir(protocol, spec)
        complete = core._complete(directory, protocol)
        logs = read_csv(directory / "update_log.csv") if complete else []
        row = {
            "arm": spec["training_arm"],
            "artifacts_complete": complete,
            "finite": bool(logs) and all(
                item["loss_finite"] == "True" and item["gradients_finite"] == "True"
                and item["parameters_finite"] == "True" and item["delays_finite"] == "True"
                for item in logs
            ),
            "delays_legal": bool(logs) and all(item["delays_legal"] == "True" for item in logs),
            "schedule_artifacts_present": all((directory / name).exists() for name in (
                "best_schedule_model.pt", "schedule_validation_results.json",
                "schedule_validation_predictions.npz",
            )),
            "runtime_panel_present": (directory / "plots/diagnostic_panel.png").exists(),
        }
        row["passed"] = all(value for key, value in row.items() if key != "arm")
        rows.append(row)
    decision = {
        "protocol_id": PROTOCOL,
        "stage": "smoke",
        "cells": len(rows),
        "invalid_for_claims": True,
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "rows": rows,
    }
    output = BASE / protocol["execution"]["generated_root"]
    core._write_json(output / "smoke_decision.json", decision)
    core._write_csv(output / "smoke_cells.csv", rows)
    return decision


def _checkpoint_row(path: Path, checkpoint: str, arm: str, seed: int) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    return {
        "arm": arm,
        "seed": seed,
        "checkpoint": checkpoint,
        "worst_query_balanced_accuracy": result["worst_query_balanced_accuracy"],
        "exact_trial_accuracy": result["exact_trial_accuracy"],
        "minimum_window_activity_fraction": min(result["per_output_window_hidden_activity_fraction"]),
        "maximum_query_delay_error_steps": result["delay_query_schedule_max_abs_error_steps"],
        "delay_query_means": result["delay_query_mean_steps"],
        "selected_update": result["selected_update"],
    }


def summarize_formal(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for spec in specs(protocol, "formal"):
        directory = run_dir(protocol, spec)
        if not core._complete(directory, protocol):
            raise RuntimeError(f"incomplete formal cell: {directory}")
        rows.append(_checkpoint_row(
            directory / "validation_results.json", "accuracy", spec["training_arm"], spec["seed"]
        ))
        rows.append(_checkpoint_row(
            directory / "schedule_validation_results.json", "schedule", spec["training_arm"], spec["seed"]
        ))
    gates = protocol["formal_recovery"]["gate"]
    arms = sorted({row["arm"] for row in rows})
    arm_pass = {}
    seed_pass = {}
    for arm in arms:
        oracle = arm == "oracle"
        per_seed = {}
        for seed in protocol["formal_recovery"]["seeds"]:
            candidates = [row for row in rows if row["arm"] == arm and row["seed"] == seed]
            passes = []
            for row in candidates:
                passes.append(bool(
                    row["worst_query_balanced_accuracy"] >= float(
                        gates["oracle_worst_bacc_each_seed_minimum"] if oracle
                        else gates["learned_worst_bacc_each_seed_minimum"]
                    )
                    and row["exact_trial_accuracy"] >= float(
                        gates["oracle_exact_each_seed_minimum"] if oracle
                        else gates["learned_exact_each_seed_minimum"]
                    )
                    and row["minimum_window_activity_fraction"] >= float(gates["each_window_activity_fraction_minimum"])
                    and (oracle or row["maximum_query_delay_error_steps"] <= float(gates["maximum_query_delay_error_steps"]))
                ))
            per_seed[str(seed)] = any(passes)
        seed_pass[arm] = per_seed
        arm_pass[arm] = all(per_seed.values())
    decision = {
        "protocol_id": PROTOCOL,
        "stage": "formal_recovery",
        "cells": len(rows) // 2,
        "checkpoint_rows": len(rows),
        "arm_seed_pass": seed_pass,
        "arm_pass": arm_pass,
        "withdrawal_authorized_by_results": any(
            arm_pass.get(arm, False) for arm in ("joint_calibrated", "routing_only_for_delays")
        ),
        "test_split_opened": False,
    }
    output = BASE / protocol["execution"]["generated_root"]
    core._write_csv(output / "formal_checkpoint_rows.csv", [
        {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
        for row in rows
    ])
    core._write_json(output / "formal_decision.json", decision)
    return decision


def preconditions(protocol: dict[str, Any], stage: str) -> None:
    auth = protocol["authorization"]
    if stage == "smoke" and auth.get("smoke_launch") is not True:
        raise SystemExit("smoke locked")
    if stage == "formal":
        if auth.get("formal_launch") is not True:
            raise SystemExit("formal locked")
        smoke = json.loads((BASE / protocol["execution"]["generated_root"] / "smoke_decision.json").read_text(encoding="utf-8"))
        if smoke.get("passed") is not True:
            raise SystemExit("formal requires passing smoke")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "smoke", "formal"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    if args.stage == "preflight":
        if protocol["authorization"].get("gradient_preflight") is not True:
            raise SystemExit("gradient preflight locked")
        decision = gradient_preflight(protocol, args.device)
        print(json.dumps({
            "stage": "preflight",
            "selected_joint": decision["selected_joint"],
            "decoupled_pass": decision["decoupled_route_gradient_gate_pass"],
        }))
        return
    cells = specs(protocol, args.stage)
    if args.dry_run:
        print(json.dumps({
            "protocol": PROTOCOL, "stage": args.stage, "cells": len(cells),
            "paths": [str(run_dir(protocol, spec).relative_to(BASE)) for spec in cells],
        }, indent=2))
        return
    preconditions(protocol, args.stage)
    for spec in cells:
        core.run_cell(protocol, spec, args.device)
    if args.stage == "smoke":
        decision = audit_smoke(protocol)
        print(json.dumps({"stage": "smoke", "passed": decision["passed"]}))
    else:
        decision = summarize_formal(protocol)
        print(json.dumps({"stage": "formal", "arm_pass": decision["arm_pass"]}))


if __name__ == "__main__":
    main()

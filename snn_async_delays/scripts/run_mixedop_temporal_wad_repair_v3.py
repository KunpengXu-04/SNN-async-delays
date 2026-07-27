"""Run schedule-aligned temporal WAD objective repair v3."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import scripts.run_mixedop_spatial_temporal_surface_preview as core
from utils.seed import set_seed


BASE = core.BASE
PROTOCOL = "mixedop_temporal_wad_repair_v3"
CONFIG_PATH = BASE / "configs" / f"{PROTOCOL}.yaml"


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL:
        raise ValueError("protocol id mismatch")
    delay = protocol["model"]["temporal_delay"]
    if delay["oracle_schedule_steps"][-1] >= delay["delay_support_steps"][-1]:
        raise ValueError("oracle targets must be interior to delay support")
    return protocol


def base_spec(seed: int, stage: str) -> dict[str, Any]:
    return {
        "K": 5, "condition": "shared_temporal_wad", "total_hidden": 120,
        "output_window_len": 4, "seed": int(seed), "updates": 0,
        "stage": stage, "event_budget_label": "event8", "r_on_hz": 990.0,
        "save_schedule_checkpoint": True, "schedule_gate_max_error_steps": 1.0,
    }


def gradient_preflight(protocol: dict[str, Any], device: str) -> dict[str, Any]:
    rows = []
    for seed in protocol["gradient_preflight"]["seeds"]:
        spec = base_spec(seed, "gradient_preflight")
        spec.update({"routing_loss_kind": "arrival_centroid_huber"})
        cfg = core.build_config(protocol, spec)
        set_seed(seed)
        model = core.build_model(cfg).to(device)
        train, _ = core.loaders(cfg)
        batch = next(iter(train))
        _, _, _, _, spikes = core._batch_input(
            batch, cfg, device, core.encode_fn(cfg)
        )
        loss = core.routing_alignment_loss(spikes, model, cfg)
        gradient = torch.autograd.grad(loss, model.syn_ih.delay_raw)[0]
        values = gradient.detach().cpu().reshape(-1).numpy()
        rows.append({
            "seed": int(seed), "routing_loss": float(loss.detach().item()),
            "raw_delay_gradient": values.tolist(),
            "all_five_target_directed": bool(np.all(values < 0)),
            "initial_delay_means": core._delay_diagnostics(model, cfg)["delay_query_mean_steps"],
        })
    passed = bool(rows) and all(row["all_five_target_directed"] for row in rows)
    decision = {
        "protocol_id": PROTOCOL, "stage": "gradient_preflight",
        "passed": passed, "records": rows, "test_split_opened": False,
    }
    output = BASE / protocol["execution"]["generated_root"]
    core._write_json(output / "gradient_preflight_decision.json", decision)
    core._write_csv(output / "gradient_preflight_seeds.csv", [
        {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
        for row in rows
    ])
    return decision


def specs(protocol: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    block = protocol["smoke" if stage == "smoke" else "formal_recovery"]
    stage_label = "smoke" if stage == "smoke" else "formal_recovery"
    seeds = [block["seed"]] if stage == "smoke" else block["seeds"]
    rows = []
    for arm, seed in product(block["arms"], seeds):
        spec = base_spec(seed, stage_label)
        spec.update({
            "training_arm": arm, "updates": int(block["optimizer_updates"]),
            "condition": "shared_temporal_oracle" if arm == "oracle" else "shared_temporal_wad",
            "routing_loss_weight": 0.0 if arm == "oracle" else 1.0,
            "routing_loss_kind": (
                "arrival_centroid_huber" if arm == "arrival_centroid_huber" else "arrival_mass_ce"
            ),
            "routing_loss_temperature": 0.75,
            "delay_credit_mode": "joint" if arm == "oracle" else "routing_only_for_delays",
        })
        rows.append(spec)
    return rows


def run_dir(protocol: dict[str, Any], spec: dict[str, Any]) -> Path:
    return core._run_directory(protocol, core.build_config(protocol, spec))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_smoke(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for spec in specs(protocol, "smoke"):
        directory = run_dir(protocol, spec)
        complete = core._complete(directory, protocol)
        logs = _csv(directory / "update_log.csv") if complete else []
        row = {
            "arm": spec["training_arm"], "artifacts_complete": complete,
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
        "protocol_id": PROTOCOL, "stage": "smoke", "cells": len(rows),
        "invalid_for_claims": True,
        "passed": bool(rows) and all(row["passed"] for row in rows), "rows": rows,
    }
    output = BASE / protocol["execution"]["generated_root"]
    core._write_json(output / "smoke_decision.json", decision)
    core._write_csv(output / "smoke_cells.csv", rows)
    return decision


def checkpoint_row(path: Path, checkpoint: str, arm: str, seed: int) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    return {
        "arm": arm, "seed": seed, "checkpoint": checkpoint,
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
        for filename, checkpoint in (
            ("validation_results.json", "accuracy"),
            ("schedule_validation_results.json", "schedule"),
        ):
            rows.append(checkpoint_row(
                directory / filename, checkpoint, spec["training_arm"], spec["seed"]
            ))
    gates = protocol["formal_recovery"]["gate"]
    seed_pass: dict[str, dict[str, bool]] = {}
    arm_pass: dict[str, bool] = {}
    for arm in protocol["formal_recovery"]["arms"]:
        oracle = arm == "oracle"
        per_seed = {}
        for seed in protocol["formal_recovery"]["seeds"]:
            candidates = [row for row in rows if row["arm"] == arm and row["seed"] == seed]
            per_seed[str(seed)] = any(
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
                for row in candidates
            )
        seed_pass[arm] = per_seed
        arm_pass[arm] = all(per_seed.values())
    decision = {
        "protocol_id": PROTOCOL, "stage": "formal_recovery", "cells": len(rows) // 2,
        "checkpoint_rows": len(rows), "arm_seed_pass": seed_pass, "arm_pass": arm_pass,
        "withdrawal_authorized_by_results": arm_pass.get("arrival_centroid_huber", False),
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
    if auth.get(f"{stage}_launch") is not True:
        raise SystemExit(f"{stage} locked")
    generated = BASE / protocol["execution"]["generated_root"]
    prerequisite = "gradient_preflight_decision.json" if stage == "smoke" else "smoke_decision.json"
    decision = json.loads((generated / prerequisite).read_text(encoding="utf-8"))
    if decision.get("passed") is not True:
        raise SystemExit(f"{stage} prerequisite failed")


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
        print(json.dumps({"stage": "preflight", "passed": decision["passed"]}))
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
    decision = audit_smoke(protocol) if args.stage == "smoke" else summarize_formal(protocol)
    summary = decision["passed"] if args.stage == "smoke" else decision["arm_pass"]
    print(json.dumps({"stage": args.stage, "decision": summary}))


if __name__ == "__main__":
    main()

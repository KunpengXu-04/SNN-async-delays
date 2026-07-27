"""Validate and materialize the gated capacity/SLAYER protocol.

Large scientific stages are intentionally launch-locked in YAML.  ``--dry-run``
is always safe and prints the exact immutable paths that a stage would use.
The S0 stage delegates to ``scripts.run_slayer_s0`` because it uses the
isolated Lava-DL environment rather than the historical project environment.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import yaml


BASE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "spatial_temporal_capacity_slayer_v1"
CONFIG_PATH = BASE / "configs" / f"{PROTOCOL_ID}.yaml"


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("protocol id mismatch")
    K_values = list(map(int, protocol["workload"]["K_values"]))
    if K_values != [1, 2, 3, 4, 6, 8]:
        raise ValueError("registered K grid changed")
    for budget in protocol["temporal_scaling"]["output_budgets_B"]:
        if any(int(budget) % K for K in K_values):
            raise ValueError("every temporal budget must be divisible by every K")
    spatial = protocol["spatial_scaling"]
    if int(spatial["total_latency_T"]) != (
        int(protocol["workload"]["input_window_steps"])
        + int(spatial["total_output_budget_B"])
    ):
        raise ValueError("spatial B/T mapping changed")
    screening = protocol["boundary_screening"]
    if any(int(screening["output_budget_B"]) % int(K) for K in screening["K_values"]):
        raise ValueError("boundary-screening budget must divide every K")
    if int(screening["cells"]) != (
        len(screening["K_values"])
        * len(screening["hidden_neurons_total"])
        * len(screening["seeds"])
    ):
        raise ValueError("boundary-screening cell count changed")
    if protocol["encoding"]["expected_input_events_per_query"] != (
        2 * int(protocol["encoding"]["packet_steps"])
    ):
        raise ValueError("packet event budget changed")
    if protocol["data"]["test_split_opened"] or protocol["authorization"]["sealed_test"]:
        raise ValueError("sealed test must remain closed")
    return protocol


def cyclic_rotations(values: list[str]) -> list[list[str]]:
    return [values[offset:] + values[:offset] for offset in range(len(values))]


def _base_spec(
    protocol: dict[str, Any], *, stage: str, condition: str, K: int,
    hidden: int, budget: int, seed: int,
) -> dict[str, Any]:
    if budget % K:
        raise ValueError("output budget must divide exactly into K windows")
    input_steps = int(protocol["workload"]["input_window_steps"])
    hidden_lif = protocol["model"]["hidden_lif"]
    return {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "condition": condition,
        "K": int(K),
        "N_hidden_total": int(hidden),
        "output_budget_B": int(budget),
        "window_width": int(budget // K),
        "T": int(input_steps + budget),
        "seed": int(seed),
        "model_backend": "slayer_native" if "slayer" in condition else "current",
        "delay_method": (
            "task_only_slayer" if condition == "task_only_slayer"
            else "task_only_current" if condition == "current_task_only"
            else "explicit_centroid" if "centroid" in condition
            else "fixed" if "fixed" in condition
            else "d0"
        ),
        "delay_granularity": protocol["delay_granularity"],
        "encoding_mode": protocol["encoding"]["mode"],
        "output_interface": protocol["interfaces"]["primary_output"],
        "lif_tau_m": float(hidden_lif["tau_m_steps"]),
        "lif_threshold": float(hidden_lif["threshold_au"]),
        "lif_reset": float(hidden_lif["reset_au"]),
        "lif_refractory": int(hidden_lif["refractory_steps"]),
        "surrogate_beta": float(hidden_lif["surrogate_beta"]),
        "input_events_expected": int(
            K * protocol["encoding"]["expected_input_events_per_query"]
        ),
        "test_split_opened": False,
    }


def specs(protocol: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    gates = protocol["gates"]
    if stage == "s0":
        return [
            {
                "protocol_id": PROTOCOL_ID, "stage": "s0", "seed": int(seed),
                "claim_status": "invalid_smoke", "model_backend": "slayer_native",
            }
            for seed in gates["s0"]["seeds"]
        ]
    if stage == "s1_calibration":
        grid = protocol["optimization"]["slayer_calibration"]
        return [
            {
                **_base_spec(
                    protocol, stage=stage, condition="task_only_slayer", K=1,
                    hidden=8, budget=24, seed=seed,
                ),
                "weight_learning_rate": float(lrw),
                "delay_learning_rate": float(lrd),
            }
            for seed, lrw, lrd in product(
                gates["s1"]["calibration_seeds"],
                grid["weight_learning_rates"], grid["delay_learning_rates"],
            )
        ]
    if stage == "s1_confirmation":
        conditions = [
            "current_d0", "current_fixed", "current_explicit_centroid",
            "current_task_only", "slayer_fixed_interface", "task_only_slayer",
        ]
        return [
            _base_spec(
                protocol, stage=stage, condition=condition, K=1, hidden=8,
                budget=24, seed=seed,
            )
            for condition, seed in product(
                conditions, gates["s1"]["confirmation_seeds"]
            )
        ]
    if stage == "s2":
        conditions = [
            "shared_d0", "shared_temporal_fixed",
            "shared_temporal_explicit_centroid", "slayer_fixed_interface",
            "task_only_slayer",
        ]
        return [
            _base_spec(
                protocol, stage=stage, condition=condition, K=2, hidden=8,
                budget=24, seed=seed,
            )
            for condition, seed in product(
                conditions, gates["s2"]["confirmation_seeds"]
            )
        ]
    if stage == "spatial_calibration":
        block = protocol["spatial_scaling"]
        return [
            _base_spec(
                protocol, stage=stage, condition="shared_temporal_fixed", K=6,
                hidden=hidden, budget=block["total_output_budget_B"], seed=seed,
            )
            for hidden, seed in product(
                block["hidden_neurons_total"], gates["s1"]["calibration_seeds"]
            )
        ]
    if stage == "boundary_screening":
        block = protocol["boundary_screening"]
        return [
            _base_spec(
                protocol, stage=stage, condition=block["condition"], K=K,
                hidden=hidden, budget=int(block["output_budget_B"]), seed=seed,
            )
            for K, hidden, seed in product(
                block["K_values"], block["hidden_neurons_total"], block["seeds"],
            )
        ]
    if stage == "spatial":
        block = protocol["spatial_scaling"]
        conditions = [
            "independent_spatial_d0", "shared_d0", "shared_temporal_fixed"
        ]
        s2_path = BASE / protocol["execution"]["generated_root"] / "s2_decision.json"
        if s2_path.exists() and json.loads(s2_path.read_text(encoding="utf-8")).get("passed"):
            conditions.append("task_only_slayer")
        rows = []
        for condition, K, hidden, seed in product(
            conditions, block["K_values"], block["hidden_neurons_total"],
            block["formal_seeds"],
        ):
            if condition == "independent_spatial_d0" and hidden % K:
                continue
            rows.append(_base_spec(
                protocol, stage=stage, condition=condition, K=K,
                hidden=hidden, budget=block["total_output_budget_B"], seed=seed,
            ))
        return rows
    if stage == "temporal":
        block = protocol["temporal_scaling"]
        # N_ref is deliberately unresolved until the registered spatial rule
        # produces docs/generated/.../N_ref_decision.json.
        decision_path = BASE / protocol["execution"]["generated_root"] / "N_ref_decision.json"
        if not decision_path.exists():
            raise RuntimeError("temporal specs require the frozen N_ref decision artifact")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        hidden = int(decision["N_ref"])
        conditions = list(block["conditions"])
        s2_path = BASE / protocol["execution"]["generated_root"] / "s2_decision.json"
        if s2_path.exists() and json.loads(s2_path.read_text(encoding="utf-8")).get("passed"):
            conditions.append("task_only_slayer")
        return [
            _base_spec(
                protocol, stage=stage, condition=condition, K=K,
                hidden=hidden, budget=budget, seed=seed,
            )
            for condition, K, budget, seed in product(
                conditions, block["K_values"],
                block["output_budgets_B"], gates["s1"]["confirmation_seeds"],
            )
        ]
    if stage == "centroid_validation":
        decision_path = (
            BASE / protocol["execution"]["generated_root"]
            / "centroid_validation_points.json"
        )
        if not decision_path.exists():
            raise RuntimeError("centroid validation requires selected fixed-oracle points")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        return [
            _base_spec(
                protocol, stage=stage,
                condition="shared_temporal_explicit_centroid", K=int(point["K"]),
                hidden=int(point["N_hidden_total"]), budget=48, seed=int(seed),
            )
            for point, seed in product(
                decision["selected_points"], gates["s1"]["confirmation_seeds"]
            )
        ]
    if stage == "counterbalance":
        block = protocol["counterbalance"]
        rotations = cyclic_rotations(list(block["operations"]))
        rows = []
        for rotation_id, ops in enumerate(rotations):
            for hidden, T, seed in product(
                block["hidden_neurons_total"], block["total_latency_steps"],
                block["seeds"],
            ):
                budget = int(T) - int(protocol["workload"]["input_window_steps"])
                row = _base_spec(
                    protocol, stage=stage, condition="explicit_centroid", K=6,
                    hidden=hidden, budget=budget, seed=seed,
                )
                row.update({"rotation_id": rotation_id, "query_ops": ops})
                rows.append(row)
        return rows
    raise ValueError(f"unknown stage: {stage}")


def run_directory(protocol: dict[str, Any], spec: dict[str, Any]) -> Path:
    root_key = "smoke_root" if spec["stage"] == "s0" else "exploratory_root"
    root = BASE / protocol["execution"][root_key] / spec["stage"]
    if spec["stage"] == "s0":
        return root / f"seed{spec['seed']}"
    suffix = f"K{spec['K']}_N{spec['N_hidden_total']}_B{spec['output_budget_B']}_seed{spec['seed']}"
    if "weight_learning_rate" in spec:
        suffix += f"_lrw{spec['weight_learning_rate']}_lrd{spec['delay_learning_rate']}"
    if "rotation_id" in spec:
        suffix += f"_rot{spec['rotation_id']}"
    return root / spec["condition"] / suffix


def authorization_key(stage: str) -> str:
    return {
        "s0": "s0_launch",
        "s1_calibration": "s1_calibration_launch",
        "s1_confirmation": "s1_confirmation_launch",
        "s2": "s2_launch",
        "spatial_calibration": "spatial_calibration_launch",
        "boundary_screening": "boundary_screening_launch",
        "spatial": "spatial_scaling_launch",
        "centroid_validation": "centroid_validation_launch",
        "temporal": "temporal_scaling_launch",
        "counterbalance": "counterbalance_launch",
    }[stage]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True,
        choices=["s0", "s1_calibration", "s1_confirmation", "s2", "spatial_calibration", "boundary_screening", "spatial", "centroid_validation", "temporal", "counterbalance"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cell-index", type=int)
    parser.add_argument(
        "--resume", action="store_true",
        help="skip only cells that already contain validation_results.json",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_protocol()
    cells = specs(protocol, args.stage)
    paths = [str(run_directory(protocol, cell)) for cell in cells]
    if args.dry_run:
        print(json.dumps({
            "protocol_id": PROTOCOL_ID, "stage": args.stage,
            "cells": len(cells), "authorized": bool(
                protocol["authorization"][authorization_key(args.stage)]
            ),
            "specs": cells, "paths": paths,
        }, indent=2))
        return
    if not protocol["authorization"][authorization_key(args.stage)]:
        raise RuntimeError(f"{args.stage} launch remains locked in the protocol")
    if args.stage == "s1_calibration":
        decision_path = BASE / protocol["execution"]["generated_root"] / "s0_decision.json"
        if not decision_path.exists() or not json.loads(
            decision_path.read_text(encoding="utf-8")
        ).get("passed", False):
            raise RuntimeError("S1 calibration requires a passing registered S0 decision")
        selected = cells if args.cell_index is None else [cells[args.cell_index]]
        from scripts.run_slayer_trainability import run_cell
        for cell in selected:
            run_cell(protocol, cell, run_directory(protocol, cell), device=args.device)
        return
    if args.stage != "s0":
        generated = BASE / protocol["execution"]["generated_root"]
        prerequisite = {
            "s1_confirmation": "s1_calibration_decision.json",
            "s2": "s1_confirmation_decision.json",
        }.get(args.stage)
        recipe = None
        if prerequisite is not None:
            decision_path = generated / prerequisite
            if not decision_path.exists():
                raise RuntimeError(f"{args.stage} requires {prerequisite}")
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            if not decision.get("passed", False):
                raise RuntimeError(f"{args.stage} prerequisite did not pass")
            recipe = decision.get("selected_recipe")
        elif any(cell["model_backend"] == "slayer_native" for cell in cells):
            s2_path = generated / "s2_decision.json"
            s1_path = generated / "s1_confirmation_decision.json"
            if not s2_path.exists() or not json.loads(
                s2_path.read_text(encoding="utf-8")
            ).get("passed", False):
                raise RuntimeError("learned-delay scaling requires a passing S2 decision")
            recipe = json.loads(s1_path.read_text(encoding="utf-8"))["selected_recipe"]
        selected = cells if args.cell_index is None else [cells[args.cell_index]]
        from scripts.run_capacity_current_backend import run_cell as run_current
        from scripts.run_slayer_trainability import run_cell as run_slayer
        for original in selected:
            cell = dict(original)
            if cell["model_backend"] == "slayer_native" and recipe is not None:
                cell["weight_learning_rate"] = recipe["weight_learning_rate"]
                cell["delay_learning_rate"] = recipe["delay_learning_rate"]
            runner = run_slayer if cell["model_backend"] == "slayer_native" else run_current
            output = run_directory(protocol, original)
            if args.resume and (output / "validation_results.json").exists():
                print(f"skipping completed cell: {output}", flush=True)
                continue
            runner(protocol, cell, output, device=args.device)
        return
    from scripts.run_slayer_s0 import run_registered_s0
    run_registered_s0(protocol)


if __name__ == "__main__":
    main()

"""Run the preregistered dimension-aware XOR micro-burst rescue."""

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
import yaml

from scripts import run_xor_delay_granularity_level1b as level1b
from scripts import run_xor_delay_granularity_rescue_level1br as rescue


BASE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "xor_delay_granularity_rescue_microburst_v1"
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


def expected_control_cells(protocol: dict[str, Any]) -> int:
    return len(protocol["optimization"]["fresh_microburst_seeds"])


def expected_learned_cells(protocol: dict[str, Any]) -> int:
    return (
        len(protocol["stage_b1_learned_rescue"]["conditions"])
        * len(protocol["optimization"]["learned_delay_initial_raw_values"])
        * len(protocol["optimization"]["fresh_microburst_seeds"])
    )


def control_specs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    condition = protocol["stage_b0_fixed_oracle_replication"]["condition"]
    return [
        {
            "stage": "stage_b0_fixed_control",
            "condition": str(condition["name"]),
            "encoding": "consecutive_microburst",
            "granularity": str(condition["granularity"]),
            "delay_tying": str(condition["tying"]),
            "independent_delay_parameters": int(condition["independent_delay_parameters"]),
            "normalization_factor": 1.0,
            "arrival_auxiliary_weight": 0.0,
            "base_arrival_lambda": float(
                protocol["losses"]["per_parameter_arrival_centroid"]["base_lambda"]
            ),
            "arrival_condition": "none_fixed_control",
            "seed": int(seed),
            "learned_delay": False,
            "fixed_delay_steps": float(condition["fixed_input_hidden_delay_steps"]),
            "target_delay_steps": float(protocol["timing"]["target_delay_steps"]),
            "initial_raw": None,
            "weight_learning_rate": float(protocol["optimization"]["weight_learning_rate"]),
            "delay_learning_rate": 0.0,
            "full_batch_updates": int(protocol["optimization"]["full_batch_updates"]),
            "selection_role": "same_seed_feasibility_gate",
        }
        for seed in protocol["optimization"]["fresh_microburst_seeds"]
    ]


def learned_specs(
    protocol: dict[str, Any], control_decision: dict[str, Any]
) -> list[dict[str, Any]]:
    if not bool(control_decision.get("learned_stage_authorized")):
        raise RuntimeError(
            "learned micro-burst rescue is locked because fixed-d4 replication failed"
        )
    specs: list[dict[str, Any]] = []
    base_lambda = float(
        protocol["losses"]["per_parameter_arrival_centroid"]["base_lambda"]
    )
    for condition in protocol["stage_b1_learned_rescue"]["conditions"]:
        for initial_raw in protocol["optimization"]["learned_delay_initial_raw_values"]:
            for seed in protocol["optimization"]["fresh_microburst_seeds"]:
                specs.append(
                    {
                        "stage": "stage_b1_learned",
                        "condition": str(condition["name"]),
                        "encoding": "consecutive_microburst",
                        "granularity": str(condition["granularity"]),
                        "delay_tying": str(condition["tying"]),
                        "independent_delay_parameters": int(
                            condition["independent_delay_parameters"]
                        ),
                        "normalization_factor": float(condition["normalization_factor"]),
                        "arrival_auxiliary_weight": float(condition["effective_lambda"]),
                        "base_arrival_lambda": base_lambda,
                        "arrival_condition": str(condition["name"]),
                        "seed": int(seed),
                        "learned_delay": True,
                        "fixed_delay_steps": None,
                        "target_delay_steps": float(protocol["timing"]["target_delay_steps"]),
                        "initial_raw": float(initial_raw),
                        "weight_learning_rate": float(
                            protocol["optimization"]["weight_learning_rate"]
                        ),
                        "delay_learning_rate": float(
                            protocol["optimization"]["delay_learning_rate"]
                        ),
                        "full_batch_updates": int(
                            protocol["optimization"]["full_batch_updates"]
                        ),
                        "selection_role": str(condition["selection_role"]),
                    }
                )
    return specs


def _token(value: Any) -> str:
    if value is None:
        return "none"
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def cell_directory(root: Path, spec: dict[str, Any]) -> Path:
    base = root / str(spec["stage"]) / str(spec["condition"])
    if not bool(spec["learned_delay"]):
        return base / f"seed_{spec['seed']}"
    return base / f"init_{_token(spec['initial_raw'])}" / f"seed_{spec['seed']}"


def run_cell(
    protocol: dict[str, Any], spec: dict[str, Any], *, root: Path, device: str
) -> dict[str, Any]:
    return rescue.run_cell(
        protocol,
        spec,
        root=root,
        device=device,
        protocol_id=PROTOCOL_ID,
        directory_builder=cell_directory,
    )


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_result(
    rows: list[dict[str, Any]], condition: str, expected: int = 10
) -> dict[str, Any]:
    selected = [row for row in rows if str(row["condition"]) == condition]
    return {
        "condition": condition,
        "granularity": selected[0]["granularity"] if selected else None,
        "arrival_auxiliary_weight": (
            float(selected[0]["arrival_auxiliary_weight"]) if selected else None
        ),
        "complete_cells": len(selected),
        "passing_cells": sum(bool(row["learned_delay_pass"]) for row in selected),
        "interface_passing_cells": sum(bool(row["interface_pass"]) for row in selected),
        "all_coordinate_direction_cells": sum(
            float(row["initial_total_gradient_correct_coordinate_fraction"]) == 1.0
            and float(row["initial_total_gradient_nonzero_coordinate_fraction"]) == 1.0
            for row in selected
        ),
        "full_delay_coverage_cells": sum(
            float(row["final_delay_fraction_within_tolerance"]) == 1.0
            for row in selected
        ),
        "gradient_clipping_flagged_cells": sum(
            bool(row["gradient_clipping_flag"]) for row in selected
        ),
        "candidate_pass": len(selected) == expected
        and all(bool(row["learned_delay_pass"]) for row in selected),
    }


def _gate_plot(candidates: list[dict[str, Any]], output: Path) -> None:
    labels = [str(item["condition"]).replace("_", "\n") for item in candidates]
    x = np.arange(len(labels))
    width = 0.22
    fig, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    axis.bar(
        x - 1.5 * width,
        [item["passing_cells"] for item in candidates],
        width,
        label="full gate",
    )
    axis.bar(
        x - 0.5 * width,
        [item["interface_passing_cells"] for item in candidates],
        width,
        label="interface",
    )
    axis.bar(
        x + 0.5 * width,
        [item["all_coordinate_direction_cells"] for item in candidates],
        width,
        label="initial direction",
    )
    axis.bar(
        x + 1.5 * width,
        [item["full_delay_coverage_cells"] for item in candidates],
        width,
        label="final coverage",
    )
    axis.axhline(10, linestyle="--", color="tab:red", label="10/10 gate")
    axis.set_xticks(x, labels)
    axis.set(
        ylabel="passing cells",
        ylim=(0, 10.8),
        title="Dimension-aware XOR micro-burst gates",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=5)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def aggregate_control(
    protocol: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    output = SUMMARY_ROOT / "stage_b0_fixed_control"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output / "cells.csv")
    expected = expected_control_cells(protocol)
    passing = sum(bool(row["interface_pass"]) for row in rows)
    gate = len(rows) == expected and passing == expected
    decision = {
        "protocol_id": PROTOCOL_ID,
        "stage": "stage_b0_fixed_control",
        "expected_cells": expected,
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == expected,
        "fixed_d4_interface_passing_cells": passing,
        "fixed_d4_gate_pass": gate,
        "learned_stage_authorized": gate,
        "K_greater_than_one_authorized": False,
        "test_split_opened": False,
    }
    level1b._strict_write_json(output / "decision.json", decision)
    return decision


def aggregate_learned(
    protocol: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    output = SUMMARY_ROOT / "stage_b1_learned"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output / "cells.csv")
    names = [
        str(item["name"])
        for item in protocol["stage_b1_learned_rescue"]["conditions"]
    ]
    candidates = [_candidate_result(rows, name) for name in names]
    by_name = {item["condition"]: item for item in candidates}
    primary_pass = bool(
        by_name["global_anchor"]["candidate_pass"]
        and by_name["per_hidden_dimension_matched"]["candidate_pass"]
    )
    decision = {
        "protocol_id": PROTOCOL_ID,
        "stage": "stage_b1_learned",
        "expected_cells": expected_learned_cells(protocol),
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == expected_learned_cells(protocol),
        "candidate_results": candidates,
        "global_anchor_pass": bool(by_name["global_anchor"]["candidate_pass"]),
        "per_hidden_primary_pass": bool(
            by_name["per_hidden_dimension_matched"]["candidate_pass"]
        ),
        "per_synapse_secondary_pass": bool(
            by_name["per_synapse_dimension_matched"]["candidate_pass"]
        ),
        "per_hidden_task_only_pass": bool(
            by_name["per_hidden_task_only"]["candidate_pass"]
        ),
        "dimension_aware_microburst_bridge_pass": primary_pass,
        "selected_granularity": "per_hidden_neuron" if primary_pass else None,
        "task_derived_timing_test_authorized": primary_pass,
        "K_greater_than_one_authorized": False,
        "test_split_opened": False,
    }
    level1b._strict_write_json(output / "decision.json", decision)
    _gate_plot(candidates, output / "microburst_rescue_gate_summary.png")
    return decision


def _read_control_decision() -> dict[str, Any]:
    path = SUMMARY_ROOT / "stage_b0_fixed_control" / "decision.json"
    if not path.exists():
        raise RuntimeError(
            "learned micro-burst rescue requires the complete formal fixed-control decision"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run_stage(
    protocol: dict[str, Any], *, stage: str, root: Path, device: str, smoke: bool
) -> dict[str, Any]:
    if stage == "control":
        specs = control_specs(protocol)
        if smoke:
            specs = [{**specs[0], "seed": 99001}]
        rows = [run_cell(protocol, spec, root=root, device=device) for spec in specs]
        return {"smoke_cells": len(rows), "cells": rows} if smoke else aggregate_control(protocol, rows)
    decision = {"learned_stage_authorized": True} if smoke else _read_control_decision()
    specs = learned_specs(protocol, decision)
    if smoke:
        by_condition = {}
        for spec in specs:
            by_condition.setdefault(str(spec["condition"]), spec)
        specs = [
            {**spec, "seed": 99011 + index}
            for index, spec in enumerate(by_condition.values())
        ]
    rows = [run_cell(protocol, spec, root=root, device=device) for spec in specs]
    return {"smoke_cells": len(rows), "cells": rows} if smoke else aggregate_learned(protocol, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["control", "learned"], default="control")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    declared = {
        "control": int(
            protocol["stage_b0_fixed_oracle_replication"]["grid"]["deterministic_cells"]
        ),
        "learned": int(
            protocol["stage_b1_learned_rescue"]["grid"]["deterministic_cells"]
        ),
    }
    expected = {
        "control": expected_control_cells(protocol),
        "learned": expected_learned_cells(protocol),
    }
    if declared != expected:
        raise SystemExit(f"declared cell counts do not match generated grids: {declared} vs {expected}")
    if not args.smoke and not args.dry_run:
        status_key = {
            "control": "stage_b0_fixed_oracle_replication",
            "learned": "stage_b1_learned_rescue",
        }[args.stage]
        if protocol[status_key]["status"] != "preregistered_ready":
            raise SystemExit(f"{args.stage} stage is not launch-ready")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "stage": args.stage,
                    "formal_cells": expected[args.stage],
                    "conditional_lock": args.stage == "learned",
                    "test_split_opened": False,
                    "K_greater_than_one_authorized": False,
                },
                indent=2,
            )
        )
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

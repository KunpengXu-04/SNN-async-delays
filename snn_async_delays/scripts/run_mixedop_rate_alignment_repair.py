"""Run the gated late-rate alignment repair and conditional per-query WAD."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import scripts.run_mixedop_spatial_temporal_surface_preview as core


BASE = core.BASE
PROTOCOL = "mixedop_rate_alignment_repair_v1"
CONFIG_PATH = BASE / "configs" / f"{PROTOCOL}.yaml"


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL:
        raise ValueError("protocol id mismatch")
    encoding = protocol["encoding"]
    timing = protocol["timing"]
    if int(encoding["rate_start_step"]) + int(encoding["rate_steps"]) != 10:
        raise ValueError("late-rate packet must end at the input-window boundary")
    expected_base = 10 - int(encoding["rate_start_step"]) - int(
        timing["buffer_effective_offset_steps"]
    )
    if int(timing["oracle_base_delay_steps"]) != expected_base:
        raise ValueError("oracle base delay does not align the packet")
    return protocol


def event_candidate(protocol: dict[str, Any], label: str) -> dict[str, float]:
    return protocol["encoding"]["event_budget_candidates"][label]


def stage_protocol(protocol: dict[str, Any], stage: str) -> dict[str, Any]:
    value = deepcopy(protocol)
    if stage == "stage_b":
        value["execution"]["formal_root"] = value["execution"]["stage_b_root"]
    return value


def specs(protocol: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "smoke":
        smoke = protocol["stage_a_oracle"]["smoke"]
        return [
            {
                "K": 5,
                "condition": "shared_temporal_oracle",
                "total_hidden": int(smoke["total_hidden_neurons"]),
                "output_window_len": int(smoke["output_window_length"]),
                "seed": int(smoke["seed"]),
                "updates": int(smoke["optimizer_updates"]),
                "stage": "smoke",
                "event_budget_label": label,
                "path_variant": label,
                "candidate_label": f"{label}_N{smoke['total_hidden_neurons']}_w{smoke['output_window_length']}",
                "r_on_hz": float(event_candidate(protocol, label)["r_on_hz"]),
            }
            for label in smoke["event_budgets"]
        ]
    if stage == "stage_a":
        formal = protocol["stage_a_oracle"]["formal"]
        return [
            {
                "K": 5,
                "condition": "shared_temporal_oracle",
                "total_hidden": int(hidden),
                "output_window_len": int(window),
                "seed": int(seed),
                "updates": int(protocol["optimization"]["optimizer_updates"]),
                "stage": "stage_a_oracle",
                "event_budget_label": label,
                "path_variant": label,
                "candidate_label": f"{label}_N{hidden}_w{window}",
                "r_on_hz": float(event_candidate(protocol, label)["r_on_hz"]),
            }
            for label, hidden, window, seed in product(
                formal["event_budgets"],
                formal["total_hidden_neurons"],
                formal["output_window_lengths"],
                formal["seeds"],
            )
        ]
    if stage == "stage_b":
        decision = json.loads(
            (BASE / protocol["execution"]["generated_root"] / "stage_a_decision.json").read_text(
                encoding="utf-8"
            )
        )
        selected = decision.get("selected_candidate")
        if not selected:
            raise RuntimeError("Stage A selected no interface")
        b = protocol["stage_b_per_query_wad"]
        rows = []
        for arm, seed in product(b["arms"], b["seeds"]):
            oracle = arm == "oracle_new_seed_control"
            assisted = arm == "routing_assisted"
            rows.append(
                {
                    "K": 5,
                    "condition": (
                        "shared_temporal_oracle" if oracle else "shared_temporal_wad"
                    ),
                    "training_arm": arm,
                    "total_hidden": int(selected["N"]),
                    "output_window_len": int(selected["w"]),
                    "seed": int(seed),
                    "updates": int(protocol["optimization"]["optimizer_updates"]),
                    "stage": "stage_b_per_query_wad",
                    "event_budget_label": selected["event_budget_label"],
                    "candidate_label": selected["candidate_label"],
                    "r_on_hz": float(selected["r_on_hz"]),
                    "routing_loss_weight": (
                        float(b["routing_assisted"]["weight"]) if assisted else 0.0
                    ),
                    "routing_loss_temperature": float(
                        b["routing_assisted"]["sigmoid_temperature_steps"]
                    ),
                }
            )
        return rows
    raise ValueError(stage)


def _run_dir(protocol: dict[str, Any], spec: dict[str, Any]) -> Path:
    return core._run_directory(protocol, core.build_config(protocol, spec))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_smoke(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for spec in specs(protocol, "smoke"):
        run_dir = _run_dir(protocol, spec)
        complete = core._complete(run_dir, protocol)
        updates = _read_csv(run_dir / "update_log.csv") if complete else []
        result = json.loads((run_dir / "validation_results.json").read_text(encoding="utf-8")) if complete else {}
        row = {
            "event_budget_label": spec["event_budget_label"],
            "artifacts_complete": complete,
            "finite": bool(updates) and all(
                item["loss_finite"] == "True"
                and item["gradients_finite"] == "True"
                and item["parameters_finite"] == "True"
                and item["delays_finite"] == "True"
                for item in updates
            ),
            "nonzero_hidden_activity": bool(result.get("mean_hidden_spikes", 0) > 0),
            "oracle_schedule_exact": result.get("oracle_schedule_exact") is True,
            "delays_legal": bool(updates) and all(item["delays_legal"] == "True" for item in updates),
            "runtime_panel_present": (run_dir / "plots/diagnostic_panel.png").exists(),
            "runtime_npz_present": (run_dir / "plots/diagnostic_data.npz").exists(),
            "resource_ledger_present": (run_dir / "resource_ledger.json").exists(),
        }
        row["passed"] = all(value for key, value in row.items() if key != "event_budget_label")
        rows.append(row)
    decision = {
        "protocol_id": PROTOCOL,
        "stage": "smoke",
        "invalid_for_claims": True,
        "cells": len(rows),
        "passed": len(rows) == 2 and all(row["passed"] for row in rows),
        "rows": rows,
    }
    output = BASE / protocol["execution"]["generated_root"]
    core._write_json(output / "smoke_decision.json", decision)
    core._write_csv(output / "smoke_cells.csv", rows)
    return decision


def summarize_stage_a(protocol: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in specs(protocol, "stage_a"):
        run_dir = _run_dir(protocol, spec)
        if not core._complete(run_dir, protocol):
            raise RuntimeError(f"incomplete Stage-A cell: {run_dir}")
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        result = json.loads((run_dir / "validation_results.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "candidate_label": spec["candidate_label"],
                "event_budget_label": spec["event_budget_label"],
                "r_on_hz": cfg["r_on"],
                "N": cfg["surface_total_hidden"],
                "w": cfg["output_window_len"],
                "T": cfg["T"],
                "seed": cfg["seed"],
                "worst_query_balanced_accuracy": result["worst_query_balanced_accuracy"],
                "exact_trial_accuracy": result["exact_trial_accuracy"],
                "mean_balanced_accuracy": result["balanced_accuracy"],
                "per_query_balanced_accuracy": result["per_query_balanced_accuracy"],
                "window_activity_fraction": result["per_output_window_hidden_activity_fraction"],
                "minimum_window_activity_fraction": min(result["per_output_window_hidden_activity_fraction"]),
                "oracle_schedule_exact": result["oracle_schedule_exact"],
                "mean_input_events": float(np.mean(result["mean_input_events_per_query"])),
                "neuron_update_proxy": result["neuron_update_proxy_N_times_T"],
                "run_dir": str(run_dir.relative_to(BASE)),
            }
        )
    rules = protocol["stage_a_oracle"]["candidate_pass"]
    candidates = []
    for label in sorted({row["candidate_label"] for row in rows}):
        group = [row for row in rows if row["candidate_label"] == label]
        passed = (
            len(group) == 3
            and all(row["worst_query_balanced_accuracy"] >= float(rules["worst_query_balanced_accuracy_minimum"]) for row in group)
            and all(row["exact_trial_accuracy"] >= float(rules["exact_trial_accuracy_minimum"]) for row in group)
            and all(row["minimum_window_activity_fraction"] >= float(rules["each_window_hidden_activity_fraction_minimum"]) for row in group)
            and all(row["oracle_schedule_exact"] for row in group)
        )
        first = group[0]
        candidates.append(
            {
                "candidate_label": label,
                "event_budget_label": first["event_budget_label"],
                "r_on_hz": first["r_on_hz"],
                "N": first["N"],
                "w": first["w"],
                "T": first["T"],
                "passed": passed,
                "minimum_worst_query_balanced_accuracy": min(row["worst_query_balanced_accuracy"] for row in group),
                "minimum_exact_trial_accuracy": min(row["exact_trial_accuracy"] for row in group),
                "minimum_window_activity_fraction": min(row["minimum_window_activity_fraction"] for row in group),
                "mean_input_events": float(np.mean([row["mean_input_events"] for row in group])),
                "neuron_update_proxy": first["neuron_update_proxy"],
            }
        )
    passing = [item for item in candidates if item["passed"]]
    passing.sort(key=lambda item: (
        item["mean_input_events"], item["neuron_update_proxy"], item["N"], item["T"]
    ))
    selected = passing[0] if passing else None
    decision = {
        "protocol_id": PROTOCOL,
        "stage": "stage_a_oracle",
        "cells": len(rows),
        "passing_candidates": len(passing),
        "selected_candidate": selected,
        "stage_b_authorized_by_results": selected is not None,
        "test_split_opened": False,
        "candidates": candidates,
    }
    output = BASE / protocol["execution"]["generated_root"]
    core._write_csv(output / "stage_a_cells.csv", [
        {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
        for row in rows
    ])
    core._write_csv(output / "stage_a_candidates.csv", candidates)
    core._write_json(output / "stage_a_decision.json", decision)
    return decision


def summarize_stage_b(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for spec in specs(protocol, "stage_b"):
        run_dir = _run_dir(protocol, spec)
        if not core._complete(run_dir, protocol):
            raise RuntimeError(f"incomplete Stage-B cell: {run_dir}")
        result = json.loads((run_dir / "validation_results.json").read_text(encoding="utf-8"))
        rows.append({
            "arm": spec["training_arm"],
            "seed": spec["seed"],
            "worst_query_balanced_accuracy": result["worst_query_balanced_accuracy"],
            "exact_trial_accuracy": result["exact_trial_accuracy"],
            "minimum_window_activity_fraction": min(result["per_output_window_hidden_activity_fraction"]),
            "maximum_query_delay_error_steps": result["delay_query_schedule_max_abs_error_steps"],
            "delay_query_means": result["delay_query_mean_steps"],
            "run_dir": str(run_dir.relative_to(BASE)),
        })
    gates = protocol["stage_b_per_query_wad"]["gates"]
    decisions = {}
    for arm in protocol["stage_b_per_query_wad"]["arms"]:
        group = [row for row in rows if row["arm"] == arm]
        oracle = arm == "oracle_new_seed_control"
        decisions[arm] = bool(
            len(group) == 3
            and all(row["worst_query_balanced_accuracy"] >= float(
                gates["oracle_each_seed_worst_bacc_minimum"] if oracle
                else gates["wad_each_seed_worst_bacc_minimum"]
            ) for row in group)
            and all(row["exact_trial_accuracy"] >= float(
                gates["oracle_each_seed_exact_minimum"] if oracle
                else gates["wad_each_seed_exact_minimum"]
            ) for row in group)
            and all(row["minimum_window_activity_fraction"] >= float(gates["each_window_hidden_activity_fraction_minimum"]) for row in group)
            and (oracle or all(row["maximum_query_delay_error_steps"] <= float(gates["maximum_query_delay_error_steps"]) for row in group))
        )
    decision = {
        "protocol_id": PROTOCOL,
        "stage": "stage_b_per_query_wad",
        "cells": len(rows),
        "arm_pass": decisions,
        "autonomous_task_only_supported": decisions.get("task_only", False),
        "routing_assisted_supported": decisions.get("routing_assisted", False),
        "stage_c_authorized_by_results": decisions.get("routing_assisted", False),
        "test_split_opened": False,
    }
    output = BASE / protocol["execution"]["generated_root"]
    core._write_csv(output / "stage_b_cells.csv", [
        {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
        for row in rows
    ])
    core._write_json(output / "stage_b_decision.json", decision)
    return decision


def preconditions(protocol: dict[str, Any], stage: str) -> None:
    auth = protocol["authorization"]
    key = {"smoke": "stage_a_smoke_launch", "stage_a": "stage_a_formal_launch", "stage_b": "stage_b_launch"}[stage]
    if auth.get(key) is not True:
        raise SystemExit(f"{stage} is locked")
    if stage == "stage_a":
        smoke = json.loads((BASE / protocol["execution"]["generated_root"] / "smoke_decision.json").read_text(encoding="utf-8"))
        if smoke.get("passed") is not True:
            raise SystemExit("Stage A requires passing smoke")
    if stage == "stage_b":
        decision = json.loads((BASE / protocol["execution"]["generated_root"] / "stage_a_decision.json").read_text(encoding="utf-8"))
        if decision.get("stage_b_authorized_by_results") is not True:
            raise SystemExit("Stage A selected no viable interface")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "stage_a", "stage_b"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = stage_protocol(load_protocol(), args.stage)
    cells = specs(protocol, args.stage)
    expected = {"smoke": 2, "stage_a": 48, "stage_b": 9}[args.stage]
    if len(cells) != expected:
        raise RuntimeError(f"expected {expected} cells, got {len(cells)}")
    if args.dry_run:
        print(json.dumps({
            "protocol": PROTOCOL,
            "stage": args.stage,
            "cells": len(cells),
            "paths": [str(_run_dir(protocol, spec).relative_to(BASE)) for spec in cells],
        }, indent=2))
        return
    preconditions(protocol, args.stage)
    for spec in cells:
        core.run_cell(protocol, spec, args.device)
    if args.stage == "smoke":
        decision = audit_smoke(protocol)
        print(json.dumps({"stage": "smoke", "passed": decision["passed"]}))
    elif args.stage == "stage_a":
        decision = summarize_stage_a(protocol)
        print(json.dumps({
            "stage": "stage_a",
            "passing_candidates": decision["passing_candidates"],
            "selected_candidate": decision["selected_candidate"],
        }))
    else:
        decision = summarize_stage_b(protocol)
        print(json.dumps({"stage": "stage_b", "arm_pass": decision["arm_pass"]}))


if __name__ == "__main__":
    main()

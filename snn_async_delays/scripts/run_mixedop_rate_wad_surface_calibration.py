"""Run the gated K=5 one-hot rate-coding WAD calibration."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import scripts.run_mixedop_spatial_temporal_surface_preview as core


BASE = core.BASE
PROTOCOL = "mixedop_rate_wad_surface_calibration_v1"
CONFIG_PATH = BASE / "configs" / f"{PROTOCOL}.yaml"


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL:
        raise ValueError("protocol id mismatch")
    if protocol["encoding"]["mode"] != "binary_one_hot_rate":
        raise ValueError("this runner requires the frozen 4K one-hot rate code")
    return protocol


def stage_protocol(protocol: dict[str, Any], stage: str) -> dict[str, Any]:
    value = deepcopy(protocol)
    if stage == "full":
        value["execution"]["formal_root"] = value["execution"]["full_root"]
    return value


def specs(protocol: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    seed = int(protocol["surface"]["seed"])
    if stage == "smoke":
        smoke = protocol["smoke"]
        return [{
            "K": 5, "condition": condition,
            "total_hidden": int(smoke["total_hidden_neurons"]),
            "output_window_len": int(smoke["output_window_length"]),
            "seed": seed, "updates": int(smoke["optimizer_updates"]),
            "stage": "smoke", "point_label": "center_smoke",
        } for condition in smoke["conditions"]]
    if stage == "pilot":
        return [{
            "K": 5, "condition": condition,
            "total_hidden": int(point["total_hidden"]),
            "output_window_len": int(point["output_window_len"]),
            "seed": seed, "updates": int(protocol["pilot"]["optimizer_updates"]),
            "stage": "pilot", "point_label": point["label"],
        } for point, condition in product(
            protocol["pilot"]["points"], protocol["pilot"]["conditions"]
        )]
    if stage == "full":
        surface = protocol["surface"]
        return [{
            "K": 5, "condition": "shared_temporal_wad",
            "total_hidden": int(hidden), "output_window_len": int(window),
            "seed": seed, "updates": int(protocol["optimization"]["optimizer_updates"]),
            "stage": "full", "point_label": f"N{hidden}_w{window}",
        } for hidden, window in product(
            surface["total_hidden_neurons"], surface["output_window_lengths"]
        )]
    raise ValueError(stage)


def _run_dir(protocol: dict[str, Any], spec: dict[str, Any]) -> Path:
    return core._run_directory(protocol, core.build_config(protocol, spec))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _reproducible_validation(run_dir: Path, device: str) -> bool:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model = core.build_model(cfg).to(device)
    model.load_state_dict(torch.load(
        run_dir / "best_model.pt", map_location=device, weights_only=True
    ))
    _, validation = core.loaders(cfg)
    encoder = core.encode_fn(cfg)
    _, first = core.validation_snapshot(
        model, validation, cfg, device, encoder, collect=True
    )
    _, second = core.validation_snapshot(
        model, validation, cfg, device, encoder, collect=True
    )
    if first is None or second is None:
        return False
    return all(np.array_equal(first[key], second[key]) for key in (
        "input_events_per_query", "logits", "predictions", "hidden_window_spikes"
    ))


def audit_smoke(protocol: dict[str, Any], device: str) -> dict[str, Any]:
    rows = []
    for spec in specs(protocol, "smoke"):
        run_dir = _run_dir(protocol, spec)
        complete = core._complete(run_dir, protocol)
        updates = _read_csv(run_dir / "update_log.csv") if complete else []
        result = json.loads((run_dir / "validation_results.json").read_text(encoding="utf-8")) if complete else {}
        condition = spec["condition"]
        per_query_nonzero = True
        if condition == "shared_temporal_wad":
            per_query_nonzero = all(
                any(float(row.get(f"delay_grad_norm_q{q}", 0.0)) > 0 for row in updates)
                for q in range(5)
            )
        finite = bool(updates) and all(
            row["loss_finite"] == "True"
            and row["gradients_finite"] == "True"
            and row["parameters_finite"] == "True"
            and row["delays_finite"] == "True"
            for row in updates
        )
        last20 = updates[-20:]
        row = {
            "condition": condition,
            "artifacts_complete": complete,
            "finite": finite,
            "nonzero_hidden_activity": bool(result.get("mean_hidden_spikes", 0) > 0),
            "oracle_schedule_exact": bool(
                condition != "shared_temporal_oracle"
                or result.get("oracle_schedule_exact") is True
            ),
            "wad_total_delay_gradient_nonzero": bool(
                condition != "shared_temporal_wad"
                or any(float(item["delay_grad_norm"]) > 0 for item in updates)
            ),
            "wad_all_query_delay_gradients_nonzero": per_query_nonzero,
            "delays_legal": bool(updates) and all(item["delays_legal"] == "True" for item in updates),
            "final_20_not_continuously_clipped": len(last20) == 20 and not all(
                item["clip_applied"] == "True" for item in last20
            ),
            "fixed_validation_realization_reproducible": (
                _reproducible_validation(run_dir, device) if complete else False
            ),
        }
        row["passed"] = all(value for key, value in row.items() if key != "condition")
        rows.append(row)
    decision = {
        "protocol_id": PROTOCOL, "stage": "smoke", "cells": len(rows),
        "invalid_for_claims": True,
        "passed": len(rows) == 2 and all(row["passed"] for row in rows),
        "rows": rows,
    }
    output = BASE / protocol["execution"]["generated_root"]
    core._write_json(output / "smoke_decision.json", decision)
    core._write_csv(output / "smoke_cells.csv", rows)
    return decision


def _spearman_with_query(values: list[float]) -> float:
    values_a = np.asarray(values, dtype=float)
    ranks = np.argsort(np.argsort(values_a)).astype(float)
    query = np.arange(len(values_a), dtype=float)
    if np.std(ranks) == 0:
        return 0.0
    return float(np.corrcoef(query, ranks)[0, 1])


def _pilot_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs(protocol, "pilot"):
        run_dir = _run_dir(protocol, spec)
        if not core._complete(run_dir, protocol):
            raise RuntimeError(f"pilot cell incomplete: {run_dir}")
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        result = json.loads((run_dir / "validation_results.json").read_text(encoding="utf-8"))
        updates = _read_csv(run_dir / "update_log.csv")
        row: dict[str, Any] = {
            "point_label": cfg["point_label"], "condition": cfg["condition"],
            "N": cfg["surface_total_hidden"], "w": cfg["output_window_len"],
            "T": cfg["T"],
            "worst_query_balanced_accuracy": result["worst_query_balanced_accuracy"],
            "mean_balanced_accuracy": result["balanced_accuracy"],
            "exact_trial_accuracy": result["exact_trial_accuracy"],
            "pooled_accuracy": result["pooled_accuracy"],
            "per_query_balanced_accuracy": result["per_query_balanced_accuracy"],
            "window_activity_fraction": result["per_output_window_hidden_activity_fraction"],
            "delay_query_means": result["delay_query_mean_steps"],
            "delay_schedule_mae_steps": result["delay_query_schedule_mae_steps"],
            "mean_input_events_per_query": result["mean_input_events_per_query"],
            "median_total_delay_grad_norm": median(float(item["delay_grad_norm"]) for item in updates),
            "run_dir": str(run_dir.relative_to(BASE)),
        }
        for q in range(5):
            values = [float(item.get(f"delay_grad_norm_q{q}", 0.0)) for item in updates]
            row[f"q{q}_delay_grad_nonzero_fraction"] = float(np.mean(np.asarray(values) > 1e-12))
            row[f"q{q}_median_delay_grad_norm"] = median(values)
        row["query_delay_spearman"] = _spearman_with_query(result["delay_query_mean_steps"])
        row["query_delay_spread_in_windows"] = (
            (max(result["delay_query_mean_steps"]) - min(result["delay_query_mean_steps"]))
            / cfg["output_window_len"]
        )
        rows.append(row)
    return rows


def _flatten(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        key: json.dumps(value) if isinstance(value, list) else value
        for key, value in row.items()
    } for row in rows]


def _plot_pilot(rows: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for ax, condition in zip(axes, ("shared_temporal_oracle", "shared_temporal_wad")):
        subset = [row for row in rows if row["condition"] == condition]
        image = ax.scatter(
            [row["T"] for row in subset], [row["N"] for row in subset],
            c=[row["worst_query_balanced_accuracy"] for row in subset],
            cmap="viridis", vmin=.5, vmax=1.0, s=320, edgecolor="black",
        )
        for row in subset:
            ax.text(row["T"], row["N"], f"{row['worst_query_balanced_accuracy']:.2f}", ha="center", va="center", color="white" if row["worst_query_balanced_accuracy"] < .75 else "black", fontweight="bold")
        ax.set_title("Fixed oracle" if "oracle" in condition else "WAD")
        ax.set_xlabel("Total latency T (steps)"); ax.grid(alpha=.2)
        ax.margins(x=.06, y=.08)
    axes[0].set_ylabel("Total hidden neurons")
    fig.colorbar(image, ax=axes, label="Worst-query balanced accuracy")
    fig.suptitle("K=5 one-hot rate pilot: four corners + center")
    fig.savefig(output / "pilot_accuracy_points.png", dpi=180); plt.close(fig)

    wad = sorted([row for row in rows if row["condition"] == "shared_temporal_wad"], key=lambda row: row["point_label"])
    gradients = np.asarray([[row[f"q{q}_median_delay_grad_norm"] for q in range(5)] for row in wad])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(np.log10(np.maximum(gradients, 1e-16)), aspect="auto", cmap="magma")
    ax.set_xticks(range(5), [f"Q{q}" for q in range(5)])
    ax.set_yticks(range(len(wad)), [row["point_label"] for row in wad])
    fig.colorbar(image, ax=ax, label="log10 median delay-gradient norm")
    ax.set_title("Per-query WAD gradient health")
    fig.savefig(output / "pilot_gradient_health.png", dpi=180); plt.close(fig)

    activity = np.asarray([row["window_activity_fraction"] for row in wad])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(activity, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(5), [f"W{q}" for q in range(5)])
    ax.set_yticks(range(len(wad)), [row["point_label"] for row in wad])
    fig.colorbar(image, ax=ax, label="Trials with hidden activity")
    ax.set_title("WAD output-window coverage")
    fig.subplots_adjust(left=.22, right=.88, bottom=.14, top=.88)
    fig.savefig(output / "pilot_window_coverage.png", dpi=180); plt.close(fig)


def summarize_pilot(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = _pilot_rows(protocol)
    wad = [row for row in rows if row["condition"] == "shared_temporal_wad"]
    oracle = [row for row in rows if row["condition"] == "shared_temporal_oracle"]
    center = next(row for row in wad if row["point_label"] == "center")
    rules = protocol["pilot_decision"]
    technical = all(
        row["median_total_delay_grad_norm"] >= float(rules["technical_gate"]["median_total_wad_delay_gradient_minimum"])
        and all(row[f"q{q}_delay_grad_nonzero_fraction"] >= float(rules["technical_gate"]["all_wad_query_gradient_nonzero_fraction_minimum"]) for q in range(5))
        for row in wad
    )
    feasibility = min(row["worst_query_balanced_accuracy"] for row in oracle) >= float(
        rules["feasibility_gate"]["oracle_worst_query_balanced_minimum_all_points"]
    )
    performance = (
        center["worst_query_balanced_accuracy"] >= float(rules["performance_sweep_gate"]["center_wad_worst_query_balanced_minimum"])
        and sum(row["worst_query_balanced_accuracy"] > .55 for row in wad)
        >= int(rules["performance_sweep_gate"]["wad_points_above_0_55_minimum"])
    )
    routing = (
        min(center["window_activity_fraction"][2:]) >= float(rules["routing_mechanism_gate"]["center_minimum_activity_fraction_in_windows_2_to_4"])
        and center["query_delay_spearman"] >= float(rules["routing_mechanism_gate"]["center_query_mean_delay_spearman_minimum"])
        and center["query_delay_spread_in_windows"] >= float(rules["routing_mechanism_gate"]["center_query_mean_delay_spread_in_windows_minimum"])
    )
    output = BASE / protocol["execution"]["generated_root"]
    _plot_pilot(rows, output)
    core._write_csv(output / "pilot_cells.csv", _flatten(rows))
    decision = {
        "protocol_id": PROTOCOL, "stage": "pilot", "cells": len(rows),
        "technical_gate_pass": technical,
        "oracle_feasibility_gate_pass": feasibility,
        "performance_sweep_gate_pass": performance,
        "routing_mechanism_gate_pass": routing,
        "full_sweep_authorized_by_results": bool(technical and feasibility and performance),
        "routing_claim_authorized": bool(technical and feasibility and performance and routing),
        "center_wad": center,
        "test_split_opened": False,
    }
    core._write_json(output / "pilot_decision.json", decision)
    return decision


def _preconditions(protocol: dict[str, Any], stage: str) -> None:
    auth = protocol["authorization"]
    if stage == "smoke" and auth.get("smoke_launch") is not True:
        raise SystemExit("smoke is locked")
    if stage == "pilot":
        if auth.get("pilot_launch") is not True:
            raise SystemExit("pilot is locked")
        decision = json.loads((BASE / protocol["execution"]["generated_root"] / "smoke_decision.json").read_text(encoding="utf-8"))
        if decision.get("passed") is not True:
            raise SystemExit("pilot requires a passing smoke decision")
    if stage == "full":
        if auth.get("full_sweep_launch") is not True:
            raise SystemExit("full surface is locked")
        decision = json.loads((BASE / protocol["execution"]["generated_root"] / "pilot_decision.json").read_text(encoding="utf-8"))
        if decision.get("full_sweep_authorized_by_results") is not True:
            raise SystemExit("pilot did not authorize the full surface")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = stage_protocol(load_protocol(), args.stage)
    cells = specs(protocol, args.stage)
    expected = {"smoke": 2, "pilot": 10, "full": 30}[args.stage]
    if len(cells) != expected:
        raise RuntimeError(f"expected {expected} cells, found {len(cells)}")
    if args.dry_run:
        print(json.dumps({
            "protocol": PROTOCOL, "stage": args.stage, "cells": len(cells),
            "paths": [str(_run_dir(protocol, spec).relative_to(BASE)) for spec in cells],
        }, indent=2))
        return
    _preconditions(protocol, args.stage)
    for spec in cells:
        core.run_cell(protocol, spec, args.device)
    if args.stage == "smoke":
        decision = audit_smoke(protocol, args.device)
        print(json.dumps({"protocol": PROTOCOL, "stage": "smoke", "passed": decision["passed"]}))
    elif args.stage == "pilot":
        decision = summarize_pilot(protocol)
        print(json.dumps({
            "protocol": PROTOCOL, "stage": "pilot",
            "full_sweep_authorized_by_results": decision["full_sweep_authorized_by_results"],
            "routing_claim_authorized": decision["routing_claim_authorized"],
        }))
    else:
        print(json.dumps({"protocol": PROTOCOL, "stage": "full", "cells_complete": 30}))


if __name__ == "__main__":
    main()

"""Run the preregistered K=6 explicitly centroid-supervised surface."""

from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

import scripts.run_mixedop_spatial_temporal_surface_preview as core


BASE = core.BASE
PROTOCOL_ID = "mixedop_k6_centroid_supervised_surface_v1"
CONFIG_PATH = BASE / "configs" / f"{PROTOCOL_ID}.yaml"
OPS = ("AND", "OR", "XOR", "XNOR", "NAND", "NOR")


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("protocol id mismatch")
    if tuple(protocol["workloads"][6]) != OPS:
        raise ValueError("frozen operation order changed")
    expected_t = [10 + 6 * w for w in protocol["surface"]["output_window_lengths"]]
    if expected_t != protocol["surface"]["total_latency_steps"]:
        raise ValueError("frozen w/T mapping is inconsistent")
    return protocol


def target_delays(window: int) -> list[float]:
    return [1.5 + 0.5 * window + q * window for q in range(6)]


def initial_delay(window: int) -> float:
    return 1.5 + 0.5 * window


def delay_support_max(window: int) -> int:
    return 6 * window + 2


def _spec(hidden: int, window: int, seed: int, stage: str, updates: int) -> dict[str, Any]:
    return {
        "K": 6,
        "condition": "shared_temporal_wad",
        "surface_condition": "centroid_supervised_shared_temporal",
        "training_arm": "arrival_centroid_huber_decoupled",
        "path_variant": "arrival_centroid_huber_decoupled",
        "total_hidden": int(hidden),
        "output_window_len": int(window),
        "point_label": f"N{hidden}_w{window}",
        "seed": int(seed),
        "updates": int(updates),
        "stage": "smoke" if stage == "preflight" else "surface",
        "protocol_stage_label": stage,
        "event_budget_label": "event8",
        "r_on_hz": 990.0,
        "r_off_hz": 10.0,
        "routing_loss_kind": "arrival_centroid_huber",
        "routing_loss_weight": 1.0,
        "delay_credit_mode": "routing_only_for_delays",
        "save_schedule_checkpoint": True,
        "schedule_gate_max_error_steps": 0.5,
        "event8_aligned": True,
        "primary_surface_metric": "worst_query_balanced_accuracy",
    }


def specs(protocol: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "preflight":
        block = protocol["preflight"]
        return [
            _spec(n, w, block["seed"], stage, block["optimizer_updates"])
            for n, w in product(
                block["total_hidden_neurons"], block["output_window_lengths"]
            )
        ]
    if stage == "surface":
        surface = protocol["surface"]
        return [
            _spec(n, w, surface["seed"], stage, protocol["optimization"]["optimizer_updates"])
            for n, w in product(
                surface["total_hidden_neurons"], surface["output_window_lengths"]
            )
        ]
    raise ValueError(f"unknown stage: {stage}")


def _core_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(protocol)
    value["execution"]["smoke_root"] = value["execution"]["preflight_root"]
    return value


def build_config(protocol: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    cfg = core.build_config(_core_protocol(protocol), spec)
    window = int(spec["output_window_len"])
    dmax = delay_support_max(window)
    init = initial_delay(window)
    fraction = init / dmax
    cfg.update({
        "protocol_stage": spec["protocol_stage_label"],
        "smoke": spec["protocol_stage_label"] == "preflight",
        "d_max": dmax,
        "delay_support": [0.0, float(dmax)],
        "delay_init_value_steps": init,
        "delay_init_raw": math.log(fraction / (1.0 - fraction)),
        "declared_delay_targets_steps": target_delays(window),
        "event_centroid_before_delay_steps": 8.5,
        "checkpoint_requires_schedule_gate": True,
        "schedule_gate_max_error_steps": 0.5,
        "routing_loss_kind": "arrival_centroid_huber",
        "routing_loss_weight": 1.0,
        "delay_credit_mode": "routing_only_for_delays",
        "optimization_schedule": "task_BCE_for_weights_readout__centroid_only_for_delays",
        "method_label": "explicitly_supervised_temporal_routing",
        "withdrawal": False,
        "oracle_delay_schedule": None,
    })
    return cfg


def run_dir(protocol: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = build_config(protocol, spec)
    return core._run_directory(_core_protocol(protocol), cfg)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _augment_cell(protocol: dict[str, Any], spec: dict[str, Any], directory: Path) -> None:
    result_path = directory / "validation_results.json"
    result = _read_json(result_path)
    cfg = build_config(protocol, spec)
    errors = [float(x) for x in result["arrival_centroid_error_per_query_steps"]]
    activity = [
        float(x) for x in result.get(
            "per_output_window_hidden_activity_fraction",
            result.get("per_query_hidden_window_activity_fraction", []),
        )
    ]
    mechanism_valid = bool(
        result.get("mechanism_valid_checkpoint_found", False)
        and max(errors) <= 0.5 + 1e-9
    )
    activity_valid = len(activity) == 6 and min(activity) >= 0.10
    pass_90 = bool(
        mechanism_valid
        and activity_valid
        and float(result["worst_query_balanced_accuracy"]) >= 0.90
    )
    targets = target_delays(cfg["output_window_len"])
    packet_containment = [
        bool(
            cfg["win_len"] + q * cfg["output_window_len"]
            <= 7.0 + d
            and 10.0 + d
            <= cfg["win_len"] + (q + 1) * cfg["output_window_len"]
        )
        for q, d in enumerate(result["delay_query_mean_steps"])
    ]
    result.update({
        "method_label": "explicitly_supervised_temporal_routing",
        "delay_credit": "arrival_centroid_huber_only",
        "weight_readout_credit": "macro_balanced_task_BCE_only",
        "declared_delay_targets_steps": targets,
        "packet_window_containment": packet_containment,
        "mechanism_valid": mechanism_valid,
        "activity_gate_passed": activity_valid,
        "mechanism_valid_90_pass": pass_90,
        "claim_status": "invalid_preflight" if cfg["smoke"] else "exploratory_single_seed",
    })
    core._write_json(result_path, result)
    prediction_path = directory / "validation_predictions.npz"
    with np.load(prediction_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays.update({
        "declared_delay_targets_steps": np.asarray(targets, dtype=np.float32),
        "arrival_centroid_steps": np.asarray(result["arrival_centroid_steps"], dtype=np.float32),
        "arrival_centroid_error_steps": np.asarray(errors, dtype=np.float32),
        "arrival_centroid_window_containment": np.asarray(
            result["arrival_centroid_window_containment"], dtype=np.bool_
        ),
        "packet_window_containment": np.asarray(packet_containment, dtype=np.bool_),
    })
    np.savez_compressed(prediction_path, **arrays)


def run_cell(
    protocol: dict[str, Any], spec: dict[str, Any], device: str, dry_run: bool = False
) -> Path:
    cfg = build_config(protocol, spec)
    core_protocol = _core_protocol(protocol)
    if dry_run:
        return core._run_directory(core_protocol, cfg)
    directory = core.run_cell(
        core_protocol, spec, device,
        config_builder=lambda _protocol, _spec: build_config(protocol, _spec),
    )
    _augment_cell(protocol, spec, directory)
    return directory


def _preflight_decision_path(protocol: dict[str, Any]) -> Path:
    return BASE / protocol["execution"]["generated_root"] / "preflight_decision.json"


def summarize_preflight(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = []
    required = protocol["required_cell_artifacts"]
    for spec in specs(protocol, "preflight"):
        directory = run_dir(protocol, spec)
        complete = all((directory / name).exists() for name in required)
        updates = _read_csv(directory / "update_log.csv") if (directory / "update_log.csv").exists() else []
        validations = _read_csv(directory / "validation_log.csv") if (directory / "validation_log.csv").exists() else []
        final = validations[-1] if validations else {}
        finite = bool(updates) and all(
            row.get("loss_finite") == "True"
            and row.get("gradients_finite") == "True"
            and row.get("parameters_finite") == "True"
            and row.get("delays_finite") == "True"
            for row in updates
        )
        legal = bool(updates) and all(row.get("delays_legal") == "True" for row in updates)
        first = updates[0] if updates else {}
        q0_finite = bool(first) and math.isfinite(float(first.get("delay_grad_mean_q0", "nan")))
        q1_q5_target_directed = bool(first) and all(
            float(first.get(f"delay_grad_mean_q{q}", 0.0)) < 0.0 for q in range(1, 6)
        )
        final_error = float(final.get("delay_query_schedule_max_abs_error_steps", "inf"))
        passed = bool(
            complete and finite and legal and q0_finite and q1_q5_target_directed
            and final_error <= 0.5
        )
        rows.append({
            "N_hidden": spec["total_hidden"], "w": spec["output_window_len"],
            "T": 10 + 6 * spec["output_window_len"], "artifacts_complete": complete,
            "finite": finite, "delays_legal": legal,
            "q0_gradient_finite_not_required_nonzero": q0_finite,
            "q1_q5_initial_gradients_nonzero_target_directed": q1_q5_target_directed,
            "final_max_centroid_error_steps": final_error,
            "accuracy_used_as_gate": False, "hidden_activity_used_as_gate": False,
            "passed": passed,
        })
    passed = len(rows) == 4 and all(row["passed"] for row in rows)
    decision = {
        "protocol_id": PROTOCOL_ID, "stage": "preflight", "invalid_for_claims": True,
        "cells_expected": 4, "cells_audited": len(rows), "rows": rows,
        "passed": passed, "formal_surface_authorized_by_results": passed,
        "formal_surface_launch_still_requires_yaml_unlock": not bool(
            protocol["authorization"]["formal_surface_launch"]
        ),
        "gradient_gate_reconciliation": protocol["preflight"]["gradient_gate_reconciliation"],
    }
    output = _preflight_decision_path(protocol).parent
    output.mkdir(parents=True, exist_ok=True)
    core._write_json(output / "preflight_decision.json", decision)
    core._write_csv(output / "preflight_cells.csv", rows)
    return decision


def n90_boundary(rows: list[dict[str, Any]], hidden_values: list[int], t_values: list[int]) -> list[dict[str, Any]]:
    boundary = []
    maximum = max(hidden_values)
    for T in t_values:
        passed = sorted(int(row["N_hidden"]) for row in rows if row["T"] == T and row["pass_90"])
        if not passed:
            boundary.append({"T": T, "N90": None, "display": f">{maximum}", "censoring": "right"})
        elif passed[0] == min(hidden_values):
            boundary.append({"T": T, "N90": passed[0], "display": f"≤{passed[0]}", "censoring": "left"})
        else:
            boundary.append({"T": T, "N90": passed[0], "display": str(passed[0]), "censoring": "none"})
    return boundary


def select_mechanism_checkpoint_rows(
    rows: list[dict[str, Any]], maximum_error: float = 0.5
) -> dict[str, Any] | None:
    """Apply the frozen eligibility/metric/tie ordering to validation rows."""
    eligible = [
        row for row in rows
        if float(row["delay_query_schedule_max_abs_error_steps"]) <= maximum_error
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row["worst_query_balanced_accuracy"]),
            float(row["exact_trial_accuracy"]),
            -int(row["update"]),
        ),
    )


def _edges(values: list[int]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mid = (values[:-1] + values[1:]) / 2
    return np.r_[values[0] - (mid[0] - values[0]), mid, values[-1] + (values[-1] - mid[-1])]


def _plane(
    rows: list[dict[str, Any]], hidden: list[int], times: list[int], key: str,
    output: Path, title: str, *, invalid_hatch: bool = True, contour90: bool = False,
) -> None:
    matrix = np.full((len(hidden), len(times)), np.nan)
    valid = np.zeros_like(matrix, dtype=bool)
    lookup = {(n, t): (i, j) for i, n in enumerate(hidden) for j, t in enumerate(times)}
    for row in rows:
        i, j = lookup[(row["N_hidden"], row["T"])]
        value = row.get(key)
        matrix[i, j] = np.nan if value is None else float(value)
        valid[i, j] = bool(row["mechanism_valid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    is_binary = key == "pass_90"
    mesh = ax.pcolormesh(
        _edges(times), _edges(hidden), matrix, shading="flat", cmap="viridis",
        vmin=0.0 if is_binary else None, vmax=1.0 if is_binary else None,
    )
    fig.colorbar(mesh, ax=ax, label=key)
    X, Y = np.meshgrid(times, hidden)
    if contour90 and np.nanmin(matrix) <= .9 <= np.nanmax(matrix):
        ax.contour(X, Y, matrix, levels=[.9], colors="white", linewidths=2)
    for i, n in enumerate(hidden):
        for j, t in enumerate(times):
            text = "--" if np.isnan(matrix[i, j]) else f"{matrix[i, j]:.2f}"
            midpoint = float(np.nanmin(matrix) + np.nanmax(matrix)) / 2.0
            text_color = "white" if not np.isnan(matrix[i, j]) and matrix[i, j] < midpoint else "black"
            ax.text(t, n, text, ha="center", va="center", fontsize=7, color=text_color)
            if invalid_hatch and not valid[i, j]:
                ax.scatter(t, n, marker="x", color="red", s=35)
    costs = [100, 300, 1000, 3000, 10000]
    dense_t = np.linspace(min(times), max(times), 300)
    for cost in costs:
        curve = cost / dense_t
        mask = (curve >= min(hidden)) & (curve <= max(hidden))
        if mask.any():
            ax.plot(dense_t[mask], curve[mask], ":", color="white", alpha=.55)
    ax.set(
        xlabel="Simulation duration T (steps)",
        ylabel="Total shared hidden neurons N_hid (log scale)", title=title,
    )
    ax.set_yscale("log")
    ax.set_ylim(_edges(hidden)[0], _edges(hidden)[-1])
    ax.set_xticks(times); ax.set_yticks(hidden); ax.set_yticklabels([str(v) for v in hidden])
    ax.text(.01, .01, "dotted: N_hid*T iso-proxy (not energy); red x: mechanism invalid",
            transform=ax.transAxes, fontsize=8, color="black",
            bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none"})
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def _fit_boundary(boundary: list[dict[str, Any]]) -> dict[str, Any]:
    points = [(float(row["T"]), float(row["N90"])) for row in boundary if row["censoring"] == "none"]
    if len(points) < 4:
        return {"performed": False, "reason": "fewer_than_four_nontruncated_columns", "points": points}
    T = np.asarray([p[0] for p in points]); N = np.asarray([p[1] for p in points])
    best = None
    for T0 in np.linspace(-2 * max(T), min(T) - 1e-3, 2000):
        design = np.c_[np.ones_like(T), 1.0 / (T - T0)]
        coef, *_ = np.linalg.lstsq(design, N, rcond=None)
        residual = N - design @ coef
        score = float(np.sum(residual ** 2))
        if best is None or score < best[0]:
            best = (score, T0, coef, residual)
    assert best is not None
    product_values = T * N
    total_variation = float(np.sum((N - N.mean()) ** 2))
    r_squared = (
        float(1.0 - best[0] / total_variation)
        if total_variation > 0 else None
    )
    return {
        "performed": True, "N_infinity": float(best[2][0]), "C": float(best[2][1]),
        "T0": float(best[1]), "sum_squared_residual": float(best[0]),
        "residuals": best[3].tolist(), "raw_points": points,
        "r_squared": r_squared,
        "C_has_expected_positive_sign_for_time_width_tradeoff": bool(best[2][1] > 0),
        "N90_times_T_coefficient_of_variation": float(np.std(product_values) / np.mean(product_values)),
    }


def _factor_diagnostics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    hidden = sorted({int(row["N_hidden"]) for row in rows})
    times = sorted({int(row["T"]) for row in rows})
    values = np.asarray([
        [next(float(row[key]) for row in rows if row["N_hidden"] == n and row["T"] == t)
         for t in times]
        for n in hidden
    ])
    grand = float(values.mean())
    total = float(np.sum((values - grand) ** 2))
    hidden_ss = float(len(times) * np.sum((values.mean(axis=1) - grand) ** 2))
    time_ss = float(len(hidden) * np.sum((values.mean(axis=0) - grand) ** 2))
    interaction = float(total - hidden_ss - time_ss)
    return {
        "metric": key,
        "hidden_marginal_range": float(np.ptp(values.mean(axis=1))),
        "T_marginal_range": float(np.ptp(values.mean(axis=0))),
        "total_grid_SS": total,
        "hidden_SS_fraction": hidden_ss / total if total else None,
        "T_SS_fraction": time_ss / total if total else None,
        "interaction_residual_fraction": interaction / total if total else None,
        "hidden_marginal_means": dict(zip(map(str, hidden), map(float, values.mean(axis=1)))),
        "T_marginal_means": dict(zip(map(str, times), map(float, values.mean(axis=0)))),
        "single_seed_descriptive_only": True,
    }


def _surface_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs(protocol, "surface"):
        directory = run_dir(protocol, spec)
        result = _read_json(directory / "validation_results.json")
        ledger = result["resource_ledger"]
        activity = result.get("per_output_window_hidden_activity_fraction", result.get("per_query_hidden_window_activity_fraction"))
        row = {
            "N_hidden": spec["total_hidden"], "w": spec["output_window_len"],
            "T": 10 + 6 * spec["output_window_len"],
            "worst_bacc": result["worst_query_balanced_accuracy"],
            "exact_trial": result["exact_trial_accuracy"],
            "max_centroid_error": result["arrival_centroid_max_abs_error_steps"],
            "mechanism_valid": result["mechanism_valid"], "pass_90": result["mechanism_valid_90_pass"],
            "N_times_T": spec["total_hidden"] * (10 + 6 * spec["output_window_len"]),
            "per_query_bacc": result["per_query_balanced_accuracy"],
            "window_activity": activity,
            "delay_means": result["delay_query_mean_steps"],
            "arrival_centroids": result["arrival_centroid_steps"],
        }
        for key in (
            "neuron_updates_per_trial", "dense_synapse_macs_per_trial",
            "mean_synaptic_events_total", "delay_buffer_elements_per_sample",
            "model_scalar_storage_elements",
        ):
            row[key] = ledger.get(key)
        rows.append(row)
    return rows


def summarize_surface(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = _surface_rows(protocol)
    hidden = [int(x) for x in protocol["surface"]["total_hidden_neurons"]]
    times = [int(x) for x in protocol["surface"]["total_latency_steps"]]
    output = BASE / protocol["execution"]["generated_root"] / "surface"
    output.mkdir(parents=True, exist_ok=True)
    _plane(rows, hidden, times, "worst_bacc", output / "K6_T_vs_nhid_worst_bacc.png",
           "K=6 explicitly supervised routing: worst-query balanced accuracy", contour90=True)
    _plane(rows, hidden, times, "pass_90", output / "K6_T_vs_nhid_mechanism_valid_90_boundary.png",
           "K=6 joint mechanism-valid 90% pass")
    _plane(rows, hidden, times, "exact_trial", output / "K6_T_vs_nhid_exact_trial.png", "Exact-trial accuracy")
    _plane(rows, hidden, times, "max_centroid_error", output / "K6_T_vs_nhid_centroid_error.png", "Maximum centroid error (steps)")
    expanded = []
    for row in rows:
        flat = {k: v for k, v in row.items() if not isinstance(v, list)}
        for q in range(6):
            flat[f"bacc_q{q}"] = row["per_query_bacc"][q]
            flat[f"activity_q{q}"] = row["window_activity"][q]
            flat[f"delay_q{q}"] = row["delay_means"][q]
            flat[f"arrival_q{q}"] = row["arrival_centroids"][q]
        expanded.append(flat)
        for q, op in enumerate(OPS):
            row[f"bacc_q{q}"] = row["per_query_bacc"][q]
            row[f"activity_q{q}"] = row["window_activity"][q]
    for q, op in enumerate(OPS):
        _plane(rows, hidden, times, f"bacc_q{q}", output / f"K6_query{q}_{op}_bacc.png", f"Query {q} ({op}) balanced accuracy")
        _plane(rows, hidden, times, f"activity_q{q}", output / f"K6_window{q}_hidden_activity.png", f"Window {q} hidden activity fraction")
    for key, label in (
        ("neuron_updates_per_trial", "Neuron updates per trial"),
        ("dense_synapse_macs_per_trial", "Dense MACs per trial"),
        ("mean_synaptic_events_total", "Mean synaptic events per trial"),
        ("delay_buffer_elements_per_sample", "Delay-buffer elements per sample"),
        ("model_scalar_storage_elements", "Model scalar storage elements"),
    ):
        _plane(rows, hidden, times, key, output / f"K6_resource_{key}.png", label, invalid_hatch=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    for q, op in enumerate(OPS):
        declared = [target_delays(int(r["w"]))[q] for r in rows]
        learned = [r["delay_means"][q] for r in rows]
        ax.scatter(declared, learned, s=14, alpha=.5, label=f"q{q} {op}")
    limit = delay_support_max(max(protocol["surface"]["output_window_lengths"]))
    ax.plot([0, limit], [0, limit], "k--", label="identity")
    ax.set(xlabel="Declared delay target (steps)", ylabel="Learned query delay (steps)",
           title="Delay-target correspondence")
    ax.legend(ncol=2, fontsize=8); fig.tight_layout()
    fig.savefig(output / "K6_delay_target_correspondence.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    for q, op in enumerate(OPS):
        target_centers = [10 + (q + .5) * int(r["w"]) for r in rows]
        arrivals = [r["arrival_centroids"][q] for r in rows]
        ax.scatter(target_centers, arrivals, s=14, alpha=.5, label=f"q{q} {op}")
    ax.plot([10, 166], [10, 166], "k--", label="identity")
    ax.set(xlabel="Output-window center (steps)", ylabel="Arrival centroid (steps)",
           title="Arrival-centroid correspondence")
    ax.legend(ncol=2, fontsize=8); fig.tight_layout()
    fig.savefig(output / "K6_arrival_centroid_correspondence.png", dpi=180); plt.close(fig)
    boundary = n90_boundary(rows, hidden, times)
    fit = _fit_boundary(boundary)
    factors = {
        "worst_query_balanced_accuracy": _factor_diagnostics(rows, "worst_bacc"),
        "exact_trial_accuracy": _factor_diagnostics(rows, "exact_trial"),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    x = [r["T"] for r in boundary]; y = [np.nan if r["N90"] is None else r["N90"] for r in boundary]
    ax.plot(x, y, "o-");
    for row in boundary:
        ax.annotate(row["display"], (row["T"], 60 if row["N90"] is None else row["N90"]))
    ax.set(xlabel="Simulation duration T (steps)", ylabel="N90", title="Mechanism-valid 90% boundary")
    fig.tight_layout(); fig.savefig(output / "K6_N90_vs_T.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([r["N_times_T"] for r in rows], [r["worst_bacc"] for r in rows],
               c=[r["T"] for r in rows], cmap="viridis")
    ax.set(xlabel="N_hid*T neuron-update proxy (not energy)", ylabel="Worst-query balanced accuracy",
           title="Accuracy versus N_hid*T proxy")
    fig.tight_layout(); fig.savefig(output / "K6_accuracy_vs_NT_proxy.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (label, values) in zip(axes, factors.items()):
        fractions = [values["hidden_SS_fraction"], values["T_SS_fraction"], values["interaction_residual_fraction"]]
        ax.bar(["N_hid", "T", "interaction"], fractions)
        ax.set_ylim(0, 1); ax.set_ylabel("fraction of grid SS"); ax.set_title(label.replace("_", " "))
        for i, value in enumerate(fractions):
            ax.text(i, value + .02, f"{value:.3f}", ha="center")
    fig.suptitle("Single-seed descriptive factor decomposition (not inferential ANOVA)")
    fig.tight_layout(); fig.savefig(output / "K6_factor_SS_decomposition.png", dpi=180); plt.close(fig)
    core._write_csv(output / "surface_cells.csv", expanded)
    core._write_json(output / "N90_boundary_and_fit.json", {"boundary": boundary, "fit": fit})
    core._write_json(output / "factor_diagnostics.json", factors)
    decision = {
        "protocol_id": PROTOCOL_ID, "stage": "surface", "cells": len(rows),
        "single_seed_exploratory": True, "autonomous_WAD_claim_authorized": False,
        "spatial_temporal_pareto_claim_authorized": False, "boundary": boundary, "fit": fit,
    }
    core._write_json(output / "surface_decision.json", decision)
    return decision


def preconditions(protocol: dict[str, Any], stage: str, dry_run: bool) -> None:
    if dry_run:
        return
    if stage == "preflight" and not protocol["authorization"]["preflight_launch"]:
        raise RuntimeError("preflight launch is locked")
    if stage == "surface":
        if not protocol["authorization"]["formal_surface_launch"]:
            raise RuntimeError("formal surface remains locked in YAML")
        decision_path = _preflight_decision_path(protocol)
        if not decision_path.exists() or not _read_json(decision_path).get("passed"):
            raise RuntimeError("formal surface requires a passing preflight decision")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "surface"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    preconditions(protocol, args.stage, args.dry_run)
    cells = specs(protocol, args.stage)
    directories = [run_cell(protocol, spec, args.device, args.dry_run) for spec in cells]
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "cells": len(directories), "paths": [str(p) for p in directories]}, indent=2))
        return
    decision = summarize_preflight(protocol) if args.stage == "preflight" else summarize_surface(protocol)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

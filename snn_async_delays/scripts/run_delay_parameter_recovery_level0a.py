"""Run the preregistered scalar delay-parameter recovery Level 0A matrix.

This deliberately bypasses spike buffers, neurons, tasks, and readouts.  It
optimizes the production ``DelayedSynapticLayer.delay_raw`` against a declared
nominal delay target and writes every diagnostic during the cell execution.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from snn.synapses import DelayedSynapticLayer


BASE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "delay_parameter_recovery_level0a_v1"
CONFIG_PATH = BASE / "configs" / f"{PROTOCOL_ID}.yaml"
RUN_ROOT = BASE / "runs" / "exploratory" / PROTOCOL_ID
SUMMARY_ROOT = BASE / "docs" / "generated" / PROTOCOL_ID


def load_protocol(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"unexpected protocol id: {protocol.get('protocol_id')!r}")
    return protocol


def target_values(protocol: dict[str, Any]) -> list[float]:
    grid = protocol["grid"]
    return [
        *[float(value) for value in grid["boundary_stress_targets_steps"][:1]],
        *[float(value) for value in grid["interior_targets_steps"]],
        *[float(value) for value in grid["boundary_stress_targets_steps"][1:]],
    ]


def expected_cells(protocol: dict[str, Any]) -> int:
    grid = protocol["grid"]
    return (
        len(grid["initial_raw_values"])
        * len(target_values(protocol))
        * len(grid["learning_rates"])
    )


def _token(value: float) -> str:
    rendered = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return rendered


def cell_directory(learning_rate: float, target: float, initial_raw: float) -> Path:
    return (
        RUN_ROOT
        / f"lr_{_token(learning_rate)}"
        / f"target_{_token(target)}"
        / f"init_raw_{_token(initial_raw)}"
    )


def is_interior_target(protocol: dict[str, Any], target: float) -> bool:
    return any(
        math.isclose(float(target), float(value), abs_tol=1e-12)
        for value in protocol["grid"]["interior_targets_steps"]
    )


def optimize_scalar_delay(
    *,
    d_max: float,
    target: float,
    initial_raw: float,
    learning_rate: float,
    optimizer_steps: int,
) -> dict[str, np.ndarray | float | int | bool | None]:
    """Optimize one production sigmoid delay and return its full trajectory."""
    torch.manual_seed(0)
    layer = DelayedSynapticLayer(
        1,
        1,
        d_max=int(d_max),
        delay_param_type="sigmoid",
        delay_init_mode="constant",
        delay_init_raw=float(initial_raw),
        train_weights=False,
        train_delays=True,
    )
    optimizer = torch.optim.Adam(
        [layer.delay_raw], lr=float(learning_rate), betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.0,
    )

    steps: list[int] = []
    raw_values: list[float] = []
    delays: list[float] = []
    losses: list[float] = []
    errors: list[float] = []
    raw_gradients: list[float] = []
    sigmoid_slopes: list[float] = []

    for step in range(int(optimizer_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        delay = layer.get_delays()[0, 0]
        loss = 0.5 * (delay - float(target)).pow(2)
        loss.backward()

        raw_value = float(layer.delay_raw.detach().item())
        delay_value = float(delay.detach().item())
        raw_gradient = float(layer.delay_raw.grad.detach().item())
        slope = float(d_max) * float(torch.sigmoid(layer.delay_raw.detach()).item()) * (
            1.0 - float(torch.sigmoid(layer.delay_raw.detach()).item())
        )
        steps.append(step)
        raw_values.append(raw_value)
        delays.append(delay_value)
        losses.append(float(loss.detach().item()))
        errors.append(abs(delay_value - float(target)))
        raw_gradients.append(raw_gradient)
        sigmoid_slopes.append(slope)

        if step < int(optimizer_steps):
            optimizer.step()

    tolerance = 0.1
    convergence = next(
        (step for step, error in zip(steps, errors) if error <= tolerance), None
    )
    return {
        "steps": np.asarray(steps, dtype=np.int32),
        "raw": np.asarray(raw_values, dtype=np.float64),
        "delay": np.asarray(delays, dtype=np.float64),
        "loss": np.asarray(losses, dtype=np.float64),
        "absolute_error": np.asarray(errors, dtype=np.float64),
        "raw_gradient": np.asarray(raw_gradients, dtype=np.float64),
        "sigmoid_slope": np.asarray(sigmoid_slopes, dtype=np.float64),
        "initial_delay": delays[0],
        "final_delay": delays[-1],
        "initial_raw": raw_values[0],
        "final_raw": raw_values[-1],
        "final_loss": losses[-1],
        "final_absolute_error": errors[-1],
        "convergence_step": convergence,
        "recovered": bool(errors[-1] <= tolerance),
    }


def _save_cell_panel(
    trace: dict[str, Any], target: float, learning_rate: float, output: Path
) -> None:
    steps = trace["steps"]
    error_floor = np.maximum(trace["absolute_error"], 1e-10)
    loss_floor = np.maximum(trace["loss"], 1e-12)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Level 0A scalar delay recovery | target={target:g}, lr={learning_rate:g}"
    )

    axes[0, 0].plot(steps, trace["delay"], color="#177e89", linewidth=2)
    axes[0, 0].axhline(target, color="#bc4749", linestyle="--", label="target")
    axes[0, 0].set(xlabel="optimizer step", ylabel="nominal delay (steps)", title="Delay trajectory")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].semilogy(steps, error_floor, color="#6a4c93", linewidth=2)
    axes[0, 1].axhline(0.1, color="#bc4749", linestyle="--", label="0.1-step gate")
    axes[0, 1].set(xlabel="optimizer step", ylabel="absolute error (steps)", title="Recovery error")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].semilogy(steps, loss_floor, color="#386641", linewidth=2)
    axes[1, 0].set(xlabel="optimizer step", ylabel="half squared error", title="Direct objective")

    axes[1, 1].plot(steps, trace["raw_gradient"], color="#f4a261", linewidth=1.7, label="raw gradient")
    axes[1, 1].plot(steps, trace["sigmoid_slope"], color="#264653", linewidth=1.7, label="dd/draw")
    axes[1, 1].axhline(0.0, color="#888888", linewidth=0.8)
    axes[1, 1].set(xlabel="optimizer step", ylabel="gradient / slope", title="Optimization geometry")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=0.25, linewidth=0.7)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def run_cell(
    protocol: dict[str, Any], learning_rate: float, target: float, initial_raw: float
) -> dict[str, Any]:
    output = cell_directory(learning_rate, target, initial_raw)
    plot_dir = output / "plots"
    required = [
        output / "config.json",
        output / "metrics.json",
        output / "final_parameter.pt",
        plot_dir / "diagnostic_data.npz",
        plot_dir / "diagnostic_panel.png",
    ]
    if all(path.exists() for path in required):
        return json.loads((output / "metrics.json").read_text(encoding="utf-8"))

    output.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    d_max = float(protocol["parameterization"]["d_max_steps"])
    steps = int(protocol["grid"]["optimizer_steps"])
    cell_config = {
        "protocol_id": PROTOCOL_ID,
        "study_class": protocol["study_class"],
        "learning_rate": float(learning_rate),
        "target_delay_steps": float(target),
        "target_kind": "interior" if is_interior_target(protocol, target) else "boundary_stress",
        "initial_raw": float(initial_raw),
        "d_max_steps": d_max,
        "optimizer_steps": steps,
        "optimizer": protocol["optimization"]["optimizer"],
        "loss": protocol["optimization"]["loss"],
        "stochastic_seed": None,
        "task_data": None,
        "test_split_opened": False,
    }
    (output / "config.json").write_text(
        json.dumps(cell_config, indent=2) + "\n", encoding="utf-8"
    )

    trace = optimize_scalar_delay(
        d_max=d_max,
        target=float(target),
        initial_raw=float(initial_raw),
        learning_rate=float(learning_rate),
        optimizer_steps=steps,
    )
    np.savez_compressed(
        plot_dir / "diagnostic_data.npz",
        steps=trace["steps"], raw=trace["raw"], delay=trace["delay"],
        loss=trace["loss"], absolute_error=trace["absolute_error"],
        raw_gradient=trace["raw_gradient"], sigmoid_slope=trace["sigmoid_slope"],
        target_delay_steps=np.asarray(float(target)),
        learning_rate=np.asarray(float(learning_rate)),
        d_max_steps=np.asarray(d_max),
    )
    _save_cell_panel(trace, float(target), float(learning_rate), plot_dir / "diagnostic_panel.png")

    torch.save(
        {
            "delay_raw": torch.tensor(float(trace["final_raw"])),
            "delay_steps": torch.tensor(float(trace["final_delay"])),
        },
        output / "final_parameter.pt",
    )
    metrics = {
        **cell_config,
        "initial_delay_steps": float(trace["initial_delay"]),
        "final_delay_steps": float(trace["final_delay"]),
        "final_raw": float(trace["final_raw"]),
        "final_loss": float(trace["final_loss"]),
        "final_absolute_delay_error_steps": float(trace["final_absolute_error"]),
        "convergence_step": trace["convergence_step"],
        "recovered_within_0p1_step": bool(trace["recovered"]),
        "initial_raw_gradient": float(trace["raw_gradient"][0]),
        "final_raw_gradient": float(trace["raw_gradient"][-1]),
        "initial_sigmoid_slope": float(trace["sigmoid_slope"][0]),
        "final_sigmoid_slope": float(trace["sigmoid_slope"][-1]),
        "delay_parameter_count": 1,
        "complete": True,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metrics


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _heatmaps(
    protocol: dict[str, Any], rows: list[dict[str, Any]], field: str, output: Path,
    *, title: str, colorbar_label: str, convergence: bool = False,
) -> None:
    inits = [float(value) for value in protocol["grid"]["initial_raw_values"]]
    targets = target_values(protocol)
    learning_rates = [float(value) for value in protocol["grid"]["learning_rates"]]
    fig, axes = plt.subplots(1, len(learning_rates), figsize=(15, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(title)
    matrices = []
    for learning_rate in learning_rates:
        matrix = np.full((len(inits), len(targets)), np.nan, dtype=float)
        for row in rows:
            if not math.isclose(float(row["learning_rate"]), learning_rate):
                continue
            i = inits.index(float(row["initial_raw"]))
            j = targets.index(float(row["target_delay_steps"]))
            value = row[field]
            matrix[i, j] = float(value) if value is not None else np.nan
        matrices.append(matrix)

    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    if convergence:
        vmin, vmax = 0.0, float(protocol["grid"]["optimizer_steps"])
        cmap = "viridis_r"
    else:
        vmin, vmax = 0.0, max(0.1, float(finite.max()))
        cmap = "magma_r"

    image = None
    for axis, learning_rate, matrix in zip(axes, learning_rates, matrices):
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(f"Adam lr={learning_rate:g}")
        axis.set_xticks(range(len(targets)), [f"{value:g}" for value in targets])
        axis.set_yticks(range(len(inits)), [f"{value:g}" for value in inits])
        axis.set_xlabel("target delay (steps)")
        axis.set_ylabel("initial raw")
        for i in range(len(inits)):
            for j in range(len(targets)):
                value = matrix[i, j]
                label = "NR" if not np.isfinite(value) else (f"{int(value)}" if convergence else f"{value:.2f}")
                axis.text(j, i, label, ha="center", va="center", fontsize=8)
        for boundary in (0, len(targets) - 1):
            axis.add_patch(plt.Rectangle((boundary - 0.5, -0.5), 1, len(inits), fill=False,
                                         edgecolor="#457b9d", linewidth=2, linestyle="--"))
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82)
    colorbar.set_label(colorbar_label)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def aggregate(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        rows,
        key=lambda row: (
            float(row["learning_rate"]), float(row["target_delay_steps"]),
            float(row["initial_raw"]),
        ),
    )
    _write_csv(rows, SUMMARY_ROOT / "cells.csv")
    _heatmaps(
        protocol, rows, "final_absolute_delay_error_steps",
        SUMMARY_ROOT / "final_error_heatmaps.png",
        title="Level 0A final scalar-delay recovery error",
        colorbar_label="final absolute error (nominal steps)",
    )
    _heatmaps(
        protocol, rows, "convergence_step",
        SUMMARY_ROOT / "convergence_heatmaps.png",
        title="Level 0A steps to enter the 0.1-step recovery tolerance",
        colorbar_label="optimizer step (NR = not recovered)", convergence=True,
    )

    declared = protocol["decision_rules"]["current_recipe_cell"]
    current_row = next(
        row for row in rows
        if math.isclose(float(row["initial_raw"]), float(declared["initial_raw"]))
        and math.isclose(float(row["learning_rate"]), float(declared["learning_rate"]))
        and math.isclose(float(row["target_delay_steps"]), float(declared["target_steps"]))
    )
    interior_targets = [float(value) for value in protocol["grid"]["interior_targets_steps"]]
    initial_raw_values = [float(value) for value in protocol["grid"]["initial_raw_values"]]
    coverage: list[dict[str, Any]] = []
    for target in interior_targets:
        for initial_raw in initial_raw_values:
            candidates = [
                row for row in rows
                if math.isclose(float(row["target_delay_steps"]), target)
                and math.isclose(float(row["initial_raw"]), initial_raw)
            ]
            best = min(candidates, key=lambda row: float(row["final_absolute_delay_error_steps"]))
            coverage.append({
                "target_delay_steps": target,
                "initial_raw": initial_raw,
                "recovered_by_any_preregistered_lr": bool(
                    any(bool(row["recovered_within_0p1_step"]) for row in candidates)
                ),
                "best_learning_rate": float(best["learning_rate"]),
                "best_final_absolute_error_steps": float(best["final_absolute_delay_error_steps"]),
            })

    decision = {
        "protocol_id": PROTOCOL_ID,
        "expected_cells": expected_cells(protocol),
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == expected_cells(protocol) and all(bool(row["complete"]) for row in rows),
        "current_recipe_cell": {
            "initial_raw": float(current_row["initial_raw"]),
            "learning_rate": float(current_row["learning_rate"]),
            "target_delay_steps": float(current_row["target_delay_steps"]),
            "final_delay_steps": float(current_row["final_delay_steps"]),
            "final_absolute_error_steps": float(current_row["final_absolute_delay_error_steps"]),
            "convergence_step": current_row["convergence_step"],
            "pass": bool(current_row["recovered_within_0p1_step"]),
        },
        "interior_recoverability_cells": coverage,
        "interior_recoverability_pass": bool(
            all(item["recovered_by_any_preregistered_lr"] for item in coverage)
        ),
        "boundary_stress_is_descriptive_only": True,
        "level0b_authorized_if_interpreted_narrowly": bool(
            len(rows) == expected_cells(protocol)
        ),
        "xor_learning_rate_change_authorized": False,
        "routing_or_accuracy_claim_authorized": False,
    }
    (SUMMARY_ROOT / "decision.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return decision


def main() -> None:
    protocol = load_protocol()
    if protocol["status"] not in {"preregistered_ready", "completed"}:
        raise SystemExit(f"protocol status {protocol['status']!r} is not launchable")
    declared_cells = int(protocol["grid"]["deterministic_cells"])
    if expected_cells(protocol) != declared_cells:
        raise SystemExit(
            f"grid has {expected_cells(protocol)} cells but config declares {declared_cells}"
        )

    rows: list[dict[str, Any]] = []
    for learning_rate in [float(value) for value in protocol["grid"]["learning_rates"]]:
        for target in target_values(protocol):
            for initial_raw in [float(value) for value in protocol["grid"]["initial_raw_values"]]:
                rows.append(run_cell(protocol, learning_rate, target, initial_raw))
    decision = aggregate(protocol, rows)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

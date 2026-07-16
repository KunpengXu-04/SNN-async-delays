"""Run the preregistered Level 0B temporal-credit diagnostic.

The experiment optimizes only one production sigmoid delay.  Loss is computed
from either the delayed current trace or a one-neuron LIF spike trace; it never
reads the delay parameter directly.
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

from snn.neurons import LIFNeurons
from snn.synapses import DelayedSynapticLayer


BASE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "delay_temporal_credit_level0b_v1"
CONFIG_PATH = BASE / "configs" / f"{PROTOCOL_ID}.yaml"
RUN_ROOT = BASE / "runs" / "exploratory" / PROTOCOL_ID
SUMMARY_ROOT = BASE / "docs" / "generated" / PROTOCOL_ID


def load_protocol(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"unexpected protocol id: {protocol.get('protocol_id')!r}")
    return protocol


def expected_cells(protocol: dict[str, Any]) -> int:
    return (
        len(protocol["paths"])
        * len(protocol["loss_modes"])
        * len(protocol["timeline"]["target_nominal_delays_steps"])
        * len(protocol["delay"]["initial_raw_values"])
        * len(protocol["delay"]["learning_rates"])
    )


def target_arrival_step(protocol: dict[str, Any], nominal_delay: float) -> float:
    return float(protocol["timeline"]["input_spike_step"]) + float(nominal_delay) + 1.0


def _token(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def cell_directory(
    path: str, loss_mode: str, learning_rate: float, target: float, initial_raw: float
) -> Path:
    return (
        RUN_ROOT / path / loss_mode / f"lr_{_token(learning_rate)}"
        / f"target_{_token(target)}" / f"init_raw_{_token(initial_raw)}"
    )


def make_delay_layer(
    *, d_max: int, initial_raw: float, fixed_weight: float,
) -> DelayedSynapticLayer:
    torch.manual_seed(0)
    layer = DelayedSynapticLayer(
        1, 1, d_max=int(d_max), delay_param_type="sigmoid",
        delay_init_mode="constant", delay_init_raw=float(initial_raw),
        train_weights=False, train_delays=True,
    )
    with torch.no_grad():
        layer.weight.fill_(float(fixed_weight))
    return layer


def render_temporal_trace(
    layer: DelayedSynapticLayer,
    *,
    path: str,
    total_steps: int,
    input_spike_step: int,
    lif_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Unroll the real circular buffer and optional production LIF neuron."""
    if path not in {"buffer_current", "lif_spike"}:
        raise ValueError(path)
    buffer = torch.zeros(1, layer.d_max + 1, 1)
    pointer = 0
    delays = layer.get_delays()
    if path == "lif_spike":
        lif = LIFNeurons(
            1,
            tau_m=float(lif_config["tau_m_steps"]),
            v_threshold=float(lif_config["threshold_au"]),
            v_reset=float(lif_config["reset_au"]),
            refractory_steps=int(lif_config["refractory_steps"]),
            dt=float(lif_config["dt_steps"]),
            surrogate_beta=float(lif_config["surrogate_beta"]),
        )
        voltage, refractory = lif.init_state(1)

    currents: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    membranes: list[torch.Tensor] = []
    for time_index in range(int(total_steps)):
        current = layer(buffer, delays, pointer).reshape(())
        if path == "lif_spike":
            spike, voltage, refractory = lif(
                current.reshape(1, 1), voltage, refractory
            )
            output = spike.reshape(())
            membranes.append(voltage.reshape(()))
        else:
            output = current
            membranes.append(torch.zeros_like(current))

        currents.append(current)
        outputs.append(output)
        buffer[:, pointer, :] = 1.0 if time_index == int(input_spike_step) else 0.0
        pointer = (pointer + 1) % (layer.d_max + 1)

    return {
        "current": torch.stack(currents),
        "output": torch.stack(outputs),
        "membrane": torch.stack(membranes),
    }


def trace_centroid(trace: torch.Tensor) -> torch.Tensor:
    times = torch.arange(trace.numel(), dtype=trace.dtype, device=trace.device)
    return (times * trace).sum() / (trace.sum() + 1e-8)


def causal_filter(trace: torch.Tensor, tau_steps: float) -> torch.Tensor:
    decay = float(math.exp(-1.0 / float(tau_steps)))
    state = torch.zeros((), dtype=trace.dtype, device=trace.device)
    filtered = []
    for value in trace:
        state = decay * state + value
        filtered.append(state)
    return torch.stack(filtered)


def temporal_loss(
    output_trace: torch.Tensor,
    *,
    loss_mode: str,
    target_arrival: int,
    filter_tau_steps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_trace = torch.zeros_like(output_trace)
    target_trace[int(target_arrival)] = 1.0
    centroid = trace_centroid(output_trace)
    if loss_mode == "arrival_centroid":
        loss = 0.5 * (centroid - float(target_arrival)).pow(2)
    elif loss_mode == "filtered_trace":
        filtered_output = causal_filter(output_trace, filter_tau_steps)
        filtered_target = causal_filter(target_trace, filter_tau_steps)
        loss = (filtered_output - filtered_target).pow(2).mean()
    else:
        raise ValueError(loss_mode)
    return loss, centroid, target_trace


def optimize_temporal_credit(
    protocol: dict[str, Any],
    *,
    path: str,
    loss_mode: str,
    target_nominal_delay: float,
    initial_raw: float,
    learning_rate: float,
) -> dict[str, Any]:
    timeline = protocol["timeline"]
    delay_cfg = protocol["delay"]
    fixed_weight = (
        float(delay_cfg["fixed_weight_for_buffer_path"])
        if path == "buffer_current"
        else float(delay_cfg["fixed_weight_for_lif_path"])
    )
    layer = make_delay_layer(
        d_max=int(timeline["d_max_steps"]), initial_raw=float(initial_raw),
        fixed_weight=fixed_weight,
    )
    optimizer = torch.optim.Adam([layer.delay_raw], lr=float(learning_rate))
    optimizer_steps = int(delay_cfg["optimizer_steps"])
    target_arrival = int(target_arrival_step(protocol, target_nominal_delay))
    filter_tau = float(protocol["loss_modes"]["filtered_trace"]["filter_tau_steps"])

    steps: list[int] = []
    raw_values: list[float] = []
    delays: list[float] = []
    losses: list[float] = []
    centroids: list[float] = []
    arrival_errors: list[float] = []
    delay_errors: list[float] = []
    masses: list[float] = []
    gradients: list[float] = []
    initial_output = initial_current = initial_membrane = None
    final_output = final_current = final_membrane = target_trace_np = None
    convergence_step = None

    for step in range(optimizer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        rendered = render_temporal_trace(
            layer, path=path, total_steps=int(timeline["total_steps"]),
            input_spike_step=int(timeline["input_spike_step"]),
            lif_config=protocol["lif"],
        )
        loss, centroid, target_trace = temporal_loss(
            rendered["output"], loss_mode=loss_mode,
            target_arrival=target_arrival, filter_tau_steps=filter_tau,
        )
        loss.backward()
        delay_value = float(layer.get_delays()[0, 0].detach().item())
        mass = float(rendered["output"].detach().sum().item())
        centroid_value = float(centroid.detach().item())
        arrival_error = abs(centroid_value - float(target_arrival))

        steps.append(step)
        raw_values.append(float(layer.delay_raw.detach().item()))
        delays.append(delay_value)
        losses.append(float(loss.detach().item()))
        centroids.append(centroid_value)
        arrival_errors.append(arrival_error)
        delay_errors.append(abs(delay_value - float(target_nominal_delay)))
        masses.append(mass)
        gradients.append(float(layer.delay_raw.grad.detach().item()))
        if convergence_step is None and arrival_error <= 0.1 and mass >= 0.5:
            convergence_step = step

        if step == 0:
            initial_output = rendered["output"].detach().cpu().numpy()
            initial_current = rendered["current"].detach().cpu().numpy()
            initial_membrane = rendered["membrane"].detach().cpu().numpy()
        if step == optimizer_steps:
            final_output = rendered["output"].detach().cpu().numpy()
            final_current = rendered["current"].detach().cpu().numpy()
            final_membrane = rendered["membrane"].detach().cpu().numpy()
            target_trace_np = target_trace.detach().cpu().numpy()
        else:
            optimizer.step()

    recovered = bool(arrival_errors[-1] <= 0.1 and masses[-1] >= 0.5)
    return {
        "steps": np.asarray(steps, dtype=np.int32),
        "raw": np.asarray(raw_values, dtype=np.float64),
        "delay": np.asarray(delays, dtype=np.float64),
        "loss": np.asarray(losses, dtype=np.float64),
        "output_centroid": np.asarray(centroids, dtype=np.float64),
        "arrival_error": np.asarray(arrival_errors, dtype=np.float64),
        "delay_error": np.asarray(delay_errors, dtype=np.float64),
        "output_mass": np.asarray(masses, dtype=np.float64),
        "raw_gradient": np.asarray(gradients, dtype=np.float64),
        "initial_output_trace": initial_output,
        "final_output_trace": final_output,
        "initial_current_trace": initial_current,
        "final_current_trace": final_current,
        "initial_membrane_trace": initial_membrane,
        "final_membrane_trace": final_membrane,
        "target_trace": target_trace_np,
        "target_arrival_step": target_arrival,
        "convergence_step": convergence_step,
        "recovered": recovered,
    }


def _save_cell_panel(
    trace: dict[str, Any], *, path: str, loss_mode: str, target_delay: float,
    learning_rate: float, output: Path,
) -> None:
    steps = trace["steps"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Level 0B | {path} | {loss_mode} | target d={target_delay:g}, lr={learning_rate:g}"
    )
    axes[0, 0].plot(steps, trace["delay"], color="#177e89", linewidth=2)
    axes[0, 0].axhline(target_delay, color="#bc4749", linestyle="--", label="nominal target")
    axes[0, 0].set(xlabel="optimizer step", ylabel="delay (nominal steps)", title="Delay trajectory")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(steps, trace["output_centroid"], color="#6a4c93", linewidth=2)
    axes[0, 1].axhline(trace["target_arrival_step"], color="#bc4749", linestyle="--", label="target arrival")
    axes[0, 1].set(xlabel="optimizer step", ylabel="output centroid (step)", title="Functional arrival")
    axes[0, 1].legend(frameon=False)

    axes[0, 2].semilogy(steps, np.maximum(trace["arrival_error"], 1e-10), color="#6a4c93", linewidth=2)
    axes[0, 2].axhline(0.1, color="#bc4749", linestyle="--", label="0.1-step gate")
    axes[0, 2].set(xlabel="optimizer step", ylabel="arrival error (steps)", title="Primary error")
    axes[0, 2].legend(frameon=False)

    axes[1, 0].semilogy(steps, np.maximum(trace["loss"], 1e-12), color="#386641", linewidth=2)
    axes[1, 0].set(xlabel="optimizer step", ylabel="loss", title="Timing objective")

    axes[1, 1].plot(steps, trace["raw_gradient"], color="#f4a261", linewidth=1.8)
    axes[1, 1].axhline(0.0, color="#888888", linewidth=0.8)
    axes[1, 1].set(xlabel="optimizer step", ylabel="dL / d(raw)", title="Delay credit")

    time = np.arange(len(trace["target_trace"]))
    axes[1, 2].step(time, trace["target_trace"], where="mid", color="#bc4749", linestyle="--", label="target")
    axes[1, 2].step(time, trace["initial_output_trace"], where="mid", color="#457b9d", alpha=0.7, label="initial")
    axes[1, 2].step(time, trace["final_output_trace"], where="mid", color="#2a9d8f", linewidth=2, label="final")
    axes[1, 2].set(xlabel="simulation step", ylabel="current" if path == "buffer_current" else "spike", title="Output trace")
    axes[1, 2].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=0.25, linewidth=0.7)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def run_cell(
    protocol: dict[str, Any], *, path: str, loss_mode: str,
    learning_rate: float, target: float, initial_raw: float,
) -> dict[str, Any]:
    output = cell_directory(path, loss_mode, learning_rate, target, initial_raw)
    plot_dir = output / "plots"
    required = [
        output / "config.json", output / "metrics.json",
        output / "final_parameter.pt", plot_dir / "diagnostic_data.npz",
        plot_dir / "diagnostic_panel.png",
    ]
    if all(item.exists() for item in required):
        return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    target_arrival = target_arrival_step(protocol, target)
    cell_config = {
        "protocol_id": PROTOCOL_ID,
        "path": path,
        "loss_mode": loss_mode,
        "learning_rate": float(learning_rate),
        "target_nominal_delay_steps": float(target),
        "target_arrival_step": float(target_arrival),
        "initial_raw": float(initial_raw),
        "d_max_steps": int(protocol["timeline"]["d_max_steps"]),
        "total_steps": int(protocol["timeline"]["total_steps"]),
        "input_spike_step": int(protocol["timeline"]["input_spike_step"]),
        "optimizer_steps": int(protocol["delay"]["optimizer_steps"]),
        "direct_delay_supervision": False,
        "stochastic_seed": None,
        "test_split_opened": False,
    }
    (output / "config.json").write_text(json.dumps(cell_config, indent=2) + "\n", encoding="utf-8")

    trace = optimize_temporal_credit(
        protocol, path=path, loss_mode=loss_mode,
        target_nominal_delay=float(target), initial_raw=float(initial_raw),
        learning_rate=float(learning_rate),
    )
    np.savez_compressed(
        plot_dir / "diagnostic_data.npz",
        **{key: value for key, value in trace.items()
           if isinstance(value, np.ndarray)},
        target_arrival_step=np.asarray(trace["target_arrival_step"]),
        convergence_step=np.asarray(-1 if trace["convergence_step"] is None else trace["convergence_step"]),
    )
    _save_cell_panel(
        trace, path=path, loss_mode=loss_mode, target_delay=float(target),
        learning_rate=float(learning_rate), output=plot_dir / "diagnostic_panel.png",
    )
    torch.save(
        {"delay_raw": torch.tensor(float(trace["raw"][-1])),
         "delay_steps": torch.tensor(float(trace["delay"][-1]))},
        output / "final_parameter.pt",
    )
    metrics = {
        **cell_config,
        "initial_delay_steps": float(trace["delay"][0]),
        "final_delay_steps": float(trace["delay"][-1]),
        "final_nominal_delay_error_steps": float(trace["delay_error"][-1]),
        "initial_output_arrival_centroid": float(trace["output_centroid"][0]),
        "final_output_arrival_centroid": float(trace["output_centroid"][-1]),
        "final_output_arrival_error_steps": float(trace["arrival_error"][-1]),
        "initial_output_trace_mass": float(trace["output_mass"][0]),
        "final_output_trace_mass": float(trace["output_mass"][-1]),
        "final_output_spike_count": (
            float(trace["final_output_trace"].sum()) if path == "lif_spike" else None
        ),
        "final_loss": float(trace["loss"][-1]),
        "initial_raw_gradient": float(trace["raw_gradient"][0]),
        "final_raw_gradient": float(trace["raw_gradient"][-1]),
        "convergence_step": trace["convergence_step"],
        "recovered_within_0p1_arrival_step": bool(trace["recovered"]),
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


def _error_heatmap(
    protocol: dict[str, Any], rows: list[dict[str, Any]], *, path: str,
    loss_mode: str, output: Path,
) -> None:
    inits = [float(value) for value in protocol["delay"]["initial_raw_values"]]
    targets = [float(value) for value in protocol["timeline"]["target_nominal_delays_steps"]]
    learning_rates = [float(value) for value in protocol["delay"]["learning_rates"]]
    selected = [row for row in rows if row["path"] == path and row["loss_mode"] == loss_mode]
    matrices = []
    for learning_rate in learning_rates:
        matrix = np.full((len(inits), len(targets)), np.nan)
        for row in selected:
            if math.isclose(float(row["learning_rate"]), learning_rate):
                i = inits.index(float(row["initial_raw"]))
                j = targets.index(float(row["target_nominal_delay_steps"]))
                matrix[i, j] = float(row["final_output_arrival_error_steps"])
        matrices.append(matrix)
    vmax = max(0.1, max(float(np.nanmax(matrix)) for matrix in matrices))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Level 0B final arrival error | {path} | {loss_mode}")
    image = None
    for axis, learning_rate, matrix in zip(axes, learning_rates, matrices):
        image = axis.imshow(matrix, cmap="magma_r", vmin=0.0, vmax=vmax, aspect="auto")
        axis.set_title(f"Adam lr={learning_rate:g}")
        axis.set_xticks(range(len(targets)), [f"{value:g}" for value in targets])
        axis.set_yticks(range(len(inits)), [f"{value:g}" for value in inits])
        axis.set_xlabel("target nominal delay")
        axis.set_ylabel("initial raw")
        for i in range(len(inits)):
            for j in range(len(targets)):
                normalized = matrix[i, j] / vmax if vmax > 0 else 0.0
                axis.text(
                    j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if normalized > 0.58 else "black",
                )
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82)
    colorbar.set_label("final output-arrival error (steps)")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _recovery_summary(protocol: dict[str, Any], rows: list[dict[str, Any]], output: Path) -> None:
    combinations = [(path, loss) for path in protocol["paths"] for loss in protocol["loss_modes"]]
    learning_rates = [float(value) for value in protocol["delay"]["learning_rates"]]
    x = np.arange(len(combinations))
    width = 0.24
    fig, axis = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    colors = ["#457b9d", "#f4a261", "#2a9d8f"]
    for index, (learning_rate, color) in enumerate(zip(learning_rates, colors)):
        counts = []
        for path, loss_mode in combinations:
            selected = [
                row for row in rows if row["path"] == path and row["loss_mode"] == loss_mode
                and math.isclose(float(row["learning_rate"]), learning_rate)
            ]
            counts.append(sum(bool(row["recovered_within_0p1_arrival_step"]) for row in selected))
        bars = axis.bar(x + (index - 1) * width, counts, width, color=color, label=f"lr={learning_rate:g}")
        axis.bar_label(bars, fontsize=8)
    axis.axhline(15, color="#bc4749", linestyle="--", linewidth=1.2, label="15/15")
    axis.set_xticks(x, [f"{path}\n{loss}" for path, loss in combinations])
    axis.set(ylabel="recovered cells out of 15", ylim=(0, 16), title="Level 0B recovery by forward path, timing loss, and learning rate")
    axis.legend(frameon=False, ncol=4, loc="upper center")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _gradient_direction_stats(
    protocol: dict[str, Any], rows: list[dict[str, Any]], path: str, loss_mode: str
) -> dict[str, int]:
    reference_lr = float(protocol["delay"]["learning_rates"][0])
    selected = [
        row for row in rows if row["path"] == path and row["loss_mode"] == loss_mode
        and math.isclose(float(row["learning_rate"]), reference_lr)
    ]
    correct = zero = wrong = already_aligned = 0
    for row in selected:
        signed_error = (
            float(row["initial_output_arrival_centroid"])
            - float(row["target_arrival_step"])
        )
        if abs(signed_error) <= 0.1:
            already_aligned += 1
            continue
        gradient = float(row["initial_raw_gradient"])
        if abs(gradient) <= 1e-12:
            zero += 1
        elif gradient * signed_error > 0:
            correct += 1
        else:
            wrong += 1
    return {
        "directional_pairs": correct + zero + wrong,
        "correct_sign": correct,
        "zero_gradient": zero,
        "wrong_nonzero_sign": wrong,
        "already_aligned": already_aligned,
    }


def _gradient_direction_summary(
    protocol: dict[str, Any], rows: list[dict[str, Any]], output: Path
) -> None:
    combinations = [(path, loss) for path in protocol["paths"] for loss in protocol["loss_modes"]]
    stats = [_gradient_direction_stats(protocol, rows, path, loss) for path, loss in combinations]
    x = np.arange(len(combinations))
    correct = np.asarray([item["correct_sign"] for item in stats])
    zero = np.asarray([item["zero_gradient"] for item in stats])
    wrong = np.asarray([item["wrong_nonzero_sign"] for item in stats])
    fig, axis = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    bars_correct = axis.bar(x, correct, color="#2a9d8f", label="correct direction")
    bars_zero = axis.bar(x, zero, bottom=correct, color="#adb5bd", label="zero gradient")
    bars_wrong = axis.bar(x, wrong, bottom=correct + zero, color="#bc4749", label="wrong direction")
    for bars in (bars_correct, bars_zero, bars_wrong):
        axis.bar_label(bars, label_type="center", fontsize=8)
    axis.set_xticks(x, [f"{path}\n{loss}" for path, loss in combinations])
    axis.set(
        ylabel="initially misaligned target/init pairs",
        title="Level 0B initial delay-gradient direction (13 misaligned pairs per condition)",
        ylim=(0, 14),
    )
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def aggregate(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (
        row["path"], row["loss_mode"], float(row["learning_rate"]),
        float(row["target_nominal_delay_steps"]), float(row["initial_raw"]),
    ))
    _write_csv(rows, SUMMARY_ROOT / "cells.csv")
    for path in protocol["paths"]:
        for loss_mode in protocol["loss_modes"]:
            _error_heatmap(
                protocol, rows, path=path, loss_mode=loss_mode,
                output=SUMMARY_ROOT / f"final_error_{path}_{loss_mode}.png",
            )
    _recovery_summary(protocol, rows, SUMMARY_ROOT / "path_loss_recovery_summary.png")
    _gradient_direction_summary(
        protocol, rows, SUMMARY_ROOT / "initial_gradient_direction_summary.png"
    )

    current = protocol["decision_rules"]["current_recipe_analogue"]
    combination_results = []
    for path in protocol["paths"]:
        for loss_mode in protocol["loss_modes"]:
            selected = [row for row in rows if row["path"] == path and row["loss_mode"] == loss_mode]
            coverage = []
            for target in protocol["timeline"]["target_nominal_delays_steps"]:
                for initial_raw in protocol["delay"]["initial_raw_values"]:
                    candidates = [
                        row for row in selected
                        if math.isclose(float(row["target_nominal_delay_steps"]), float(target))
                        and math.isclose(float(row["initial_raw"]), float(initial_raw))
                    ]
                    coverage.append(bool(any(row["recovered_within_0p1_arrival_step"] for row in candidates)))
            current_row = next(
                row for row in selected
                if math.isclose(float(row["target_nominal_delay_steps"]), float(current["target_nominal_delay_steps"]))
                and math.isclose(float(row["initial_raw"]), float(current["initial_raw"]))
                and math.isclose(float(row["learning_rate"]), float(current["learning_rate"]))
            )
            per_lr = {
                f"{float(lr):g}": sum(
                    bool(row["recovered_within_0p1_arrival_step"]) for row in selected
                    if math.isclose(float(row["learning_rate"]), float(lr))
                )
                for lr in protocol["delay"]["learning_rates"]
            }
            combination_results.append({
                "path": path,
                "loss_mode": loss_mode,
                "recovered_pairs_by_any_lr": int(sum(coverage)),
                "total_target_initialization_pairs": len(coverage),
                "recoverability_pass": bool(all(coverage)),
                "recovered_cells_per_learning_rate": per_lr,
                "initial_gradient_direction": _gradient_direction_stats(
                    protocol, rows, path, loss_mode
                ),
                "current_recipe_analogue": {
                    "final_arrival_error_steps": float(current_row["final_output_arrival_error_steps"]),
                    "final_delay_steps": float(current_row["final_delay_steps"]),
                    "final_trace_mass": float(current_row["final_output_trace_mass"]),
                    "convergence_step": current_row["convergence_step"],
                    "pass": bool(current_row["recovered_within_0p1_arrival_step"]),
                },
            })
    strict_pass = all(item["recoverability_pass"] for item in combination_results)
    decision = {
        "protocol_id": PROTOCOL_ID,
        "expected_cells": expected_cells(protocol),
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == expected_cells(protocol) and all(row["complete"] for row in rows),
        "path_loss_results": combination_results,
        "level0b_strict_pass": bool(strict_pass),
        "level1_xor_authorized": bool(strict_pass),
        "xor_learning_rate_change_authorized": False,
        "routing_or_accuracy_claim_authorized": False,
    }
    (SUMMARY_ROOT / "decision.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return decision


def main() -> None:
    protocol = load_protocol()
    if protocol["status"] != "preregistered_ready" and not str(protocol["status"]).startswith("completed"):
        raise SystemExit(f"protocol status {protocol['status']!r} is not launchable")
    if expected_cells(protocol) != int(protocol["grid"]["deterministic_cells"]):
        raise SystemExit("declared Level-0B cell count does not match the grid")
    rows = []
    for path in protocol["paths"]:
        for loss_mode in protocol["loss_modes"]:
            for learning_rate in [float(value) for value in protocol["delay"]["learning_rates"]]:
                for target in [float(value) for value in protocol["timeline"]["target_nominal_delays_steps"]]:
                    for initial_raw in [float(value) for value in protocol["delay"]["initial_raw_values"]]:
                        rows.append(run_cell(
                            protocol, path=path, loss_mode=loss_mode,
                            learning_rate=learning_rate, target=target,
                            initial_raw=initial_raw,
                        ))
    print(json.dumps(aggregate(protocol, rows), indent=2))


if __name__ == "__main__":
    main()

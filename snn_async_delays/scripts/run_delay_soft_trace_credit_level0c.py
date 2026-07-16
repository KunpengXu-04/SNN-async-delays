"""Run the preregistered Level 0C soft-trace temporal-credit diagnostic."""

from __future__ import annotations

import argparse
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
PROTOCOL_ID = "delay_soft_trace_credit_level0c_v1"
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


def expected_cells(protocol: dict[str, Any]) -> int:
    return (
        len(protocol["delay"]["parameterizations"])
        * len(protocol["paths"])
        * len(protocol["objectives"])
        * len(protocol["timeline"]["target_nominal_delays_steps"])
        * len(protocol["delay"]["matched_initialization_labels_raw"])
        * len(protocol["delay"]["learning_rates"])
    )


def target_arrival_step(protocol: dict[str, Any], nominal_delay: float) -> int:
    return int(protocol["timeline"]["input_spike_step"] + nominal_delay + 1)


def matched_initial_delay(protocol: dict[str, Any], initialization_label: float) -> float:
    d_max = float(protocol["timeline"]["d_max_steps"])
    return d_max / (1.0 + math.exp(-float(initialization_label)))


def initial_parameter_value(
    protocol: dict[str, Any], parameterization: str, initialization_label: float
) -> float:
    if parameterization == "sigmoid":
        return float(initialization_label)
    if parameterization == "direct":
        return matched_initial_delay(protocol, initialization_label)
    raise ValueError(parameterization)


def _token(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def cell_directory(
    root: Path,
    parameterization: str,
    path: str,
    objective: str,
    learning_rate: float,
    target: float,
    initialization_label: float,
) -> Path:
    return (
        root
        / parameterization
        / path
        / objective
        / f"lr_{_token(learning_rate)}"
        / f"target_{_token(target)}"
        / f"init_label_{_token(initialization_label)}"
    )


def make_delay_layer(
    protocol: dict[str, Any],
    *,
    parameterization: str,
    initialization_label: float,
    fixed_weight: float,
) -> DelayedSynapticLayer:
    torch.manual_seed(0)
    layer = DelayedSynapticLayer(
        1,
        1,
        d_max=int(protocol["timeline"]["d_max_steps"]),
        delay_param_type=parameterization,
        delay_init_mode="constant",
        delay_init_raw=initial_parameter_value(
            protocol, parameterization, initialization_label
        ),
        train_weights=False,
        train_delays=True,
    )
    with torch.no_grad():
        layer.weight.fill_(float(fixed_weight))
    return layer


def make_lif(config: dict[str, Any]) -> LIFNeurons:
    return LIFNeurons(
        1,
        tau_m=float(config["tau_m_steps"]),
        v_threshold=float(config["threshold_au"]),
        v_reset=float(config["reset_au"]),
        refractory_steps=int(config["refractory_steps"]),
        dt=float(config["dt_steps"]),
        surrogate_beta=float(config["surrogate_beta"]),
    )


def lif_response(
    current_trace: torch.Tensor, lif_config: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    lif = make_lif(lif_config)
    voltage, refractory = lif.init_state(1)
    spikes: list[torch.Tensor] = []
    membranes: list[torch.Tensor] = []
    for current in current_trace:
        spike, voltage, refractory = lif(
            current.reshape(1, 1), voltage, refractory
        )
        spikes.append(spike.reshape(()))
        membranes.append(voltage.reshape(()))
    return torch.stack(membranes), torch.stack(spikes)


def render_soft_trace(
    layer: DelayedSynapticLayer,
    *,
    path: str,
    total_steps: int,
    input_spike_step: int,
    lif_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if path not in {"buffer_current", "lif_membrane"}:
        raise ValueError(path)
    buffer = torch.zeros(1, layer.d_max + 1, 1)
    pointer = 0
    delays = layer.get_delays()
    currents: list[torch.Tensor] = []
    for time_index in range(int(total_steps)):
        current = layer(buffer, delays, pointer).reshape(())
        currents.append(current)
        buffer[:, pointer, :] = 1.0 if time_index == int(input_spike_step) else 0.0
        pointer = (pointer + 1) % (layer.d_max + 1)
    current_trace = torch.stack(currents)
    if path == "buffer_current":
        return {
            "trace": current_trace,
            "current": current_trace,
            "membrane": torch.zeros_like(current_trace),
            "spike": torch.zeros_like(current_trace),
        }
    membrane, spikes = lif_response(current_trace, lif_config)
    return {
        "trace": membrane,
        "current": current_trace,
        "membrane": membrane,
        "spike": spikes,
    }


def target_template(
    protocol: dict[str, Any], *, path: str, target_nominal_delay: float
) -> dict[str, torch.Tensor]:
    total_steps = int(protocol["timeline"]["total_steps"])
    arrival = target_arrival_step(protocol, target_nominal_delay)
    current = torch.zeros(total_steps)
    current[arrival] = float(protocol["paths"][path]["fixed_synaptic_weight"])
    if path == "buffer_current":
        return {
            "trace": current,
            "current": current,
            "membrane": torch.zeros_like(current),
            "spike": torch.zeros_like(current),
        }
    membrane, spikes = lif_response(current, protocol["lif"])
    return {
        "trace": membrane.detach(),
        "current": current,
        "membrane": membrane.detach(),
        "spike": spikes.detach(),
    }


def normalized_trace(trace: torch.Tensor) -> torch.Tensor:
    if bool((trace.detach() < -1e-10).any()):
        raise ValueError("Level 0C requires a nonnegative soft trace")
    return trace / (trace.sum() + 1e-8)


def trace_centroid(trace: torch.Tensor) -> torch.Tensor:
    probability = normalized_trace(trace)
    time = torch.arange(trace.numel(), dtype=trace.dtype, device=trace.device)
    return (time * probability).sum()


def trace_w1_steps(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    output_cdf = torch.cumsum(normalized_trace(output), dim=0)
    target_cdf = torch.cumsum(normalized_trace(target), dim=0)
    return (output_cdf - target_cdf).abs().sum()


def causal_filter(trace: torch.Tensor, tau_steps: float) -> torch.Tensor:
    decay = float(math.exp(-1.0 / float(tau_steps)))
    state = torch.zeros((), dtype=trace.dtype, device=trace.device)
    values = []
    for value in trace:
        state = decay * state + value
        values.append(state)
    return torch.stack(values)


def objective_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    *,
    objective: str,
    filter_tau_steps: float,
    kernel_tau_steps: float,
) -> torch.Tensor:
    if objective == "causal_filtered_trace":
        filtered_output = causal_filter(output, filter_tau_steps)
        filtered_target = causal_filter(target, filter_tau_steps)
        return (filtered_output - filtered_target).pow(2).mean()
    if objective == "soft_centroid":
        return 0.5 * (trace_centroid(output) - trace_centroid(target)).pow(2)
    if objective == "symmetric_kernel_alignment":
        output_probability = normalized_trace(output)
        target_probability = normalized_trace(target)
        time = torch.arange(output.numel(), dtype=output.dtype, device=output.device)
        distance = (time[:, None] - time[None, :]).abs()
        kernel = torch.exp(-distance / float(kernel_tau_steps))
        similarity = output_probability @ kernel @ target_probability
        return -torch.log(similarity + 1e-8)
    raise ValueError(objective)


def delay_jacobian(layer: DelayedSynapticLayer, parameterization: str) -> float:
    if parameterization == "direct":
        raw = float(layer.delay_raw.detach().item())
        return 1.0 if 0.0 < raw < float(layer.d_max) else 0.0
    raw = layer.delay_raw.detach()
    sigmoid = torch.sigmoid(raw)
    return float((float(layer.d_max) * sigmoid * (1.0 - sigmoid)).item())


def optimize_soft_credit(
    protocol: dict[str, Any],
    *,
    parameterization: str,
    path: str,
    objective: str,
    target_nominal_delay: float,
    initialization_label: float,
    learning_rate: float,
) -> dict[str, Any]:
    path_config = protocol["paths"][path]
    layer = make_delay_layer(
        protocol,
        parameterization=parameterization,
        initialization_label=initialization_label,
        fixed_weight=float(path_config["fixed_synaptic_weight"]),
    )
    optimizer = torch.optim.Adam([layer.delay_raw], lr=float(learning_rate))
    target = target_template(
        protocol, path=path, target_nominal_delay=target_nominal_delay
    )
    target_trace = target["trace"]
    target_centroid = float(trace_centroid(target_trace).detach().item())
    optimizer_steps = int(protocol["delay"]["optimizer_steps"])
    tolerance = float(protocol["metrics"]["recovery_tolerance_steps"])
    min_mass = float(protocol["metrics"]["minimum_output_trace_mass"])
    filter_tau = float(
        protocol["objectives"]["causal_filtered_trace"]["filter_tau_steps"]
    )
    kernel_tau = float(
        protocol["objectives"]["symmetric_kernel_alignment"]["kernel_tau_steps"]
    )

    steps: list[int] = []
    raw_values: list[float] = []
    delay_values: list[float] = []
    losses: list[float] = []
    centroids: list[float] = []
    w1_errors: list[float] = []
    masses: list[float] = []
    gradients: list[float] = []
    hard_spike_counts: list[float] = []
    initial_rendered = final_rendered = None
    convergence_step = None

    for step in range(optimizer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        rendered = render_soft_trace(
            layer,
            path=path,
            total_steps=int(protocol["timeline"]["total_steps"]),
            input_spike_step=int(protocol["timeline"]["input_spike_step"]),
            lif_config=protocol["lif"],
        )
        trace = rendered["trace"]
        loss = objective_loss(
            trace, target_trace, objective=objective,
            filter_tau_steps=filter_tau, kernel_tau_steps=kernel_tau,
        )
        centroid = trace_centroid(trace)
        w1 = trace_w1_steps(trace, target_trace)
        loss.backward()

        raw_gradient = float(layer.delay_raw.grad.detach().item())
        delay_value = float(layer.get_delays().detach().item())
        trace_mass = float(trace.detach().sum().item())
        hard_spike_count = float(rendered["spike"].detach().sum().item())
        valid = hard_spike_count == 0.0

        steps.append(step)
        raw_values.append(float(layer.delay_raw.detach().item()))
        delay_values.append(delay_value)
        losses.append(float(loss.detach().item()))
        centroids.append(float(centroid.detach().item()))
        w1_errors.append(float(w1.detach().item()))
        masses.append(trace_mass)
        gradients.append(raw_gradient)
        hard_spike_counts.append(hard_spike_count)

        if step == 0:
            initial_rendered = {
                key: value.detach().cpu().numpy() for key, value in rendered.items()
            }
        if convergence_step is None and w1_errors[-1] <= tolerance and trace_mass >= min_mass and valid:
            convergence_step = step
        if step < optimizer_steps:
            optimizer.step()
        else:
            final_rendered = {
                key: value.detach().cpu().numpy() for key, value in rendered.items()
            }

    assert initial_rendered is not None and final_rendered is not None
    initial_jacobian = (
        float(protocol["timeline"]["d_max_steps"])
        * (1.0 / (1.0 + math.exp(-float(initialization_label))))
        * (1.0 - 1.0 / (1.0 + math.exp(-float(initialization_label))))
        if parameterization == "sigmoid"
        else 1.0
    )
    initial_delay_gradient = (
        gradients[0] / initial_jacobian if abs(initial_jacobian) > 1e-12 else None
    )
    final_valid = max(hard_spike_counts) == 0.0
    recovered = (
        w1_errors[-1] <= tolerance
        and masses[-1] >= min_mass
        and final_valid
    )
    result = {
        "steps": np.asarray(steps, dtype=np.int64),
        "raw_values": np.asarray(raw_values, dtype=np.float64),
        "delay_values": np.asarray(delay_values, dtype=np.float64),
        "losses": np.asarray(losses, dtype=np.float64),
        "trace_centroids": np.asarray(centroids, dtype=np.float64),
        "w1_errors": np.asarray(w1_errors, dtype=np.float64),
        "trace_masses": np.asarray(masses, dtype=np.float64),
        "raw_gradients": np.asarray(gradients, dtype=np.float64),
        "hard_spike_counts": np.asarray(hard_spike_counts, dtype=np.float64),
        "initial_rendered": initial_rendered,
        "final_rendered": final_rendered,
        "target": {key: value.detach().cpu().numpy() for key, value in target.items()},
        "target_centroid": target_centroid,
        "target_arrival_step": target_arrival_step(protocol, target_nominal_delay),
        "initial_delay_jacobian": float(initial_jacobian),
        "initial_delay_space_gradient": initial_delay_gradient,
        "convergence_step": convergence_step,
        "recovered": bool(recovered),
        "path_valid": bool(final_valid),
    }
    return result


def metrics_from_result(
    protocol: dict[str, Any],
    result: dict[str, Any],
    *,
    parameterization: str,
    path: str,
    objective: str,
    target: float,
    initialization_label: float,
    learning_rate: float,
) -> dict[str, Any]:
    delays = result["delay_values"]
    centroids = result["trace_centroids"]
    target_centroid = float(result["target_centroid"])
    return {
        "protocol_id": PROTOCOL_ID,
        "parameterization": parameterization,
        "path": path,
        "objective": objective,
        "objective_role": protocol["objectives"][objective]["role"],
        "learning_rate": float(learning_rate),
        "target_nominal_delay_steps": float(target),
        "target_arrival_step": int(result["target_arrival_step"]),
        "initialization_label_raw": float(initialization_label),
        "initial_parameter_value": initial_parameter_value(
            protocol, parameterization, initialization_label
        ),
        "optimizer_steps": int(protocol["delay"]["optimizer_steps"]),
        "direct_delay_supervision": False,
        "stochastic_seed": None,
        "test_split_opened": False,
        "initial_delay_steps": float(delays[0]),
        "final_delay_steps": float(delays[-1]),
        "final_nominal_delay_error_steps": abs(float(delays[-1]) - float(target)),
        "target_trace_centroid": target_centroid,
        "initial_trace_centroid": float(centroids[0]),
        "final_trace_centroid": float(centroids[-1]),
        "final_trace_centroid_error_steps": abs(float(centroids[-1]) - target_centroid),
        "initial_normalized_trace_w1_steps": float(result["w1_errors"][0]),
        "final_normalized_trace_w1_steps": float(result["w1_errors"][-1]),
        "initial_output_trace_mass": float(result["trace_masses"][0]),
        "final_output_trace_mass": float(result["trace_masses"][-1]),
        "initial_raw_gradient": float(result["raw_gradients"][0]),
        "initial_delay_jacobian": float(result["initial_delay_jacobian"]),
        "initial_delay_space_gradient": result["initial_delay_space_gradient"],
        "final_raw_gradient": float(result["raw_gradients"][-1]),
        "maximum_hard_spike_count": float(result["hard_spike_counts"].max()),
        "path_valid": bool(result["path_valid"]),
        "convergence_step": result["convergence_step"],
        "final_loss": float(result["losses"][-1]),
        "recovered_within_0p1_w1_step": bool(result["recovered"]),
        "complete": True,
    }


def save_diagnostic_panel(
    protocol: dict[str, Any],
    result: dict[str, Any],
    metrics: dict[str, Any],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    steps = result["steps"]
    target_delay = metrics["target_nominal_delay_steps"]
    target_centroid = metrics["target_trace_centroid"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Level 0C | {metrics['parameterization']} | {metrics['path']} | "
        f"{metrics['objective']} | target d={target_delay:g}, lr={metrics['learning_rate']:g}"
    )

    axes[0, 0].plot(steps, result["delay_values"], color="#1f8a9b")
    axes[0, 0].axhline(target_delay, color="#c44e52", linestyle="--", label="target")
    axes[0, 0].set(title="Delay trajectory", xlabel="optimizer step", ylabel="delay (steps)")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(steps, result["trace_centroids"], color="#6a4c93")
    axes[0, 1].axhline(target_centroid, color="#c44e52", linestyle="--", label="target")
    axes[0, 1].set(title="Soft temporal coordinate", xlabel="optimizer step", ylabel="trace centroid")
    axes[0, 1].legend(frameon=False)

    axes[0, 2].semilogy(steps, np.maximum(result["w1_errors"], 1e-10), color="#6a4c93")
    axes[0, 2].axhline(
        float(protocol["metrics"]["recovery_tolerance_steps"]),
        color="#c44e52", linestyle="--", label="0.1-step gate",
    )
    axes[0, 2].set(title="Primary functional error", xlabel="optimizer step", ylabel="normalized W1 (steps)")
    axes[0, 2].legend(frameon=False)

    axes[1, 0].semilogy(steps, np.maximum(result["losses"], 1e-12), color="#386641")
    axes[1, 0].set(title="Timing objective", xlabel="optimizer step", ylabel="loss")

    axes[1, 1].plot(steps, result["raw_gradients"], color="#f4a261")
    axes[1, 1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1, 1].set(title="Delay credit", xlabel="optimizer step", ylabel="dL / d(parameter)")

    time = np.arange(int(protocol["timeline"]["total_steps"]))
    axes[1, 2].plot(time, result["target"]["trace"], "--", color="#c44e52", label="target")
    axes[1, 2].plot(time, result["initial_rendered"]["trace"], color="#7aa6c2", label="initial")
    axes[1, 2].plot(time, result["final_rendered"]["trace"], color="#2a9d8f", label="final")
    axes[1, 2].set(title="Soft output trace", xlabel="simulation step", ylabel="trace value")
    axes[1, 2].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def required_cell_artifacts(directory: Path) -> list[Path]:
    return [
        directory / "config.json",
        directory / "metrics.json",
        directory / "final_parameter.pt",
        directory / "plots" / "diagnostic_data.npz",
        directory / "plots" / "diagnostic_panel.png",
    ]


def run_cell(
    protocol: dict[str, Any],
    *,
    root: Path,
    parameterization: str,
    path: str,
    objective: str,
    learning_rate: float,
    target: float,
    initialization_label: float,
) -> dict[str, Any]:
    directory = cell_directory(
        root, parameterization, path, objective, learning_rate, target,
        initialization_label,
    )
    metrics_path = directory / "metrics.json"
    if all(item.exists() for item in required_cell_artifacts(directory)):
        with metrics_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("complete"):
            return existing

    directory.mkdir(parents=True, exist_ok=True)
    cell_config = {
        "protocol_id": PROTOCOL_ID,
        "parameterization": parameterization,
        "path": path,
        "objective": objective,
        "learning_rate": float(learning_rate),
        "target_nominal_delay_steps": float(target),
        "initialization_label_raw": float(initialization_label),
        "initial_parameter_value": initial_parameter_value(
            protocol, parameterization, initialization_label
        ),
        "fixed_synaptic_weight": float(protocol["paths"][path]["fixed_synaptic_weight"]),
        "optimizer_steps": int(protocol["delay"]["optimizer_steps"]),
        "test_split_opened": False,
    }
    (directory / "config.json").write_text(
        json.dumps(cell_config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    result = optimize_soft_credit(
        protocol,
        parameterization=parameterization,
        path=path,
        objective=objective,
        target_nominal_delay=target,
        initialization_label=initialization_label,
        learning_rate=learning_rate,
    )
    metrics = metrics_from_result(
        protocol,
        result,
        parameterization=parameterization,
        path=path,
        objective=objective,
        target=target,
        initialization_label=initialization_label,
        learning_rate=learning_rate,
    )
    torch.save(
        {
            "parameterization": parameterization,
            "delay_parameter": float(result["raw_values"][-1]),
            "delay_steps": float(result["delay_values"][-1]),
        },
        directory / "final_parameter.pt",
    )
    plots = directory / "plots"
    plots.mkdir(exist_ok=True)
    np.savez_compressed(
        plots / "diagnostic_data.npz",
        steps=result["steps"],
        raw_values=result["raw_values"],
        delay_values=result["delay_values"],
        losses=result["losses"],
        trace_centroids=result["trace_centroids"],
        w1_errors=result["w1_errors"],
        trace_masses=result["trace_masses"],
        raw_gradients=result["raw_gradients"],
        hard_spike_counts=result["hard_spike_counts"],
        initial_trace=result["initial_rendered"]["trace"],
        final_trace=result["final_rendered"]["trace"],
        target_trace=result["target"]["trace"],
        initial_current=result["initial_rendered"]["current"],
        final_current=result["final_rendered"]["current"],
        initial_membrane=result["initial_rendered"]["membrane"],
        final_membrane=result["final_rendered"]["membrane"],
    )
    save_diagnostic_panel(protocol, result, metrics, plots / "diagnostic_panel.png")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metrics


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gradient_direction_stats(
    protocol: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    parameterization: str,
    path: str,
    objective: str,
) -> dict[str, int]:
    reference_lr = float(protocol["delay"]["learning_rates"][0])
    selected = [
        row for row in rows
        if row["parameterization"] == parameterization
        and row["path"] == path
        and row["objective"] == objective
        and math.isclose(float(row["learning_rate"]), reference_lr)
    ]
    correct = zero = wrong = aligned = 0
    zero_tolerance = float(protocol["metrics"]["initial_gradient_zero_tolerance"])
    recovery_tolerance = float(protocol["metrics"]["recovery_tolerance_steps"])
    for row in selected:
        if float(row["initial_normalized_trace_w1_steps"]) <= recovery_tolerance:
            aligned += 1
            continue
        signed_error = float(row["initial_trace_centroid"]) - float(row["target_trace_centroid"])
        gradient = float(row["initial_raw_gradient"])
        if abs(gradient) <= zero_tolerance:
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
        "already_aligned": aligned,
    }


def _error_heatmap(
    protocol: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    parameterization: str,
    path: str,
    objective: str,
    output: Path,
) -> None:
    learning_rates = [float(value) for value in protocol["delay"]["learning_rates"]]
    targets = [float(value) for value in protocol["timeline"]["target_nominal_delays_steps"]]
    initializations = [float(value) for value in protocol["delay"]["matched_initialization_labels_raw"]]
    selected = [
        row for row in rows
        if row["parameterization"] == parameterization
        and row["path"] == path
        and row["objective"] == objective
    ]
    matrices = []
    for learning_rate in learning_rates:
        matrix = np.zeros((len(initializations), len(targets)))
        for i, initialization in enumerate(initializations):
            for j, target in enumerate(targets):
                row = next(
                    item for item in selected
                    if math.isclose(float(item["learning_rate"]), learning_rate)
                    and math.isclose(float(item["target_nominal_delay_steps"]), target)
                    and math.isclose(float(item["initialization_label_raw"]), initialization)
                )
                matrix[i, j] = float(row["final_normalized_trace_w1_steps"])
        matrices.append(matrix)
    vmax = max(0.1, float(max(matrix.max() for matrix in matrices)))
    fig, axes = plt.subplots(1, len(learning_rates), figsize=(9.5, 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    image = None
    for axis, learning_rate, matrix in zip(axes, learning_rates, matrices):
        image = axis.imshow(matrix, vmin=0.0, vmax=vmax, cmap="magma_r", aspect="auto")
        axis.set_xticks(range(len(targets)), [f"{value:g}" for value in targets])
        axis.set_yticks(range(len(initializations)), [f"{value:g}" for value in initializations])
        axis.set(title=f"Adam lr={learning_rate:g}", xlabel="target delay label", ylabel="initial raw label")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                color = "white" if matrix[i, j] / vmax > 0.58 else "black"
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.suptitle(f"Level 0C final W1 | {parameterization} | {path} | {objective}")
    assert image is not None
    fig.colorbar(image, ax=axes.tolist(), label="normalized trace W1 (steps)", shrink=0.84)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _recovery_summary(
    protocol: dict[str, Any], rows: list[dict[str, Any]], output: Path
) -> None:
    parameterizations = list(protocol["delay"]["parameterizations"])
    learning_rates = [float(value) for value in protocol["delay"]["learning_rates"]]
    combinations = [(path, objective) for path in protocol["paths"] for objective in protocol["objectives"]]
    short = {
        "causal_filtered_trace": "causal filtered",
        "soft_centroid": "soft centroid",
        "symmetric_kernel_alignment": "symmetric kernel",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    colors = ["#577590", "#2a9d8f"]
    width = 0.35
    x = np.arange(len(combinations))
    for axis, parameterization in zip(axes, parameterizations):
        for offset, (learning_rate, color) in enumerate(zip(learning_rates, colors)):
            values = []
            for path, objective in combinations:
                values.append(sum(
                    bool(row["recovered_within_0p1_w1_step"])
                    for row in rows
                    if row["parameterization"] == parameterization
                    and row["path"] == path
                    and row["objective"] == objective
                    and math.isclose(float(row["learning_rate"]), learning_rate)
                ))
            positions = x + (offset - 0.5) * width
            bars = axis.bar(positions, values, width=width, color=color, label=f"lr={learning_rate:g}")
            axis.bar_label(bars, fontsize=8)
        axis.axhline(15, color="#c44e52", linestyle="--", linewidth=1)
        axis.set_xticks(
            x,
            [f"{'buffer' if path == 'buffer_current' else 'membrane'}\n{short[objective]}" for path, objective in combinations],
            rotation=16,
            ha="center",
        )
        axis.set(title=parameterization, ylabel="recovered target/init cells (of 15)", ylim=(0, 16))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle("Level 0C fixed-learning-rate recovery")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _gradient_summary(
    protocol: dict[str, Any], rows: list[dict[str, Any]], output: Path
) -> None:
    parameterizations = list(protocol["delay"]["parameterizations"])
    combinations = [(path, objective) for path in protocol["paths"] for objective in protocol["objectives"]]
    short = {
        "causal_filtered_trace": "causal filtered",
        "soft_centroid": "soft centroid",
        "symmetric_kernel_alignment": "symmetric kernel",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    for axis, parameterization in zip(axes, parameterizations):
        stats = [
            gradient_direction_stats(
                protocol, rows, parameterization=parameterization, path=path,
                objective=objective,
            )
            for path, objective in combinations
        ]
        x = np.arange(len(combinations))
        correct = np.asarray([item["correct_sign"] for item in stats])
        zero = np.asarray([item["zero_gradient"] for item in stats])
        wrong = np.asarray([item["wrong_nonzero_sign"] for item in stats])
        bars_correct = axis.bar(x, correct, color="#2a9d8f", label="correct")
        bars_zero = axis.bar(x, zero, bottom=correct, color="#adb5bd", label="zero")
        bars_wrong = axis.bar(x, wrong, bottom=correct + zero, color="#bc4749", label="wrong")
        for bars in (bars_correct, bars_zero, bars_wrong):
            labels = [f"{value:g}" if value else "" for value in bars.datavalues]
            axis.bar_label(bars, labels=labels, label_type="center", fontsize=8)
        axis.set_xticks(
            x,
            [f"{'buffer' if path == 'buffer_current' else 'membrane'}\n{short[objective]}" for path, objective in combinations],
            rotation=16,
            ha="center",
        )
        axis.set(title=parameterization, ylabel="initially misaligned pairs", ylim=(0, 14))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, ncol=3)
    fig.suptitle("Level 0C initial delay-gradient direction")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def aggregate(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (
        row["parameterization"], row["path"], row["objective"],
        float(row["learning_rate"]), float(row["target_nominal_delay_steps"]),
        float(row["initialization_label_raw"]),
    ))
    _write_csv(rows, SUMMARY_ROOT / "cells.csv")
    combination_results = []
    for parameterization in protocol["delay"]["parameterizations"]:
        for path in protocol["paths"]:
            for objective in protocol["objectives"]:
                selected = [
                    row for row in rows
                    if row["parameterization"] == parameterization
                    and row["path"] == path
                    and row["objective"] == objective
                ]
                per_lr = {
                    f"{float(learning_rate):g}": sum(
                        bool(row["recovered_within_0p1_w1_step"])
                        for row in selected
                        if math.isclose(float(row["learning_rate"]), float(learning_rate))
                    )
                    for learning_rate in protocol["delay"]["learning_rates"]
                }
                coverage_any_lr = 0
                for target in protocol["timeline"]["target_nominal_delays_steps"]:
                    for initialization in protocol["delay"]["matched_initialization_labels_raw"]:
                        coverage_any_lr += int(any(
                            bool(row["recovered_within_0p1_w1_step"])
                            for row in selected
                            if math.isclose(float(row["target_nominal_delay_steps"]), float(target))
                            and math.isclose(float(row["initialization_label_raw"]), float(initialization))
                        ))
                combination_results.append({
                    "parameterization": parameterization,
                    "path": path,
                    "objective": objective,
                    "objective_role": protocol["objectives"][objective]["role"],
                    "recovered_pairs_by_any_lr": coverage_any_lr,
                    "recovered_cells_per_learning_rate": per_lr,
                    "fixed_lr_path_passes": [
                        key for key, value in per_lr.items() if value == 15
                    ],
                    "initial_gradient_direction": gradient_direction_stats(
                        protocol, rows, parameterization=parameterization,
                        path=path, objective=objective,
                    ),
                    "all_cells_path_valid": all(bool(row["path_valid"]) for row in selected),
                })
                _error_heatmap(
                    protocol, rows, parameterization=parameterization, path=path,
                    objective=objective,
                    output=SUMMARY_ROOT / f"final_w1_{parameterization}_{path}_{objective}.png",
                )
    _recovery_summary(protocol, rows, SUMMARY_ROOT / "fixed_lr_recovery_summary.png")
    _gradient_summary(protocol, rows, SUMMARY_ROOT / "initial_gradient_direction_summary.png")

    candidate_evaluations = []
    for objective in [
        name for name, config in protocol["objectives"].items()
        if config["role"] == "candidate"
    ]:
        for learning_rate in [float(value) for value in protocol["delay"]["learning_rates"]]:
            relevant = [
                row for row in rows
                if row["parameterization"] == "sigmoid"
                and row["objective"] == objective
                and math.isclose(float(row["learning_rate"]), learning_rate)
            ]
            directions = [
                gradient_direction_stats(
                    protocol, rows, parameterization="sigmoid", path=path,
                    objective=objective,
                )
                for path in protocol["paths"]
            ]
            direction_pass = all(
                item["correct_sign"] == item["directional_pairs"]
                and item["zero_gradient"] == 0
                and item["wrong_nonzero_sign"] == 0
                for item in directions
            )
            recovered = sum(bool(row["recovered_within_0p1_w1_step"]) for row in relevant)
            candidate_evaluations.append({
                "objective": objective,
                "learning_rate": learning_rate,
                "recovered_sigmoid_cells_across_both_paths": recovered,
                "total_sigmoid_cells_across_both_paths": 30,
                "gradient_direction_pass_both_paths": direction_pass,
                "all_paths_valid": all(bool(row["path_valid"]) for row in relevant),
                "production_candidate_pass": bool(
                    recovered == 30 and direction_pass
                    and all(bool(row["path_valid"]) for row in relevant)
                ),
            })
    passing = [item for item in candidate_evaluations if item["production_candidate_pass"]]
    selected = None
    for objective in protocol["decision_rules"]["selection_priority"]["objectives"]:
        objective_passes = sorted(
            [item for item in passing if item["objective"] == objective],
            key=lambda item: float(item["learning_rate"]),
        )
        if objective_passes:
            selected = objective_passes[0]
            break
    strict_pass = selected is not None
    decision = {
        "protocol_id": PROTOCOL_ID,
        "expected_cells": expected_cells(protocol),
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == expected_cells(protocol) and all(bool(row["complete"]) for row in rows),
        "combination_results": combination_results,
        "candidate_evaluations": candidate_evaluations,
        "selected_candidate": selected,
        "level0c_pass": strict_pass,
        "level0d_authorized": strict_pass,
        "level1_xor_authorized": False,
        "routing_or_accuracy_claim_authorized": False,
    }
    (SUMMARY_ROOT / "decision.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return decision


def run_formal(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for parameterization in protocol["delay"]["parameterizations"]:
        for path in protocol["paths"]:
            for objective in protocol["objectives"]:
                for learning_rate in [float(value) for value in protocol["delay"]["learning_rates"]]:
                    for target in [float(value) for value in protocol["timeline"]["target_nominal_delays_steps"]]:
                        for initialization in [float(value) for value in protocol["delay"]["matched_initialization_labels_raw"]]:
                            rows.append(run_cell(
                                protocol,
                                root=RUN_ROOT,
                                parameterization=parameterization,
                                path=path,
                                objective=objective,
                                learning_rate=learning_rate,
                                target=target,
                                initialization_label=initialization,
                            ))
    return aggregate(protocol, rows)


def run_smoke(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for path in protocol["paths"]:
        for objective in ("soft_centroid", "symmetric_kernel_alignment"):
            for target, initialization in ((5.0, -2.0), (1.0, 0.0)):
                rows.append(run_cell(
                    protocol,
                    root=SMOKE_ROOT,
                    parameterization="sigmoid",
                    path=path,
                    objective=objective,
                    learning_rate=0.05,
                    target=target,
                    initialization_label=initialization,
                ))
    return {
        "smoke_cells": len(rows),
        "recovered": sum(bool(row["recovered_within_0p1_w1_step"]) for row in rows),
        "path_valid": all(bool(row["path_valid"]) for row in rows),
        "cells": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    if protocol["status"] != "preregistered_ready" and not str(protocol["status"]).startswith("completed"):
        raise SystemExit(f"protocol status {protocol['status']!r} is not launchable")
    if expected_cells(protocol) != int(protocol["grid"]["deterministic_cells"]):
        raise SystemExit("declared Level-0C cell count does not match the grid")
    result = run_smoke(protocol) if args.smoke else run_formal(protocol)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

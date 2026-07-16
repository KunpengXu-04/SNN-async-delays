"""Run the preregistered Level 0D hard-output / soft-credit diagnostic."""

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
PROTOCOL_ID = "delay_hard_output_soft_credit_level0d_v1"
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
        len(protocol["conditions"])
        * len(protocol["timeline"]["target_nominal_delays_steps"])
        * len(protocol["delay"]["initial_raw_values"])
    )


def condition_map(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in protocol["conditions"]}


def target_arrival_step(protocol: dict[str, Any], nominal_delay: float) -> int:
    return int(protocol["timeline"]["input_spike_step"] + nominal_delay + 1)


def _token(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def cell_directory(
    root: Path, condition: str, target: float, initial_raw: float
) -> Path:
    return root / condition / f"target_{_token(target)}" / f"init_raw_{_token(initial_raw)}"


def make_delay_layer(
    protocol: dict[str, Any], *, initial_raw: float
) -> DelayedSynapticLayer:
    torch.manual_seed(0)
    layer = DelayedSynapticLayer(
        1,
        1,
        d_max=int(protocol["timeline"]["d_max_steps"]),
        delay_param_type="sigmoid",
        delay_init_mode="constant",
        delay_init_raw=float(initial_raw),
        train_weights=False,
        train_delays=True,
    )
    with torch.no_grad():
        layer.weight.fill_(float(protocol["lif"]["fixed_synaptic_weight"]))
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


def lif_response_with_pre_reset(
    current_trace: torch.Tensor, lif_config: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Use the production LIF and expose the same pre-reset state as snn.model."""
    lif = make_lif(lif_config)
    voltage, refractory = lif.init_state(1)
    spikes: list[torch.Tensor] = []
    pre_reset: list[torch.Tensor] = []
    post_reset: list[torch.Tensor] = []
    for current in current_trace:
        not_refractory = (refractory <= 0.0).float()
        voltage_pre = (
            lif.decay * voltage
            + (1.0 - lif.decay) * current.reshape(1, 1) * not_refractory
        )
        spike, voltage, refractory = lif(
            current.reshape(1, 1), voltage, refractory
        )
        spikes.append(spike.reshape(()))
        pre_reset.append(voltage_pre.reshape(()))
        post_reset.append(voltage.reshape(()))
    return {
        "spike": torch.stack(spikes),
        "pre_reset_membrane": torch.stack(pre_reset),
        "post_reset_membrane": torch.stack(post_reset),
    }


def render_hard_path(
    layer: DelayedSynapticLayer,
    *,
    total_steps: int,
    input_spike_step: int,
    lif_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
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
    response = lif_response_with_pre_reset(current_trace, lif_config)
    return {"current": current_trace, **response}


def target_templates(
    protocol: dict[str, Any], target_nominal_delay: float
) -> dict[str, torch.Tensor]:
    total_steps = int(protocol["timeline"]["total_steps"])
    current = torch.zeros(total_steps)
    current[target_arrival_step(protocol, target_nominal_delay)] = float(
        protocol["lif"]["fixed_synaptic_weight"]
    )
    response = lif_response_with_pre_reset(current, protocol["lif"])
    return {
        "current": current,
        "spike": response["spike"].detach(),
        "pre_reset_membrane": response["pre_reset_membrane"].detach(),
        "post_reset_membrane": response["post_reset_membrane"].detach(),
    }


def normalized_trace(trace: torch.Tensor) -> torch.Tensor:
    if bool((trace.detach() < -1e-10).any()):
        raise ValueError("Level 0D temporal traces must be nonnegative")
    return trace / (trace.sum() + 1e-8)


def trace_centroid(trace: torch.Tensor) -> torch.Tensor:
    probability = normalized_trace(trace)
    time = torch.arange(trace.numel(), dtype=trace.dtype, device=trace.device)
    return (time * probability).sum()


def trace_w1_steps(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        torch.cumsum(normalized_trace(output), dim=0)
        - torch.cumsum(normalized_trace(target), dim=0)
    ).abs().sum()


def causal_filter(trace: torch.Tensor, tau_steps: float) -> torch.Tensor:
    decay = float(math.exp(-1.0 / float(tau_steps)))
    state = torch.zeros((), dtype=trace.dtype, device=trace.device)
    values = []
    for value in trace:
        state = decay * state + value
        values.append(state)
    return torch.stack(values)


def hard_filtered_loss(
    output_spikes: torch.Tensor, target_spikes: torch.Tensor, tau_steps: float
) -> torch.Tensor:
    return (
        causal_filter(output_spikes, tau_steps)
        - causal_filter(target_spikes, tau_steps)
    ).pow(2).mean()


def soft_centroid_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (trace_centroid(output) - trace_centroid(target)).pow(2)


def auxiliary_trace(
    rendered: dict[str, torch.Tensor], source: str
) -> torch.Tensor:
    if source == "synaptic_current":
        return rendered["current"]
    if source == "pre_reset_membrane":
        return rendered["pre_reset_membrane"]
    raise ValueError(source)


def _gradient(
    loss: torch.Tensor, parameter: torch.Tensor, *, retain_graph: bool
) -> float:
    value = torch.autograd.grad(
        loss, parameter, retain_graph=retain_graph, allow_unused=True
    )[0]
    return 0.0 if value is None else float(value.detach().item())


def optimize_bridge(
    protocol: dict[str, Any],
    *,
    condition_name: str,
    target_nominal_delay: float,
    initial_raw: float,
) -> dict[str, Any]:
    condition = condition_map(protocol)[condition_name]
    layer = make_delay_layer(protocol, initial_raw=initial_raw)
    optimizer = torch.optim.Adam(
        [layer.delay_raw], lr=float(protocol["delay"]["learning_rate"])
    )
    target = target_templates(protocol, target_nominal_delay)
    if float(target["spike"].sum().item()) != 1.0:
        raise RuntimeError("declared target must produce exactly one hard spike")
    source = str(condition["auxiliary_source"])
    hard_weight = float(condition["hard_weight"])
    auxiliary_weight = float(condition["auxiliary_weight"])
    filter_tau = float(protocol["losses"]["hard_filtered"]["filter_tau_steps"])
    optimizer_steps = int(protocol["delay"]["optimizer_steps"])
    tolerance = float(protocol["metrics"]["recovery_tolerance_steps"])
    required_spikes = float(protocol["metrics"]["required_final_hard_spike_count"])

    steps: list[int] = []
    raw_values: list[float] = []
    delay_values: list[float] = []
    total_losses: list[float] = []
    hard_losses: list[float] = []
    auxiliary_losses: list[float] = []
    total_gradients: list[float] = []
    hard_gradients: list[float] = []
    auxiliary_gradients: list[float] = []
    hard_w1_errors: list[float] = []
    hard_centroids: list[float] = []
    spike_counts: list[float] = []
    current_centroids: list[float] = []
    pre_reset_centroids: list[float] = []
    initial_rendered = final_rendered = None
    convergence_step = None

    for step in range(optimizer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        rendered = render_hard_path(
            layer,
            total_steps=int(protocol["timeline"]["total_steps"]),
            input_spike_step=int(protocol["timeline"]["input_spike_step"]),
            lif_config=protocol["lif"],
        )
        hard_loss = hard_filtered_loss(rendered["spike"], target["spike"], filter_tau)
        if source == "none":
            auxiliary_loss = hard_loss * 0.0
        else:
            auxiliary_loss = soft_centroid_loss(
                auxiliary_trace(rendered, source), auxiliary_trace(target, source)
            )
        total_loss = hard_weight * hard_loss + auxiliary_weight * auxiliary_loss
        hard_gradient = _gradient(hard_loss, layer.delay_raw, retain_graph=True)
        auxiliary_gradient = _gradient(
            auxiliary_loss, layer.delay_raw, retain_graph=True
        )
        total_loss.backward()

        total_gradient = float(layer.delay_raw.grad.detach().item())
        hard_w1 = float(trace_w1_steps(rendered["spike"], target["spike"]).detach().item())
        spike_count = float(rendered["spike"].detach().sum().item())
        hard_centroid = float(trace_centroid(rendered["spike"]).detach().item())
        current_centroid = float(trace_centroid(rendered["current"]).detach().item())
        pre_reset_centroid = float(
            trace_centroid(rendered["pre_reset_membrane"]).detach().item()
        )

        steps.append(step)
        raw_values.append(float(layer.delay_raw.detach().item()))
        delay_values.append(float(layer.get_delays().detach().item()))
        total_losses.append(float(total_loss.detach().item()))
        hard_losses.append(float(hard_loss.detach().item()))
        auxiliary_losses.append(float(auxiliary_loss.detach().item()))
        total_gradients.append(total_gradient)
        hard_gradients.append(hard_gradient)
        auxiliary_gradients.append(auxiliary_gradient)
        hard_w1_errors.append(hard_w1)
        hard_centroids.append(hard_centroid)
        spike_counts.append(spike_count)
        current_centroids.append(current_centroid)
        pre_reset_centroids.append(pre_reset_centroid)

        if step == 0:
            initial_rendered = {
                key: value.detach().cpu().numpy() for key, value in rendered.items()
            }
        if convergence_step is None and hard_w1 <= tolerance and spike_count == required_spikes:
            convergence_step = step
        if step < optimizer_steps:
            optimizer.step()
        else:
            final_rendered = {
                key: value.detach().cpu().numpy() for key, value in rendered.items()
            }

    assert initial_rendered is not None and final_rendered is not None
    final_recovered = (
        hard_w1_errors[-1] <= tolerance and spike_counts[-1] == required_spikes
    )
    return {
        "steps": np.asarray(steps, dtype=np.int64),
        "raw_values": np.asarray(raw_values, dtype=np.float64),
        "delay_values": np.asarray(delay_values, dtype=np.float64),
        "total_losses": np.asarray(total_losses, dtype=np.float64),
        "hard_losses": np.asarray(hard_losses, dtype=np.float64),
        "auxiliary_losses": np.asarray(auxiliary_losses, dtype=np.float64),
        "total_gradients": np.asarray(total_gradients, dtype=np.float64),
        "hard_gradients": np.asarray(hard_gradients, dtype=np.float64),
        "auxiliary_gradients": np.asarray(auxiliary_gradients, dtype=np.float64),
        "hard_w1_errors": np.asarray(hard_w1_errors, dtype=np.float64),
        "hard_centroids": np.asarray(hard_centroids, dtype=np.float64),
        "spike_counts": np.asarray(spike_counts, dtype=np.float64),
        "current_centroids": np.asarray(current_centroids, dtype=np.float64),
        "pre_reset_centroids": np.asarray(pre_reset_centroids, dtype=np.float64),
        "initial_rendered": initial_rendered,
        "final_rendered": final_rendered,
        "target": {key: value.detach().cpu().numpy() for key, value in target.items()},
        "target_hard_centroid": float(trace_centroid(target["spike"]).item()),
        "target_current_centroid": float(trace_centroid(target["current"]).item()),
        "target_pre_reset_centroid": float(
            trace_centroid(target["pre_reset_membrane"]).item()
        ),
        "convergence_step": convergence_step,
        "recovered": bool(final_recovered),
    }


def metrics_from_result(
    protocol: dict[str, Any],
    result: dict[str, Any],
    *,
    condition_name: str,
    target: float,
    initial_raw: float,
) -> dict[str, Any]:
    condition = condition_map(protocol)[condition_name]
    source = str(condition["auxiliary_source"])
    if source == "synaptic_current":
        initial_aux_centroid = float(result["current_centroids"][0])
        target_aux_centroid = float(result["target_current_centroid"])
    elif source == "pre_reset_membrane":
        initial_aux_centroid = float(result["pre_reset_centroids"][0])
        target_aux_centroid = float(result["target_pre_reset_centroid"])
    else:
        initial_aux_centroid = float(result["hard_centroids"][0])
        target_aux_centroid = float(result["target_hard_centroid"])
    initial_hard_gradient = float(result["hard_gradients"][0])
    initial_aux_gradient = float(result["auxiliary_gradients"][0])
    conflict = (
        source != "none"
        and abs(initial_hard_gradient) > 1e-12
        and abs(initial_aux_gradient) > 1e-12
        and initial_hard_gradient * initial_aux_gradient < 0.0
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "condition": condition_name,
        "condition_role": condition["role"],
        "hard_weight": float(condition["hard_weight"]),
        "auxiliary_source": source,
        "auxiliary_weight": float(condition["auxiliary_weight"]),
        "learning_rate": float(protocol["delay"]["learning_rate"]),
        "target_nominal_delay_steps": float(target),
        "target_arrival_step": target_arrival_step(protocol, target),
        "initial_raw": float(initial_raw),
        "optimizer_steps": int(protocol["delay"]["optimizer_steps"]),
        "direct_delay_supervision": False,
        "stochastic_seed": None,
        "test_split_opened": False,
        "initial_delay_steps": float(result["delay_values"][0]),
        "final_delay_steps": float(result["delay_values"][-1]),
        "final_nominal_delay_error_steps": abs(float(result["delay_values"][-1]) - float(target)),
        "initial_hard_spike_w1_steps": float(result["hard_w1_errors"][0]),
        "final_hard_spike_w1_steps": float(result["hard_w1_errors"][-1]),
        "initial_hard_spike_arrival": float(result["hard_centroids"][0]),
        "final_hard_spike_arrival": float(result["hard_centroids"][-1]),
        "target_hard_spike_arrival": float(result["target_hard_centroid"]),
        "initial_hard_spike_count": float(result["spike_counts"][0]),
        "final_hard_spike_count": float(result["spike_counts"][-1]),
        "minimum_hard_spike_count_during_training": float(result["spike_counts"].min()),
        "maximum_hard_spike_count_during_training": float(result["spike_counts"].max()),
        "initial_current_centroid": float(result["current_centroids"][0]),
        "final_current_centroid": float(result["current_centroids"][-1]),
        "target_current_centroid": float(result["target_current_centroid"]),
        "initial_pre_reset_centroid": float(result["pre_reset_centroids"][0]),
        "final_pre_reset_centroid": float(result["pre_reset_centroids"][-1]),
        "target_pre_reset_centroid": float(result["target_pre_reset_centroid"]),
        "direction_coordinate_initial": initial_aux_centroid,
        "direction_coordinate_target": target_aux_centroid,
        "initial_total_raw_gradient": float(result["total_gradients"][0]),
        "initial_hard_raw_gradient": initial_hard_gradient,
        "initial_auxiliary_raw_gradient": initial_aux_gradient,
        "initial_gradient_conflict": bool(conflict),
        "final_total_raw_gradient": float(result["total_gradients"][-1]),
        "final_total_loss": float(result["total_losses"][-1]),
        "final_hard_loss": float(result["hard_losses"][-1]),
        "final_auxiliary_loss": float(result["auxiliary_losses"][-1]),
        "convergence_step": result["convergence_step"],
        "recovered_exact_hard_arrival": bool(result["recovered"]),
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
    target_delay = float(metrics["target_nominal_delay_steps"])
    target_arrival = float(metrics["target_hard_spike_arrival"])
    hard_weight = float(metrics["hard_weight"])
    auxiliary_weight = float(metrics["auxiliary_weight"])
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Level 0D | {metrics['condition']} | target d={target_delay:g} | init raw={metrics['initial_raw']:g}"
    )

    axes[0, 0].plot(steps, result["delay_values"], color="#1f8a9b")
    axes[0, 0].axhline(target_delay, color="#c44e52", linestyle="--", label="target")
    axes[0, 0].set(title="Delay trajectory", xlabel="optimizer step", ylabel="delay (steps)")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(steps, result["hard_centroids"], color="#6a4c93")
    axes[0, 1].axhline(target_arrival, color="#c44e52", linestyle="--", label="target")
    axes[0, 1].set(title="Hard-spike arrival", xlabel="optimizer step", ylabel="arrival step")
    axes[0, 1].legend(frameon=False)

    axes[0, 2].semilogy(steps, np.maximum(result["hard_w1_errors"], 1e-10), color="#6a4c93")
    axes[0, 2].axhline(0.1, color="#c44e52", linestyle="--", label="0.1-step gate")
    axes[0, 2].set(title="Primary hard error", xlabel="optimizer step", ylabel="hard W1 (steps)")
    axes[0, 2].legend(frameon=False)

    axes[0, 3].plot(steps, result["spike_counts"], color="#264653")
    axes[0, 3].axhline(1.0, color="#c44e52", linestyle="--", label="required")
    axes[0, 3].set(title="Hard-spike count", xlabel="optimizer step", ylabel="count")
    axes[0, 3].legend(frameon=False)

    axes[1, 0].semilogy(steps, np.maximum(result["total_losses"], 1e-12), label="total", color="#386641")
    axes[1, 0].semilogy(steps, np.maximum(hard_weight * result["hard_losses"], 1e-12), label="weighted hard", color="#577590")
    axes[1, 0].semilogy(steps, np.maximum(auxiliary_weight * result["auxiliary_losses"], 1e-12), label="weighted aux", color="#e76f51")
    axes[1, 0].set(title="Loss components", xlabel="optimizer step", ylabel="loss")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(steps, result["total_gradients"], label="total", color="#386641")
    axes[1, 1].plot(steps, hard_weight * result["hard_gradients"], label="weighted hard", color="#577590", alpha=.8)
    axes[1, 1].plot(steps, auxiliary_weight * result["auxiliary_gradients"], label="weighted aux", color="#e76f51", alpha=.8)
    axes[1, 1].axhline(0.0, color="#777777", linewidth=.8)
    axes[1, 1].set(title="Delay-gradient components", xlabel="optimizer step", ylabel="dL/d(raw)")
    axes[1, 1].legend(frameon=False)

    time = np.arange(int(protocol["timeline"]["total_steps"]))
    axes[1, 2].step(time, result["target"]["spike"], where="mid", linestyle="--", color="#c44e52", label="target")
    axes[1, 2].step(time, result["initial_rendered"]["spike"], where="mid", color="#7aa6c2", label="initial")
    axes[1, 2].step(time, result["final_rendered"]["spike"], where="mid", color="#2a9d8f", label="final")
    axes[1, 2].set(title="Hard output trace", xlabel="simulation step", ylabel="spike")
    axes[1, 2].legend(frameon=False)

    source = str(metrics["auxiliary_source"])
    display_source = source if source != "none" else "pre_reset_membrane (diagnostic)"
    trace_key = {
        "none": "pre_reset_membrane",
        "synaptic_current": "current",
        "pre_reset_membrane": "pre_reset_membrane",
    }[source]
    axes[1, 3].plot(time, result["target"][trace_key], "--", color="#c44e52", label="target")
    axes[1, 3].plot(time, result["initial_rendered"][trace_key], color="#7aa6c2", label="initial")
    axes[1, 3].plot(time, result["final_rendered"][trace_key], color="#2a9d8f", label="final")
    axes[1, 3].set(title=f"Soft trace: {display_source}", xlabel="simulation step", ylabel="state")
    axes[1, 3].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.savefig(output, dpi=165, facecolor="white")
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
    condition_name: str,
    target: float,
    initial_raw: float,
) -> dict[str, Any]:
    directory = cell_directory(root, condition_name, target, initial_raw)
    metrics_path = directory / "metrics.json"
    if all(path.exists() for path in required_cell_artifacts(directory)):
        with metrics_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("complete"):
            return existing
    directory.mkdir(parents=True, exist_ok=True)
    condition = condition_map(protocol)[condition_name]
    cell_config = {
        "protocol_id": PROTOCOL_ID,
        "condition": condition_name,
        "condition_role": condition["role"],
        "hard_weight": float(condition["hard_weight"]),
        "auxiliary_source": condition["auxiliary_source"],
        "auxiliary_weight": float(condition["auxiliary_weight"]),
        "target_nominal_delay_steps": float(target),
        "initial_raw": float(initial_raw),
        "learning_rate": float(protocol["delay"]["learning_rate"]),
        "optimizer_steps": int(protocol["delay"]["optimizer_steps"]),
        "test_split_opened": False,
    }
    (directory / "config.json").write_text(
        json.dumps(cell_config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    result = optimize_bridge(
        protocol,
        condition_name=condition_name,
        target_nominal_delay=target,
        initial_raw=initial_raw,
    )
    metrics = metrics_from_result(
        protocol,
        result,
        condition_name=condition_name,
        target=target,
        initial_raw=initial_raw,
    )
    torch.save(
        {
            "delay_raw": float(result["raw_values"][-1]),
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
        total_losses=result["total_losses"],
        hard_losses=result["hard_losses"],
        auxiliary_losses=result["auxiliary_losses"],
        total_gradients=result["total_gradients"],
        hard_gradients=result["hard_gradients"],
        auxiliary_gradients=result["auxiliary_gradients"],
        hard_w1_errors=result["hard_w1_errors"],
        hard_centroids=result["hard_centroids"],
        spike_counts=result["spike_counts"],
        current_centroids=result["current_centroids"],
        pre_reset_centroids=result["pre_reset_centroids"],
        initial_hard_spikes=result["initial_rendered"]["spike"],
        final_hard_spikes=result["final_rendered"]["spike"],
        target_hard_spikes=result["target"]["spike"],
        initial_current=result["initial_rendered"]["current"],
        final_current=result["final_rendered"]["current"],
        target_current=result["target"]["current"],
        initial_pre_reset=result["initial_rendered"]["pre_reset_membrane"],
        final_pre_reset=result["final_rendered"]["pre_reset_membrane"],
        target_pre_reset=result["target"]["pre_reset_membrane"],
    )
    save_diagnostic_panel(protocol, result, metrics, plots / "diagnostic_panel.png")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metrics


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gradient_direction_stats(
    protocol: dict[str, Any], rows: list[dict[str, Any]], condition_name: str
) -> dict[str, int]:
    selected = [row for row in rows if row["condition"] == condition_name]
    tolerance = float(protocol["metrics"]["recovery_tolerance_steps"])
    zero_tolerance = float(protocol["metrics"]["initial_gradient_zero_tolerance"])
    correct = zero = wrong = aligned = 0
    for row in selected:
        if float(row["initial_hard_spike_w1_steps"]) <= tolerance:
            aligned += 1
            continue
        signed_error = float(row["direction_coordinate_initial"]) - float(
            row["direction_coordinate_target"]
        )
        gradient = float(row["initial_total_raw_gradient"])
        if abs(gradient) <= zero_tolerance:
            zero += 1
        elif gradient * signed_error > 0.0:
            correct += 1
        else:
            wrong += 1
    return {
        "directional_pairs": correct + zero + wrong,
        "correct_sign": correct,
        "zero_gradient": zero,
        "wrong_nonzero_sign": wrong,
        "already_hard_aligned": aligned,
    }


def _condition_heatmap(
    protocol: dict[str, Any], rows: list[dict[str, Any]], condition_name: str, output: Path
) -> None:
    targets = [float(value) for value in protocol["timeline"]["target_nominal_delays_steps"]]
    initializations = [float(value) for value in protocol["delay"]["initial_raw_values"]]
    matrix = np.zeros((len(initializations), len(targets)))
    selected = [row for row in rows if row["condition"] == condition_name]
    for i, initial_raw in enumerate(initializations):
        for j, target in enumerate(targets):
            row = next(
                item for item in selected
                if math.isclose(float(item["target_nominal_delay_steps"]), target)
                and math.isclose(float(item["initial_raw"]), initial_raw)
            )
            matrix[i, j] = float(row["final_hard_spike_w1_steps"])
    vmax = max(1.0, float(matrix.max()))
    fig, axis = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0.0, vmax=vmax, cmap="magma_r", aspect="auto")
    axis.set_xticks(range(len(targets)), [f"{value:g}" for value in targets])
    axis.set_yticks(range(len(initializations)), [f"{value:g}" for value in initializations])
    axis.set(
        title=f"Level 0D final hard W1 | {condition_name}",
        xlabel="target nominal delay",
        ylabel="initial raw",
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] / vmax > .58 else "black"
            axis.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color=color)
    fig.colorbar(image, ax=axis, label="hard spike W1 (steps)")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _short_condition(name: str) -> str:
    return (
        name.replace("hard_filtered_only", "hard only")
        .replace("current_soft_only", "current soft")
        .replace("pre_reset_soft_only", "pre-reset soft")
        .replace("hard_plus_current_lam_", "hard+current ")
        .replace("hard_plus_pre_reset_lam_", "hard+pre-reset ")
        .replace("0p01", ".01")
        .replace("0p1", ".1")
    )


def _summary_figures(
    protocol: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    names = [str(item["name"]) for item in protocol["conditions"]]
    x = np.arange(len(names))
    recovered = np.asarray([
        sum(bool(row["recovered_exact_hard_arrival"]) for row in rows if row["condition"] == name)
        for name in names
    ])
    fig, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    bars = axis.bar(x, recovered, color="#2a9d8f")
    axis.bar_label(bars)
    axis.axhline(15, color="#c44e52", linestyle="--", label="15/15 gate")
    axis.set_xticks(x, [_short_condition(name) for name in names], rotation=18, ha="right")
    axis.set(title="Level 0D hard-output recovery", ylabel="recovered target/init cells", ylim=(0, 16))
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=.2)
    fig.savefig(SUMMARY_ROOT / "condition_recovery_summary.png", dpi=180, facecolor="white")
    plt.close(fig)

    stats = [gradient_direction_stats(protocol, rows, name) for name in names]
    correct = np.asarray([item["correct_sign"] for item in stats])
    zero = np.asarray([item["zero_gradient"] for item in stats])
    wrong = np.asarray([item["wrong_nonzero_sign"] for item in stats])
    fig, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    bars_correct = axis.bar(x, correct, color="#2a9d8f", label="correct")
    bars_zero = axis.bar(x, zero, bottom=correct, color="#adb5bd", label="zero")
    bars_wrong = axis.bar(x, wrong, bottom=correct + zero, color="#bc4749", label="wrong")
    for bars_group in (bars_correct, bars_zero, bars_wrong):
        labels = [f"{value:g}" if value else "" for value in bars_group.datavalues]
        axis.bar_label(bars_group, labels=labels, label_type="center", fontsize=8)
    axis.set_xticks(x, [_short_condition(name) for name in names], rotation=18, ha="right")
    axis.set(title="Level 0D initial total-gradient direction", ylabel="initially hard-misaligned pairs", ylim=(0, 14))
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=.2)
    fig.savefig(SUMMARY_ROOT / "initial_gradient_direction_summary.png", dpi=180, facecolor="white")
    plt.close(fig)

    auxiliary_names = [name for name in names if condition_map(protocol)[name]["auxiliary_source"] != "none"]
    conflicts = [
        sum(bool(row["initial_gradient_conflict"]) for row in rows if row["condition"] == name)
        for name in auxiliary_names
    ]
    fig, axis = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    bars = axis.bar(np.arange(len(auxiliary_names)), conflicts, color="#e76f51")
    axis.bar_label(bars)
    axis.set_xticks(
        np.arange(len(auxiliary_names)),
        [_short_condition(name) for name in auxiliary_names],
        rotation=18,
        ha="right",
    )
    axis.set(title="Level 0D initial hard-versus-auxiliary gradient conflicts", ylabel="conflicting target/init cells", ylim=(0, 16))
    axis.grid(axis="y", alpha=.2)
    fig.savefig(SUMMARY_ROOT / "initial_gradient_conflict_summary.png", dpi=180, facecolor="white")
    plt.close(fig)


def aggregate(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (
        row["condition"], float(row["target_nominal_delay_steps"]), float(row["initial_raw"])
    ))
    _write_csv(rows, SUMMARY_ROOT / "cells.csv")
    condition_results = []
    for condition in protocol["conditions"]:
        name = str(condition["name"])
        selected = [row for row in rows if row["condition"] == name]
        directions = gradient_direction_stats(protocol, rows, name)
        recovered = sum(bool(row["recovered_exact_hard_arrival"]) for row in selected)
        direction_pass = (
            directions["correct_sign"] == directions["directional_pairs"]
            and directions["zero_gradient"] == 0
            and directions["wrong_nonzero_sign"] == 0
        )
        condition_results.append({
            "condition": name,
            "role": condition["role"],
            "hard_weight": float(condition["hard_weight"]),
            "auxiliary_source": condition["auxiliary_source"],
            "auxiliary_weight": float(condition["auxiliary_weight"]),
            "recovered_cells": recovered,
            "total_cells": 15,
            "condition_coverage_pass": recovered == 15,
            "initial_gradient_direction": directions,
            "direction_pass": direction_pass,
            "initial_gradient_conflict_cells": sum(
                bool(row["initial_gradient_conflict"]) for row in selected
            ),
            "compatibility_pass": bool(
                condition["role"] == "compatibility_candidate"
                and recovered == 15
                and direction_pass
            ),
        })
        _condition_heatmap(
            protocol, rows, name, SUMMARY_ROOT / f"final_hard_w1_{name}.png"
        )
    _summary_figures(protocol, rows)

    passing = [item for item in condition_results if item["compatibility_pass"]]
    selected_candidate = None
    for source in protocol["decision_rules"]["selection_priority"]["auxiliary_sources"]:
        source_passes = sorted(
            [item for item in passing if item["auxiliary_source"] == source],
            key=lambda item: float(item["auxiliary_weight"]),
        )
        if source_passes:
            selected_candidate = source_passes[0]
            break
    level0d_pass = selected_candidate is not None
    decision = {
        "protocol_id": PROTOCOL_ID,
        "expected_cells": expected_cells(protocol),
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == expected_cells(protocol) and all(bool(row["complete"]) for row in rows),
        "condition_results": condition_results,
        "selected_compatibility_candidate": selected_candidate,
        "level0d_pass": level0d_pass,
        "level1a_k1_xor_preregistration_authorized": level0d_pass,
        "level1_xor_result_claim_authorized": False,
        "pairwise_wad_or_scaling_authorized": False,
    }
    (SUMMARY_ROOT / "decision.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return decision


def run_formal(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for condition in protocol["conditions"]:
        for target in [float(value) for value in protocol["timeline"]["target_nominal_delays_steps"]]:
            for initial_raw in [float(value) for value in protocol["delay"]["initial_raw_values"]]:
                rows.append(run_cell(
                    protocol,
                    root=RUN_ROOT,
                    condition_name=str(condition["name"]),
                    target=target,
                    initial_raw=initial_raw,
                ))
    return aggregate(protocol, rows)


def run_smoke(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for condition_name in (
        "hard_filtered_only",
        "current_soft_only",
        "pre_reset_soft_only",
        "hard_plus_current_lam_0p1",
        "hard_plus_pre_reset_lam_0p1",
    ):
        for target, initial_raw in ((5.0, -2.0), (1.0, 0.0)):
            rows.append(run_cell(
                protocol,
                root=SMOKE_ROOT,
                condition_name=condition_name,
                target=target,
                initial_raw=initial_raw,
            ))
    return {
        "smoke_cells": len(rows),
        "recovered": sum(bool(row["recovered_exact_hard_arrival"]) for row in rows),
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
        raise SystemExit("declared Level-0D cell count does not match the grid")
    result = run_smoke(protocol) if args.smoke else run_formal(protocol)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

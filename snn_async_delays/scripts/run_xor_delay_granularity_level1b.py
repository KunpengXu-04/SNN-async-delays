"""Run the preregistered K=1 XOR delay-granularity study (Level 1B v1)."""

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
import torch
import yaml

from scripts.run_xor_task_bridge_level1a import (
    _strict_write_json,
    _truth_records,
    exact_interface_metrics,
    exact_truth_batch,
    per_cell_interface_pass,
)
from snn.model import SNNSimultaneousModel
from train.trainer import filtered_opponent_spike_train_loss, opponent_target_spike_train
from utils.resource_ledger import dynamic_resource_ledger, static_resource_ledger
from utils.seed import set_seed


BASE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "xor_delay_granularity_level1b_v1"
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


def _granularities(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return list(protocol["model"]["delay_granularities"])


def expected_stage_a_cells(protocol: dict[str, Any]) -> int:
    return (
        len(_granularities(protocol))
        * len(protocol["losses"]["per_parameter_arrival_centroid"]["conditions"])
        * len(protocol["optimization"]["learned_delay_initial_raw_values"])
        * len(protocol["optimization"]["heldout_initialization_seeds"])
    )


def expected_stage_b_control_cells(protocol: dict[str, Any]) -> int:
    return (
        len(protocol["stage_b_microburst_controls"]["fixed_controls"])
        * len(protocol["optimization"]["heldout_initialization_seeds"])
    )


def expected_stage_b_learned_cells(protocol: dict[str, Any]) -> int:
    return expected_stage_a_cells(protocol)


def _learned_specs(protocol: dict[str, Any], *, stage: str, encoding: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for granularity in _granularities(protocol):
        for condition in protocol["losses"]["per_parameter_arrival_centroid"]["conditions"]:
            for initial_raw in protocol["optimization"]["learned_delay_initial_raw_values"]:
                for seed in protocol["optimization"]["heldout_initialization_seeds"]:
                    specs.append({
                        "stage": stage,
                        "condition": "learned_delay",
                        "encoding": encoding,
                        "granularity": str(granularity["name"]),
                        "delay_tying": str(granularity["tying"]),
                        "independent_delay_parameters": int(granularity["independent_delay_parameters"]),
                        "seed": int(seed),
                        "learned_delay": True,
                        "fixed_delay_steps": None,
                        "target_delay_steps": float(protocol["timing"]["target_delay_steps"]),
                        "arrival_condition": str(condition["name"]),
                        "arrival_auxiliary_weight": float(condition["weight"]),
                        "initial_raw": float(initial_raw),
                    })
    return specs


def stage_a_specs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return _learned_specs(protocol, stage="stage_a", encoding="single_event")


def stage_b_control_specs(
    protocol: dict[str, Any], stage_a_decision: dict[str, Any]
) -> list[dict[str, Any]]:
    if not bool(stage_a_decision.get("stage_b_controls_authorized")):
        raise RuntimeError("Stage-B controls are locked because the global Stage-A replication failed")
    specs: list[dict[str, Any]] = []
    for control in protocol["stage_b_microburst_controls"]["fixed_controls"]:
        for seed in protocol["optimization"]["heldout_initialization_seeds"]:
            specs.append({
                "stage": "stage_b_controls",
                "condition": str(control["name"]),
                "encoding": "consecutive_microburst",
                "granularity": "fixed_pair_storage",
                "delay_tying": "pair",
                "independent_delay_parameters": 64,
                "seed": int(seed),
                "learned_delay": False,
                "fixed_delay_steps": float(control["fixed_input_hidden_delay_steps"]),
                "target_delay_steps": float(control["target_delay_steps"]),
                "arrival_condition": "none",
                "arrival_auxiliary_weight": 0.0,
                "initial_raw": None,
            })
    return specs


def stage_b_learned_specs(
    protocol: dict[str, Any], control_decision: dict[str, Any]
) -> list[dict[str, Any]]:
    if not bool(control_decision.get("stage_b_learned_authorized")):
        raise RuntimeError("learned micro-burst cells are locked because fixed delay 4 failed")
    return _learned_specs(
        protocol, stage="stage_b_learned", encoding="consecutive_microburst"
    )


def _token(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def cell_directory(root: Path, spec: dict[str, Any]) -> Path:
    base = root / str(spec["stage"]) / str(spec["condition"])
    if not bool(spec["learned_delay"]):
        return base / f"seed_{spec['seed']}"
    return (
        base
        / str(spec["granularity"])
        / f"lambda_{_token(spec['arrival_auxiliary_weight'])}"
        / f"init_{_token(spec['initial_raw'])}"
        / f"seed_{spec['seed']}"
    )


def encode_exact_truth(
    protocol: dict[str, Any], A: torch.Tensor, B: torch.Tensor, *, encoding: str, device: str
) -> torch.Tensor:
    total_steps = int(protocol["timing"]["total_steps"])
    events = [int(value) for value in protocol["encodings"][encoding]["event_steps"]]
    spikes = torch.zeros((A.shape[0], total_steps, 4), dtype=torch.float32, device=device)
    for index in range(A.shape[0]):
        a_channel = int(A[index].reshape(-1)[0].item())
        b_channel = 2 + int(B[index].reshape(-1)[0].item())
        for step in events:
            spikes[index, step, a_channel] = 1.0
            spikes[index, step, b_channel] = 1.0
    return spikes


def target_output_step(protocol: dict[str, Any]) -> int:
    return int(protocol["timing"]["output_target_step"])


def build_model(protocol: dict[str, Any], spec: dict[str, Any]) -> SNNSimultaneousModel:
    model_cfg = protocol["model"]
    hidden_cfg = model_cfg["hidden_lif"]
    output_cfg = model_cfg["output_lif"]
    learned = bool(spec["learned_delay"])
    return SNNSimultaneousModel(
        n_queries=1,
        n_hidden=int(model_cfg["hidden_neurons"]),
        win_len=int(protocol["timing"]["input_window_steps"]),
        read_len=int(protocol["timing"]["read_window_steps"]),
        d_max=int(model_cfg["d_max_steps"]),
        train_mode="weights_and_delays" if learned else "weights_only",
        delay_param_type=str(model_cfg["input_hidden_delay_parameterization"]),
        fixed_delay_value=None if learned else float(spec["fixed_delay_steps"]),
        shared_delay=False,
        delay_tying=str(spec["delay_tying"]) if learned else None,
        delay_init_mode="constant",
        delay_init_raw=float(spec["initial_raw"] if learned else -2.0),
        lif_tau_m=float(hidden_cfg["tau_m_steps"]),
        lif_threshold=float(hidden_cfg["threshold_au"]),
        lif_reset=float(hidden_cfg["reset_au"]),
        lif_refractory=int(hidden_cfg["refractory_steps"]),
        dt=float(model_cfg["dt_steps"]),
        surrogate_beta=float(model_cfg["surrogate_beta"]),
        n_input_channels=4,
        readout_type="linear",
        use_output_spikes=True,
        n_output_neurons=2,
        lif_output_threshold=float(output_cfg["threshold_au"]),
        observation_mode="all_time",
        opponent_output_mode="parallel_pairs",
        output_window_len=int(protocol["timing"]["read_window_steps"]),
        output_delay_mode="d0",
    )


def build_optimizer(
    model: SNNSimultaneousModel, protocol: dict[str, Any], spec: dict[str, Any] | None = None
) -> torch.optim.Optimizer:
    spec = spec or {}
    groups: list[dict[str, Any]] = []
    weights = model.weight_params()
    delays = model.delay_params()
    if weights:
        groups.append({"params": weights, "lr": float(spec.get("weight_learning_rate", protocol["optimization"]["weight_learning_rate"]))})
    if delays:
        groups.append({"params": delays, "lr": float(spec.get("delay_learning_rate", protocol["optimization"]["delay_learning_rate"]))})
    return torch.optim.Adam(groups)


def independent_delay_values(
    model: SNNSimultaneousModel, spec: dict[str, Any]
) -> torch.Tensor:
    if not bool(spec["learned_delay"]):
        return torch.tensor(
            [float(spec["fixed_delay_steps"])],
            dtype=model.syn_ih.weight.dtype,
            device=model.syn_ih.weight.device,
        )
    raw = model.syn_ih.delay_raw
    if model.syn_ih.delay_param_type != "sigmoid":
        raise ValueError("Level 1B preregisters sigmoid delay parameters only")
    return (float(model.syn_ih.d_max) * torch.sigmoid(raw)).reshape(-1)


def per_coordinate_base_traces(
    spike_input: torch.Tensor, *, tying: str, hidden_neurons: int
) -> torch.Tensor:
    """Unweighted presynaptic event mass for each independent delay coordinate."""
    per_input = spike_input.sum(dim=0).transpose(0, 1)  # [input,time]
    if tying == "global":
        return per_input.sum(dim=0, keepdim=True)
    if tying == "post_neuron":
        return per_input.sum(dim=0, keepdim=True).expand(hidden_neurons, -1)
    if tying == "pair":
        return (
            per_input[:, None, :]
            .expand(per_input.shape[0], hidden_neurons, per_input.shape[1])
            .reshape(-1, per_input.shape[1])
        )
    raise ValueError(f"unsupported delay tying: {tying}")


def shifted_coordinate_traces(
    base_traces: torch.Tensor, delays: torch.Tensor, *, d_max: int
) -> torch.Tensor:
    """Apply the production d+1 delay interpolation to P independent traces."""
    if base_traces.dim() != 2 or delays.numel() != base_traces.shape[0]:
        raise ValueError("base traces and delays must agree on coordinate count")
    delays = torch.clamp(delays.reshape(-1), 0.0, float(d_max))
    floor = torch.floor(delays.detach()).long()
    ceil = torch.clamp(floor + 1, max=int(d_max))
    alpha = delays - floor.to(delays.dtype)
    time = torch.arange(base_traces.shape[1], device=base_traces.device)
    source_floor = time[None, :] - 1 - floor[:, None]
    source_ceil = time[None, :] - 1 - ceil[:, None]
    floor_valid = source_floor >= 0
    ceil_valid = source_ceil >= 0
    floor_values = torch.gather(base_traces, 1, source_floor.clamp(min=0))
    ceil_values = torch.gather(base_traces, 1, source_ceil.clamp(min=0))
    floor_values = floor_values * floor_valid.to(base_traces.dtype)
    ceil_values = ceil_values * ceil_valid.to(base_traces.dtype)
    return (1.0 - alpha[:, None]) * floor_values + alpha[:, None] * ceil_values


def trace_centroids(traces: torch.Tensor) -> torch.Tensor:
    time = torch.arange(traces.shape[1], dtype=traces.dtype, device=traces.device)
    return (traces * time[None, :]).sum(dim=1) / (traces.sum(dim=1) + 1e-8)


def per_parameter_arrival_loss(
    base_traces: torch.Tensor,
    delays: torch.Tensor,
    *,
    target_delay_steps: float,
    d_max: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    arrivals = shifted_coordinate_traces(base_traces, delays, d_max=d_max)
    target_delays = torch.full_like(delays, float(target_delay_steps))
    target_arrivals = shifted_coordinate_traces(base_traces, target_delays, d_max=d_max).detach()
    centroids = trace_centroids(arrivals)
    targets = trace_centroids(target_arrivals)
    loss = 0.5 * (centroids - targets).pow(2).mean()
    return loss, centroids, targets, arrivals, target_arrivals


def _parameter_gradient(
    loss: torch.Tensor, parameter: torch.Tensor | None, *, retain_graph: bool
) -> torch.Tensor | None:
    if parameter is None:
        return None
    gradient = torch.autograd.grad(
        loss, parameter, retain_graph=retain_graph, allow_unused=True
    )[0]
    if gradient is None:
        return torch.zeros_like(parameter)
    return gradient.detach().reshape(-1)


def _coordinate_gradient_stats(
    gradient: torch.Tensor | None,
    delays: torch.Tensor,
    *,
    target: float,
    epsilon: float = 1e-10,
) -> dict[str, float]:
    if gradient is None:
        return {"norm": 0.0, "direction_fraction": 0.0, "nonzero_fraction": 0.0}
    values = gradient.detach().reshape(-1)
    delay_values = delays.detach().reshape(-1)
    nonzero = values.abs() > epsilon
    correct = values * (delay_values - float(target)) > 0.0
    return {
        "norm": float(values.norm().item()),
        "direction_fraction": float(correct.float().mean().item()),
        "nonzero_fraction": float(nonzero.float().mean().item()),
    }


def _weight_grad_norm(parameter: torch.Tensor) -> float:
    return 0.0 if parameter.grad is None else float(parameter.grad.detach().norm().item())


def train_cell(
    protocol: dict[str, Any], spec: dict[str, Any], *, device: str
) -> tuple[SNNSimultaneousModel, dict[str, Any]]:
    set_seed(int(spec["seed"]))
    model = build_model(protocol, spec).to(device)
    optimizer = build_optimizer(model, protocol, spec)
    A, B, _, labels = exact_truth_batch(device)
    spike_input = encode_exact_truth(protocol, A, B, encoding=str(spec["encoding"]), device=device)
    target_spikes = opponent_target_spike_train(
        labels,
        total_steps=int(protocol["timing"]["total_steps"]),
        input_steps=int(protocol["timing"]["input_window_steps"]),
        output_window_len=int(protocol["timing"]["read_window_steps"]),
        timing_mode="simultaneous_offset",
        target_offset_steps=target_output_step(protocol) - int(protocol["timing"]["input_window_steps"]),
    )
    learned = bool(spec["learned_delay"])
    delay_parameter = model.syn_ih.delay_raw if learned else None
    tying = str(spec["delay_tying"]) if learned else "global"
    base_traces = per_coordinate_base_traces(
        spike_input, tying=tying, hidden_neurons=int(protocol["model"]["hidden_neurons"])
    )
    lam = float(spec["arrival_auxiliary_weight"])
    target_delay = float(spec["target_delay_steps"])
    updates = int(spec.get("full_batch_updates", protocol["optimization"]["full_batch_updates"]))
    d_max = int(protocol["model"]["d_max_steps"])
    tolerance = float(protocol["per_cell_gates"]["learned_delay"]["max_independent_delay_error_steps"])

    keys = (
        "step", "total_loss", "task_loss", "arrival_loss",
        "delay_mean", "delay_min", "delay_max", "delay_std", "delay_fraction_within",
        "arrival_centroid_mean", "arrival_centroid_min", "arrival_centroid_max",
        "task_gradient_norm", "arrival_gradient_norm", "total_gradient_norm",
        "total_direction_fraction", "total_nonzero_fraction",
        "ih_weight_gradient_norm", "ho_weight_gradient_norm", "global_preclip_gradient_norm",
        "clip_coefficient", "balanced_accuracy",
        "exact_interface_patterns", "hidden_active_patterns", "silent_rate", "collision_rate",
    )
    history: dict[str, list[float]] = {key: [] for key in keys}
    initial_arrival = target_arrival = None
    initial_delays = None
    initial_centroids = target_centroids = None
    initial_gradients: dict[str, np.ndarray | None] = {}

    model.train()
    for step in range(updates + 1):
        optimizer.zero_grad(set_to_none=True)
        _, info = model(spike_input, return_output_spike_train=True)
        task_loss = filtered_opponent_spike_train_loss(
            info["output_spike_train"], target_spikes, labels,
            tau_steps=float(protocol["losses"]["filtered_hard_spike"]["filter_tau_steps"]),
        )
        delays = independent_delay_values(model, spec)
        arrival_loss, centroids, centroid_targets, arrivals, target_arrivals = per_parameter_arrival_loss(
            base_traces, delays, target_delay_steps=target_delay, d_max=d_max
        )
        total_loss = task_loss + lam * arrival_loss
        task_gradient = _parameter_gradient(task_loss, delay_parameter, retain_graph=True)
        arrival_gradient = _parameter_gradient(arrival_loss, delay_parameter, retain_graph=True)
        total_loss.backward()
        total_gradient = None if delay_parameter is None else delay_parameter.grad.detach().reshape(-1).clone()
        gradient_norms = [
            parameter.grad.detach().norm()
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        global_preclip_norm = float(
            torch.stack(gradient_norms).norm().item() if gradient_norms else 0.0
        )
        clip_limit = float(protocol["optimization"]["gradient_clip_norm"])
        clip_coefficient = min(1.0, clip_limit / (global_preclip_norm + 1e-12))
        interface = exact_interface_metrics(
            info["output_spike_train"], target_spikes, labels, info["total_hidden_spikes"]
        )
        total_stats = _coordinate_gradient_stats(total_gradient, delays, target=target_delay)
        errors = (delays.detach() - target_delay).abs()
        values = {
            "step": float(step),
            "total_loss": float(total_loss.detach().item()),
            "task_loss": float(task_loss.detach().item()),
            "arrival_loss": float(arrival_loss.detach().item()),
            "delay_mean": float(delays.detach().mean().item()),
            "delay_min": float(delays.detach().min().item()),
            "delay_max": float(delays.detach().max().item()),
            "delay_std": float(delays.detach().std(unbiased=False).item()),
            "delay_fraction_within": float((errors <= tolerance).float().mean().item()),
            "arrival_centroid_mean": float(centroids.detach().mean().item()),
            "arrival_centroid_min": float(centroids.detach().min().item()),
            "arrival_centroid_max": float(centroids.detach().max().item()),
            "task_gradient_norm": 0.0 if task_gradient is None else float(task_gradient.norm().item()),
            "arrival_gradient_norm": 0.0 if arrival_gradient is None else float(arrival_gradient.norm().item()),
            "total_gradient_norm": total_stats["norm"],
            "total_direction_fraction": total_stats["direction_fraction"],
            "total_nonzero_fraction": total_stats["nonzero_fraction"],
            "ih_weight_gradient_norm": _weight_grad_norm(model.syn_ih.weight),
            "ho_weight_gradient_norm": _weight_grad_norm(model.syn_ho.weight),
            "global_preclip_gradient_norm": global_preclip_norm,
            "clip_coefficient": clip_coefficient,
            "balanced_accuracy": float(interface["balanced_accuracy"]),
            "exact_interface_patterns": float(interface["exact_target_spike_train_matches"]),
            "hidden_active_patterns": float(interface["hidden_active_pattern_count"]),
            "silent_rate": float(interface["silent_rate"]),
            "collision_rate": float(interface["collision_rate"]),
        }
        for key, value in values.items():
            history[key].append(value)
        if step == 0:
            initial_arrival = arrivals.detach().cpu().numpy()
            target_arrival = target_arrivals.detach().cpu().numpy()
            initial_delays = delays.detach().cpu().numpy()
            initial_centroids = centroids.detach().cpu().numpy()
            target_centroids = centroid_targets.detach().cpu().numpy()
            initial_gradients = {
                "task": None if task_gradient is None else task_gradient.cpu().numpy(),
                "arrival": None if arrival_gradient is None else arrival_gradient.cpu().numpy(),
                "total": None if total_gradient is None else total_gradient.cpu().numpy(),
            }
        if step < updates:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=clip_limit,
            )
            optimizer.step()

    model.eval()
    with torch.no_grad():
        _, final_info = model(spike_input, record=True, return_output_spike_train=True)
        final_delays = independent_delay_values(model, spec)
        _, final_centroids, _, final_arrival, _ = per_parameter_arrival_loss(
            base_traces, final_delays, target_delay_steps=target_delay, d_max=d_max
        )
    assert initial_arrival is not None and target_arrival is not None
    assert initial_delays is not None and initial_centroids is not None and target_centroids is not None
    final_interface = exact_interface_metrics(
        final_info["output_spike_train"], target_spikes, labels, final_info["total_hidden_spikes"]
    )
    interface_pass = per_cell_interface_pass(final_interface)
    final_errors = (final_delays.detach() - target_delay).abs()
    initial_total = initial_gradients.get("total")
    if learned and initial_total is not None:
        grad_tensor = torch.from_numpy(initial_total)
        initial_stats = _coordinate_gradient_stats(
            grad_tensor, torch.from_numpy(initial_delays), target=target_delay
        )
    else:
        initial_stats = {"norm": 0.0, "direction_fraction": 0.0, "nonzero_fraction": 0.0}
    task_gradient = initial_gradients.get("task")
    arrival_gradient = initial_gradients.get("arrival")
    if learned and task_gradient is not None and arrival_gradient is not None:
        task_array = np.asarray(task_gradient)
        arrival_array = np.asarray(arrival_gradient)
        valid = (np.abs(task_array) > 1e-10) & (np.abs(arrival_array) > 1e-10)
        conflict_fraction = float(np.mean((task_array * arrival_array < 0.0)[valid])) if valid.any() else 0.0
    else:
        conflict_fraction = 0.0
    learned_pass = bool(
        learned
        and interface_pass
        and float(final_errors.max().item()) <= tolerance
        and float((final_errors <= tolerance).float().mean().item()) == 1.0
        and initial_stats["direction_fraction"] == 1.0
        and initial_stats["nonzero_fraction"] == 1.0
    )
    record = {
        "input": spike_input.detach().cpu().numpy(),
        "hidden": final_info["hidden_spike_train"].detach().cpu().numpy(),
        "output": final_info["output_spike_train"].detach().cpu().numpy(),
        "output_pre_reset": final_info["output_membrane_train"].detach().cpu().numpy(),
        "output_current": final_info["output_synaptic_current_train"].detach().cpu().numpy(),
        "target_output": target_spikes.detach().cpu().numpy(),
        "initial_arrival": initial_arrival,
        "final_arrival": final_arrival.detach().cpu().numpy(),
        "target_arrival": target_arrival,
        "initial_arrival_centroids": initial_centroids,
        "final_arrival_centroids": final_centroids.detach().cpu().numpy(),
        "target_arrival_centroids": target_centroids,
        "initial_independent_delays": initial_delays,
        "final_independent_delays": final_delays.detach().cpu().numpy(),
        "labels": labels.detach().cpu().numpy(),
        "A": A.detach().cpu().numpy(),
        "B": B.detach().cpu().numpy(),
        "final_ih_weight": model.syn_ih.weight.detach().cpu().numpy(),
        "final_ho_weight": model.syn_ho.weight.detach().cpu().numpy(),
    }
    return model, {
        "history": {key: np.asarray(values, dtype=np.float64) for key, values in history.items()},
        "final_interface": final_interface,
        "initial_gradients": initial_gradients,
        "initial_gradient_stats": initial_stats,
        "initial_task_arrival_gradient_conflict_fraction": conflict_fraction,
        "interface_pass": interface_pass,
        "learned_delay_pass": learned_pass,
        "final_delay_max_error_steps": float(final_errors.max().item()),
        "final_delay_fraction_within_tolerance": float((final_errors <= tolerance).float().mean().item()),
        "final_record": record,
    }


def resource_ledger(
    model: SNNSimultaneousModel, spec: dict[str, Any], record: dict[str, np.ndarray]
) -> dict[str, Any]:
    ledger = static_resource_ledger(model)
    ledger.update(dynamic_resource_ledger(
        model,
        mean_input_spikes=float(record["input"].sum(axis=(1, 2)).mean()),
        mean_hidden1_spikes=float(record["hidden"].sum(axis=(1, 2)).mean()),
        mean_output_spikes=float(record["output"].sum(axis=(1, 2)).mean()),
    ))
    ledger.update({
        "delay_granularity": str(spec["granularity"]),
        "input_hidden_physical_synapses": int(model.n_input * model.n_hidden),
        "input_hidden_independent_delay_values": int(model.syn_ih.delay_raw.numel()),
        "input_hidden_trainable_delay_parameters": int(
            model.syn_ih.delay_raw.numel() if model.syn_ih.delay_raw.requires_grad else 0
        ),
        "delay_tying": str(model.syn_ih.delay_tying),
    })
    return ledger


def _delay_map(values: np.ndarray, granularity: str, hidden_neurons: int) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if granularity == "global":
        return flat.reshape(1, 1)
    if granularity == "per_hidden_neuron":
        return flat.reshape(1, hidden_neurons)
    if granularity == "per_synapse":
        return flat.reshape(4, hidden_neurons)
    return flat.reshape(1, -1)


def save_diagnostic_panel(
    protocol: dict[str, Any], spec: dict[str, Any], result: dict[str, Any], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    history = result["history"]
    record = result["final_record"]
    steps = history["step"]
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), constrained_layout=True)
    panel_label = str(protocol.get("panel_label", "Level 1B"))
    scale_label = (
        f" | scale={float(spec['normalization_factor']):g}"
        if "normalization_factor" in spec else ""
    )
    fig.suptitle(
        f"{panel_label} {spec['stage']} | {spec['granularity']} | "
        f"lambda={spec['arrival_auxiliary_weight']:g}{scale_label} | seed {spec['seed']}"
    )
    axes[0, 0].semilogy(steps, np.maximum(history["total_loss"], 1e-12), label="total")
    axes[0, 0].semilogy(steps, np.maximum(history["task_loss"], 1e-12), label="hard spike task")
    axes[0, 0].semilogy(
        steps,
        np.maximum(float(spec["arrival_auxiliary_weight"]) * history["arrival_loss"], 1e-12),
        label="weighted arrival",
    )
    axes[0, 0].set(title="Loss components", xlabel="update", ylabel="loss")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(steps, history["exact_interface_patterns"], label="exact spike trains")
    axes[0, 1].plot(steps, history["hidden_active_patterns"], label="hidden-active")
    axes[0, 1].axhline(4, linestyle="--", color="tab:red", label="4/4 gate")
    axes[0, 1].set(title="Exact interface", xlabel="update", ylabel="patterns", ylim=(-0.1, 4.3))
    axes[0, 1].legend(frameon=False)
    axes[0, 2].plot(steps, history["delay_mean"], label="mean")
    axes[0, 2].fill_between(steps, history["delay_min"], history["delay_max"], alpha=.25, label="min-max")
    axes[0, 2].axhline(float(spec["target_delay_steps"]), linestyle="--", color="tab:red", label="target")
    axes[0, 2].set(title="Independent delays", xlabel="update", ylabel="steps")
    axes[0, 2].legend(frameon=False)
    axes[0, 3].plot(steps, history["delay_std"], label="std")
    axes[0, 3].plot(steps, history["delay_fraction_within"], label="fraction within 0.1")
    axes[0, 3].set(title="Delay coverage", xlabel="update", ylabel="value")
    axes[0, 3].legend(frameon=False)
    axes[1, 0].semilogy(steps, np.maximum(history["task_gradient_norm"], 1e-12), label="task")
    axes[1, 0].semilogy(
        steps,
        np.maximum(float(spec["arrival_auxiliary_weight"]) * history["arrival_gradient_norm"], 1e-12),
        label="weighted arrival",
    )
    axes[1, 0].semilogy(steps, np.maximum(history["total_gradient_norm"], 1e-12), label="total")
    axes[1, 0].set(title="Delay-gradient norms", xlabel="update", ylabel="L2 norm")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(steps, history["total_direction_fraction"], label="correct direction")
    axes[1, 1].plot(steps, history["total_nonzero_fraction"], label="nonzero")
    if "clip_coefficient" in history:
        axes[1, 1].plot(steps, history["clip_coefficient"], ":", label="clip coefficient")
    axes[1, 1].axhline(1, linestyle="--", color="tab:red")
    axes[1, 1].set(title="Coordinate gradient coverage", xlabel="update", ylabel="fraction", ylim=(-.05, 1.05))
    axes[1, 1].legend(frameon=False)
    input_rows = record["input"].transpose(0, 2, 1).reshape(-1, record["input"].shape[1])
    axes[1, 2].imshow(input_rows, aspect="auto", interpolation="nearest", cmap="Greys")
    axes[1, 2].set(title="Truth-table input raster", xlabel="time", ylabel="pattern x input")
    hidden_rows = record["hidden"].transpose(0, 2, 1).reshape(-1, record["hidden"].shape[1])
    axes[1, 3].imshow(hidden_rows, aspect="auto", interpolation="nearest", cmap="Greys")
    axes[1, 3].set(title="Truth-table hidden raster", xlabel="time", ylabel="pattern x hidden")
    output_rows = record["output"].transpose(0, 2, 1).reshape(-1, record["output"].shape[1])
    target_rows = record["target_output"].transpose(0, 2, 1).reshape(-1, record["target_output"].shape[1])
    axes[2, 0].imshow(output_rows + 2 * target_rows, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=3)
    axes[2, 0].set(title="Output raster (target=2, actual=1)", xlabel="time", ylabel="pattern x output")
    voltage_rows = record["output_pre_reset"].transpose(0, 2, 1).reshape(-1, record["output_pre_reset"].shape[1])
    voltage_image = axes[2, 1].imshow(voltage_rows, aspect="auto", interpolation="nearest", cmap="coolwarm")
    axes[2, 1].set(title="Output pre-reset voltage", xlabel="time", ylabel="pattern x output")
    fig.colorbar(voltage_image, ax=axes[2, 1], fraction=.046)
    delay_image = axes[2, 2].imshow(
        _delay_map(record["final_independent_delays"], str(spec["granularity"]), int(protocol["model"]["hidden_neurons"])),
        aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=float(protocol["model"]["d_max_steps"]),
    )
    axes[2, 2].set(title="Final independent delay map", xlabel="hidden coordinate", ylabel="input coordinate")
    fig.colorbar(delay_image, ax=axes[2, 2], fraction=.046, label="steps")
    coordinates = np.arange(record["final_arrival_centroids"].size)
    axes[2, 3].plot(coordinates, record["initial_arrival_centroids"], ".", label="initial")
    axes[2, 3].plot(coordinates, record["final_arrival_centroids"], ".", label="final")
    axes[2, 3].plot(coordinates, record["target_arrival_centroids"], "_", label="target")
    axes[2, 3].set(title="Arrival centroids by coordinate", xlabel="independent coordinate", ylabel="time")
    axes[2, 3].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=.15)
    fig.savefig(output, dpi=165, facecolor="white")
    plt.close(fig)


def run_cell(
    protocol: dict[str, Any], spec: dict[str, Any], *, root: Path, device: str
) -> dict[str, Any]:
    directory = cell_directory(root, spec)
    metrics_path = directory / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    encoding_cfg = protocol["encodings"][str(spec["encoding"])]
    cell_config = {
        "protocol_id": PROTOCOL_ID,
        **spec,
        "operation": "XOR",
        "K": 1,
        "hidden_neurons": int(protocol["model"]["hidden_neurons"]),
        "input_events_per_trial": int(encoding_cfg["events_per_trial"]),
        "target_output_step": target_output_step(protocol),
        "weight_learning_rate": float(spec.get("weight_learning_rate", protocol["optimization"]["weight_learning_rate"])),
        "delay_learning_rate": float(spec.get("delay_learning_rate", protocol["optimization"]["delay_learning_rate"])) if spec["learned_delay"] else 0.0,
        "full_batch_updates": int(spec.get("full_batch_updates", protocol["optimization"]["full_batch_updates"])),
        "checkpoint_selection": "final_only",
        "test_split_opened": False,
    }
    _strict_write_json(directory / "config.json", cell_config)
    model, result = train_cell(protocol, spec, device=device)
    record = result["final_record"]
    interface = result["final_interface"]
    metrics = {
        **cell_config,
        **{key: value for key, value in interface.items() if key not in {"predictions", "labels", "valid_pattern_mask"}},
        "initial_delay_mean_steps": float(np.mean(record["initial_independent_delays"])),
        "final_delay_mean_steps": float(np.mean(record["final_independent_delays"])),
        "final_delay_min_steps": float(np.min(record["final_independent_delays"])),
        "final_delay_max_steps": float(np.max(record["final_independent_delays"])),
        "final_delay_max_error_steps": float(result["final_delay_max_error_steps"]),
        "final_delay_fraction_within_tolerance": float(result["final_delay_fraction_within_tolerance"]),
        "initial_total_gradient_norm": float(result["initial_gradient_stats"]["norm"]),
        "initial_total_gradient_correct_coordinate_fraction": float(result["initial_gradient_stats"]["direction_fraction"]),
        "initial_total_gradient_nonzero_coordinate_fraction": float(result["initial_gradient_stats"]["nonzero_fraction"]),
        "initial_task_arrival_gradient_conflict_fraction": float(result["initial_task_arrival_gradient_conflict_fraction"]),
        "interface_pass": bool(result["interface_pass"]),
        "learned_delay_pass": bool(result["learned_delay_pass"]),
        "complete": True,
    }
    truth = {
        "protocol_id": PROTOCOL_ID,
        "evaluation_split": "exhaustive_truth_table_training_domain",
        "patterns": _truth_records(record),
        "predictions": interface["predictions"],
        "labels": interface["labels"],
        "valid_pattern_mask": interface["valid_pattern_mask"],
        "exact_truth_table_completion": bool(interface["exact_truth_table_completion"]),
        "exact_interface_completion": bool(interface["exact_interface_completion"]),
        "test_split_opened": False,
    }
    torch.save(model.state_dict(), directory / "final_model.pt")
    _strict_write_json(
        directory / "training_log.json",
        [{key: float(values[index]) for key, values in result["history"].items()} for index in range(len(result["history"]["step"]))],
    )
    _strict_write_json(directory / "exhaustive_truth_table_results.json", truth)
    _strict_write_json(directory / "resource_ledger.json", resource_ledger(model, spec, record))
    _strict_write_json(metrics_path, metrics)
    plots = directory / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    gradient_arrays = {
        f"initial_{name}_gradient": np.asarray(value if value is not None else [], dtype=np.float64)
        for name, value in result["initial_gradients"].items()
    }
    np.savez_compressed(plots / "diagnostic_data.npz", **result["history"], **record, **gradient_arrays)
    save_diagnostic_panel(protocol, spec, result, plots / "diagnostic_panel.png")
    return metrics


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_rows(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for granularity in [str(item["name"]) for item in _granularities(protocol)]:
        for condition in protocol["losses"]["per_parameter_arrival_centroid"]["conditions"]:
            weight = float(condition["weight"])
            selected = [
                row for row in rows
                if str(row["granularity"]) == granularity
                and float(row["arrival_auxiliary_weight"]) == weight
            ]
            candidates.append({
                "granularity": granularity,
                "arrival_condition": str(condition["name"]),
                "arrival_auxiliary_weight": weight,
                "complete_cells": len(selected),
                "passing_cells": sum(bool(row["learned_delay_pass"]) for row in selected),
                "candidate_pass": len(selected) == 10 and all(bool(row["learned_delay_pass"]) for row in selected),
            })
    return candidates


def _granularity_summary_plot(candidates: list[dict[str, Any]], output: Path, title: str) -> None:
    granularities = ["global", "per_hidden_neuron", "per_synapse"]
    weights = [0.0, 0.01]
    matrix = np.zeros((len(weights), len(granularities)))
    for i, weight in enumerate(weights):
        for j, granularity in enumerate(granularities):
            item = next(
                row for row in candidates
                if row["granularity"] == granularity and float(row["arrival_auxiliary_weight"]) == weight
            )
            matrix[i, j] = float(item["passing_cells"])
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=10, cmap="viridis", aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{int(matrix[i, j])}/10", ha="center", va="center")
    axis.set_xticks(range(len(granularities)), granularities)
    axis.set_yticks(range(len(weights)), [f"lambda={value:g}" for value in weights])
    axis.set(title=title, xlabel="delay granularity", ylabel="arrival scaffold")
    fig.colorbar(image, ax=axis, label="passing cells")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def aggregate_stage_a(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = SUMMARY_ROOT / "stage_a"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output / "cells.csv")
    candidates = _candidate_rows(protocol, rows)
    global_scaffold = next(
        item for item in candidates
        if item["granularity"] == "global" and float(item["arrival_auxiliary_weight"]) == 0.01
    )
    higher = [
        item for item in candidates
        if item["granularity"] in {"per_hidden_neuron", "per_synapse"}
        and float(item["arrival_auxiliary_weight"]) == 0.01
    ]
    decision = {
        "protocol_id": PROTOCOL_ID,
        "stage": "stage_a",
        "expected_cells": expected_stage_a_cells(protocol),
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == expected_stage_a_cells(protocol),
        "candidate_results": candidates,
        "global_scaffold_replication_pass": bool(global_scaffold["candidate_pass"]),
        "higher_dof_extension_pass": any(bool(item["candidate_pass"]) for item in higher),
        "stage_b_controls_authorized": bool(global_scaffold["candidate_pass"]),
        "K_greater_than_one_authorized": False,
        "test_split_opened": False,
    }
    _strict_write_json(output / "decision.json", decision)
    _granularity_summary_plot(candidates, output / "granularity_gate_heatmap.png", "Level 1B Stage A: single-event exact gates")
    return decision


def aggregate_stage_b_controls(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = SUMMARY_ROOT / "stage_b_controls"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output / "cells.csv")
    oracle = [row for row in rows if row["condition"] == "fixed_oracle_d4_microburst"]
    negative = [row for row in rows if row["condition"] == "fixed_d0_microburst_delayed_target"]
    oracle_pass = len(oracle) == 5 and all(bool(row["interface_pass"]) for row in oracle)
    decision = {
        "protocol_id": PROTOCOL_ID,
        "stage": "stage_b_controls",
        "expected_cells": expected_stage_b_control_cells(protocol),
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == expected_stage_b_control_cells(protocol),
        "fixed_oracle_passing_cells": sum(bool(row["interface_pass"]) for row in oracle),
        "fixed_d0_passing_cells": sum(bool(row["interface_pass"]) for row in negative),
        "fixed_oracle_gate_pass": oracle_pass,
        "stage_b_learned_authorized": oracle_pass,
        "K_greater_than_one_authorized": False,
        "test_split_opened": False,
    }
    _strict_write_json(output / "decision.json", decision)
    return decision


def aggregate_stage_b_learned(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = SUMMARY_ROOT / "stage_b_learned"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output / "cells.csv")
    candidates = _candidate_rows(protocol, rows)
    scaffold = [item for item in candidates if float(item["arrival_auxiliary_weight"]) == 0.01]
    task_only = [item for item in candidates if float(item["arrival_auxiliary_weight"]) == 0.0]
    decision = {
        "protocol_id": PROTOCOL_ID,
        "stage": "stage_b_learned",
        "expected_cells": expected_stage_b_learned_cells(protocol),
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == expected_stage_b_learned_cells(protocol),
        "candidate_results": candidates,
        "burst_scaffold_gate_pass": any(bool(item["candidate_pass"]) for item in scaffold),
        "passing_scaffold_granularities": [item["granularity"] for item in scaffold if item["candidate_pass"]],
        "passing_task_only_granularities": [item["granularity"] for item in task_only if item["candidate_pass"]],
        "level1b_complete": len(rows) == expected_stage_b_learned_cells(protocol),
        "K_greater_than_one_authorized": False,
        "test_split_opened": False,
    }
    _strict_write_json(output / "decision.json", decision)
    _granularity_summary_plot(candidates, output / "microburst_granularity_gate_heatmap.png", "Level 1B Stage B: micro-burst exact gates")
    return decision


def _read_decision(path: Path, error: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(error)
    return json.loads(path.read_text(encoding="utf-8"))


def run_stage(
    protocol: dict[str, Any], *, stage: str, root: Path, device: str, smoke: bool
) -> dict[str, Any]:
    if stage == "a":
        specs = stage_a_specs(protocol)
        if smoke:
            specs = [specs[0], specs[-1]]
        rows = [run_cell(protocol, spec, root=root, device=device) for spec in specs]
        return {"smoke_cells": len(rows), "cells": rows} if smoke else aggregate_stage_a(protocol, rows)
    if smoke:
        stage_a_decision = {"stage_b_controls_authorized": True}
    else:
        stage_a_decision = _read_decision(
            SUMMARY_ROOT / "stage_a" / "decision.json",
            "Stage-B controls require the complete formal Stage-A decision",
        )
    if stage == "b_controls":
        specs = stage_b_control_specs(protocol, stage_a_decision)
        if smoke:
            specs = [specs[0], specs[-1]]
        rows = [run_cell(protocol, spec, root=root, device=device) for spec in specs]
        return {"smoke_cells": len(rows), "cells": rows} if smoke else aggregate_stage_b_controls(protocol, rows)
    if smoke:
        control_decision = {"stage_b_learned_authorized": True}
    else:
        control_decision = _read_decision(
            SUMMARY_ROOT / "stage_b_controls" / "decision.json",
            "learned micro-burst cells require the complete fixed-control decision",
        )
    specs = stage_b_learned_specs(protocol, control_decision)
    if smoke:
        specs = [specs[0], specs[-1]]
    rows = [run_cell(protocol, spec, root=root, device=device) for spec in specs]
    return {"smoke_cells": len(rows), "cells": rows} if smoke else aggregate_stage_b_learned(protocol, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a", "b_controls", "b_learned"], default="a")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    declared = {
        "a": int(protocol["stage_a_single_event_granularity"]["grid"]["deterministic_cells"]),
        "b_controls": int(protocol["stage_b_microburst_controls"]["grid"]["deterministic_cells"]),
        "b_learned": int(protocol["stage_b_learned_microburst"]["grid"]["deterministic_cells"]),
    }
    expected = {
        "a": expected_stage_a_cells(protocol),
        "b_controls": expected_stage_b_control_cells(protocol),
        "b_learned": expected_stage_b_learned_cells(protocol),
    }
    if expected != declared:
        raise SystemExit(f"declared cell counts do not match generated grids: {declared} vs {expected}")
    if args.stage == "a" and not args.smoke and not args.dry_run:
        if protocol["stage_a_single_event_granularity"]["status"] != "preregistered_ready":
            raise SystemExit("Stage A is not launch-ready")
    if args.stage == "b_controls" and not args.smoke and not args.dry_run:
        if protocol["stage_b_microburst_controls"]["status"] != "preregistered_ready":
            raise SystemExit("Stage-B controls are not launch-ready")
    if args.stage == "b_learned" and not args.smoke and not args.dry_run:
        if protocol["stage_b_learned_microburst"]["status"] != "preregistered_ready":
            raise SystemExit("learned micro-burst cells are not launch-ready")
    if args.dry_run:
        print(json.dumps({
            "protocol_id": PROTOCOL_ID,
            "stage": args.stage,
            "formal_cells": expected[args.stage],
            "conditional_lock": args.stage != "a",
            "test_split_opened": False,
        }, indent=2))
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

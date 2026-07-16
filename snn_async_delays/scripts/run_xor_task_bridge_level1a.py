"""Run the preregistered K=1 XOR task bridge (Level 1A v1)."""

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
import torch.nn.functional as F
import yaml

from data.boolean_dataset import ExhaustiveFixedOperationQueryDataset
from data.encoding import encode_simultaneous_trial
from snn.model import SNNSimultaneousModel
from train.trainer import (
    filtered_opponent_spike_train_loss,
    opponent_target_spike_train,
)
from utils.resource_ledger import dynamic_resource_ledger, static_resource_ledger
from utils.seed import set_seed


BASE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "xor_task_bridge_level1a_v1"
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


def expected_stage_i_cells(protocol: dict[str, Any]) -> int:
    grid = protocol["stage_i_interface_gate"]["grid"]
    return (
        len(protocol["stage_i_interface_gate"]["fixed_controls"])
        * len(protocol["losses"]["global_voltage_envelope"]["candidate_weights"])
        * len(protocol["optimization"]["candidate_weight_learning_rates"])
        * len(protocol["optimization"]["seeds"])
    )


def expected_stage_ii_cells(protocol: dict[str, Any]) -> int:
    learned = (
        len(protocol["losses"]["shared_arrival_centroid"]["candidate_weights"])
        * len(protocol["optimization"]["candidate_delay_learning_rates"])
        * len(protocol["optimization"]["learned_delay_initial_raw_values"])
        * len(protocol["optimization"]["seeds"])
    )
    fixed = len(protocol["optimization"]["seeds"])
    return learned + fixed


def stage_i_specs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for control in protocol["stage_i_interface_gate"]["fixed_controls"]:
        for eta in protocol["losses"]["global_voltage_envelope"]["candidate_weights"]:
            for lr_w in protocol["optimization"]["candidate_weight_learning_rates"]:
                for seed in protocol["optimization"]["seeds"]:
                    specs.append({
                        "stage": "stage_i",
                        "condition": str(control["name"]),
                        "seed": int(seed),
                        "fixed_delay_steps": float(control["fixed_input_hidden_delay_steps"]),
                        "target_delay_steps": float(control["target_delay_steps"]),
                        "voltage_envelope_weight": float(eta),
                        "weight_learning_rate": float(lr_w),
                        "arrival_auxiliary_weight": 0.0,
                        "delay_learning_rate": 0.0,
                        "initial_raw": None,
                        "learned_delay": False,
                        "selection_eligible": True,
                    })
    return specs


def stage_ii_specs(
    protocol: dict[str, Any], stage_i_decision: dict[str, Any]
) -> list[dict[str, Any]]:
    if not bool(stage_i_decision.get("stage_ii_authorized")):
        raise RuntimeError("Stage II is locked because Stage I did not pass")
    selected = stage_i_decision.get("selected_interface_candidate")
    if not isinstance(selected, dict):
        raise RuntimeError("Stage-I decision has no selected interface candidate")
    eta = float(selected["voltage_envelope_weight"])
    lr_w = float(selected["weight_learning_rate"])
    specs: list[dict[str, Any]] = []
    negative = protocol["stage_ii_learned_delay_gate"]["fixed_negative_control"]
    for seed in protocol["optimization"]["seeds"]:
        specs.append({
            "stage": "stage_ii",
            "condition": str(negative["name"]),
            "seed": int(seed),
            "fixed_delay_steps": float(negative["fixed_input_hidden_delay_steps"]),
            "target_delay_steps": float(negative["target_delay_steps"]),
            "voltage_envelope_weight": eta,
            "weight_learning_rate": lr_w,
            "arrival_auxiliary_weight": 0.0,
            "delay_learning_rate": 0.0,
            "initial_raw": None,
            "learned_delay": False,
            "selection_eligible": False,
        })
    learned = protocol["stage_ii_learned_delay_gate"]["learned_condition"]
    for lam in protocol["losses"]["shared_arrival_centroid"]["candidate_weights"]:
        for lr_d in protocol["optimization"]["candidate_delay_learning_rates"]:
            for initial_raw in protocol["optimization"]["learned_delay_initial_raw_values"]:
                for seed in protocol["optimization"]["seeds"]:
                    specs.append({
                        "stage": "stage_ii",
                        "condition": str(learned["name"]),
                        "seed": int(seed),
                        "fixed_delay_steps": None,
                        "target_delay_steps": float(learned["target_delay_steps"]),
                        "voltage_envelope_weight": eta,
                        "weight_learning_rate": lr_w,
                        "arrival_auxiliary_weight": float(lam),
                        "delay_learning_rate": float(lr_d),
                        "initial_raw": float(initial_raw),
                        "learned_delay": True,
                        "selection_eligible": True,
                    })
    return specs


def _token(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def cell_directory(root: Path, spec: dict[str, Any]) -> Path:
    base = root / spec["stage"] / spec["condition"]
    if spec["stage"] == "stage_i":
        return (
            base
            / f"eta_{_token(spec['voltage_envelope_weight'])}"
            / f"lrw_{_token(spec['weight_learning_rate'])}"
            / f"seed_{spec['seed']}"
        )
    if bool(spec["learned_delay"]):
        return (
            base
            / f"lambda_{_token(spec['arrival_auxiliary_weight'])}"
            / f"lrd_{_token(spec['delay_learning_rate'])}"
            / f"init_{_token(spec['initial_raw'])}"
            / f"seed_{spec['seed']}"
        )
    return base / f"seed_{spec['seed']}"


def target_output_step(protocol: dict[str, Any], target_delay_steps: float) -> int:
    return int(
        protocol["encoding"]["event_step"]
        + float(target_delay_steps)
        + 2
    )


def exact_truth_batch(device: str = "cpu") -> tuple[torch.Tensor, ...]:
    dataset = ExhaustiveFixedOperationQueryDataset(["XOR"])
    return (
        dataset.A.to(device),
        dataset.B.to(device),
        dataset.op_ids.to(device),
        dataset.labels.to(device),
    )


def encode_exact_truth(
    protocol: dict[str, Any], A: torch.Tensor, B: torch.Tensor, *, device: str
) -> torch.Tensor:
    return encode_simultaneous_trial(
        A,
        B,
        win_len=int(protocol["encoding"]["input_window_steps"]),
        read_len=int(protocol["timing"]["read_window_steps"]),
        device=device,
        encoding_mode="binary_one_hot",
        one_hot_phase=float(protocol["encoding"]["event_phase"]),
        one_hot_n_spikes=int(protocol["encoding"]["selected_channel_events"]),
    )


def build_model(protocol: dict[str, Any], spec: dict[str, Any]) -> SNNSimultaneousModel:
    model_cfg = protocol["model"]
    hidden_cfg = model_cfg["hidden_lif"]
    output_cfg = model_cfg["output_lif"]
    learned = bool(spec["learned_delay"])
    return SNNSimultaneousModel(
        n_queries=1,
        n_hidden=int(model_cfg["hidden_neurons"]),
        win_len=int(protocol["encoding"]["input_window_steps"]),
        read_len=int(protocol["timing"]["read_window_steps"]),
        d_max=int(model_cfg["d_max_steps"]),
        train_mode="weights_and_delays" if learned else "weights_only",
        delay_param_type=str(model_cfg["input_hidden_delay_parameterization"]),
        fixed_delay_value=(None if learned else float(spec["fixed_delay_steps"])),
        shared_delay=learned,
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
    model: SNNSimultaneousModel, spec: dict[str, Any]
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = []
    weights = model.weight_params()
    if weights:
        groups.append({"params": weights, "lr": float(spec["weight_learning_rate"])})
    delays = model.delay_params()
    if delays:
        groups.append({"params": delays, "lr": float(spec["delay_learning_rate"])})
    if not groups:
        raise RuntimeError("Level 1A requires at least one trainable parameter group")
    return torch.optim.Adam(groups)


def global_voltage_envelope_loss(
    pre_reset_voltage: torch.Tensor,
    target_spikes: torch.Tensor,
    *,
    threshold: float,
    surrogate_beta: float,
) -> torch.Tensor:
    """Balanced crossing/suppression loss over every output-time position."""
    if pre_reset_voltage.shape != target_spikes.shape:
        raise ValueError("pre-reset voltage and target spikes must have identical shapes")
    positive = target_spikes > 0.5
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("global envelope requires positive and negative positions")
    logits = float(surrogate_beta) * (pre_reset_voltage - float(threshold))
    positive_loss = F.softplus(-logits[positive]).mean()
    negative_loss = F.softplus(logits[negative]).mean()
    return 0.5 * positive_loss + 0.5 * negative_loss


def shared_arrival_trace(
    spike_input: torch.Tensor, delay_steps: torch.Tensor, *, d_max: int
) -> torch.Tensor:
    """Unweighted arrival mass with the production d+1 buffer convention."""
    if delay_steps.numel() != 1:
        raise ValueError("Level 1A arrival trace requires one shared delay scalar")
    delay = torch.clamp(delay_steps.reshape(()), 0.0, float(d_max))
    floor_index = int(torch.floor(delay.detach()).item())
    ceil_index = min(floor_index + 1, int(d_max))
    alpha = delay - float(floor_index)
    batch, total_steps, _ = spike_input.shape
    values: list[torch.Tensor] = []
    zero = torch.zeros(batch, dtype=spike_input.dtype, device=spike_input.device)
    for time_index in range(total_steps):
        source_floor = time_index - 1 - floor_index
        source_ceil = time_index - 1 - ceil_index
        floor_mass = (
            spike_input[:, source_floor, :].sum(dim=1)
            if source_floor >= 0 else zero
        )
        ceil_mass = (
            spike_input[:, source_ceil, :].sum(dim=1)
            if source_ceil >= 0 else zero
        )
        values.append((1.0 - alpha) * floor_mass + alpha * ceil_mass)
    return torch.stack(values, dim=1)


def trace_centroid_batch(trace: torch.Tensor) -> torch.Tensor:
    if trace.dim() != 2:
        raise ValueError("trace must have shape [batch,time]")
    if bool((trace.detach() < -1e-10).any()):
        raise ValueError("arrival traces must be nonnegative")
    time = torch.arange(trace.shape[1], dtype=trace.dtype, device=trace.device)
    return (trace * time.unsqueeze(0)).sum(dim=1) / (trace.sum(dim=1) + 1e-8)


def arrival_centroid_loss(
    spike_input: torch.Tensor,
    delay_steps: torch.Tensor,
    *,
    target_delay_steps: float,
    d_max: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output = shared_arrival_trace(spike_input, delay_steps, d_max=d_max)
    target_delay = torch.tensor(
        float(target_delay_steps), dtype=spike_input.dtype, device=spike_input.device
    )
    target = shared_arrival_trace(spike_input, target_delay, d_max=d_max).detach()
    output_centroid = trace_centroid_batch(output)
    target_centroid = trace_centroid_batch(target)
    loss = 0.5 * (output_centroid - target_centroid).pow(2).mean()
    return loss, output_centroid, target_centroid, output, target


def _parameter_gradient(
    loss: torch.Tensor, parameter: torch.Tensor | None, *, retain_graph: bool
) -> float | None:
    if parameter is None:
        return None
    value = torch.autograd.grad(
        loss, parameter, retain_graph=retain_graph, allow_unused=True
    )[0]
    return 0.0 if value is None else float(value.detach().reshape(()).item())


def _weight_grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().norm().item())


def exact_interface_metrics(
    output_spikes: torch.Tensor,
    target_spikes: torch.Tensor,
    labels: torch.Tensor,
    hidden_spikes_per_pattern: torch.Tensor,
) -> dict[str, Any]:
    spikes = output_spikes.detach()
    target = target_spikes.detach()
    labels_flat = labels.detach().reshape(-1)
    counts = spikes.sum(dim=1).reshape(spikes.shape[0], 1, 2)
    logits = counts[:, :, 1] - counts[:, :, 0]
    predictions = (logits > 0).float().reshape(-1)
    correct = predictions == labels_flat
    class_scores = []
    for class_value in (0.0, 1.0):
        mask = labels_flat == class_value
        class_scores.append(float(correct[mask].float().mean().item()))
    exact_matches = (spikes == target).all(dim=(1, 2))
    silent = counts.sum(dim=2).reshape(-1) == 0
    collision = (
        (counts[:, :, 0] > 0) & (counts[:, :, 1] > 0)
    ).reshape(-1)
    correct_time = []
    for index in range(spikes.shape[0]):
        channel = int(labels_flat[index].item())
        target_time = int(target[index, :, channel].argmax().item())
        correct_time.append(bool(spikes[index, target_time, channel].item() > 0.5))
    active_hidden = hidden_spikes_per_pattern.detach().reshape(-1) > 0
    return {
        "accuracy": float(correct.float().mean().item()),
        "balanced_accuracy": float(sum(class_scores) / 2.0),
        "exact_target_spike_train_matches": int(exact_matches.sum().item()),
        "exact_truth_table_completion": bool(correct.all().item()),
        "exact_interface_completion": bool(exact_matches.all().item()),
        "silent_rate": float(silent.float().mean().item()),
        "collision_rate": float(collision.float().mean().item()),
        "correct_target_time_rate": float(np.mean(correct_time)),
        "mean_output_spikes_per_trial": float(spikes.sum(dim=(1, 2)).mean().item()),
        "hidden_active_pattern_count": int(active_hidden.sum().item()),
        "predictions": [int(value) for value in predictions.cpu().tolist()],
        "labels": [int(value) for value in labels_flat.cpu().tolist()],
        "valid_pattern_mask": [bool(value) for value in exact_matches.cpu().tolist()],
    }


def per_cell_interface_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["exact_truth_table_completion"]
        and metrics["exact_interface_completion"]
        and float(metrics["balanced_accuracy"]) == 1.0
        and float(metrics["silent_rate"]) == 0.0
        and float(metrics["collision_rate"]) == 0.0
        and float(metrics["correct_target_time_rate"]) == 1.0
        and float(metrics["mean_output_spikes_per_trial"]) == 1.0
        and int(metrics["hidden_active_pattern_count"]) == 4
    )


def train_cell(
    protocol: dict[str, Any], spec: dict[str, Any], *, device: str
) -> tuple[SNNSimultaneousModel, dict[str, Any]]:
    set_seed(int(spec["seed"]))
    model = build_model(protocol, spec).to(device)
    optimizer = build_optimizer(model, spec)
    A, B, _, labels = exact_truth_batch(device)
    spike_input = encode_exact_truth(protocol, A, B, device=device)
    target_step = target_output_step(protocol, float(spec["target_delay_steps"]))
    target_spikes = opponent_target_spike_train(
        labels,
        total_steps=int(protocol["timing"]["total_steps"]),
        input_steps=int(protocol["encoding"]["input_window_steps"]),
        output_window_len=int(protocol["timing"]["read_window_steps"]),
        timing_mode="simultaneous_offset",
        target_offset_steps=target_step - int(protocol["encoding"]["input_window_steps"]),
    )
    eta = float(spec["voltage_envelope_weight"])
    lam = float(spec["arrival_auxiliary_weight"])
    updates = int(protocol["optimization"]["full_batch_updates"])
    filter_tau = float(protocol["losses"]["filtered_hard_spike"]["filter_tau_steps"])
    d_max = int(protocol["model"]["d_max_steps"])
    threshold = float(protocol["model"]["output_lif"]["threshold_au"])
    beta = float(protocol["model"]["surrogate_beta"])
    delay_parameter = model.syn_ih.delay_raw if bool(spec["learned_delay"]) else None

    history: dict[str, list[float]] = {
        key: [] for key in (
            "step", "total_loss", "spike_loss", "envelope_loss", "arrival_loss",
            "delay_steps", "arrival_centroid", "target_arrival_centroid",
            "task_delay_gradient", "spike_delay_gradient", "envelope_delay_gradient",
            "arrival_delay_gradient", "total_delay_gradient", "ih_weight_gradient_norm",
            "ho_weight_gradient_norm", "accuracy", "balanced_accuracy",
            "exact_interface_patterns", "hidden_active_patterns", "silent_rate",
            "collision_rate",
        )
    }
    initial_arrival = None
    target_arrival = None
    initial_component_gradients: dict[str, float | None] = {}

    model.train()
    for step in range(updates + 1):
        optimizer.zero_grad(set_to_none=True)
        _, info = model(spike_input, return_output_spike_train=True)
        output_spikes = info["output_spike_train"]
        spike_loss = filtered_opponent_spike_train_loss(
            output_spikes, target_spikes, labels, tau_steps=filter_tau
        )
        envelope_loss = global_voltage_envelope_loss(
            info["output_pre_reset_train"],
            target_spikes,
            threshold=threshold,
            surrogate_beta=beta,
        )
        current_delay = model.syn_ih.get_delays()[0, 0]
        (
            arrival_loss,
            arrival_centroid,
            target_arrival_centroid,
            arrival_trace,
            target_arrival_trace,
        ) = arrival_centroid_loss(
            spike_input,
            current_delay,
            target_delay_steps=float(spec["target_delay_steps"]),
            d_max=d_max,
        )
        task_loss = spike_loss + eta * envelope_loss
        total_loss = task_loss + lam * arrival_loss

        spike_gradient = _parameter_gradient(
            spike_loss, delay_parameter, retain_graph=True
        )
        envelope_gradient = _parameter_gradient(
            envelope_loss, delay_parameter, retain_graph=True
        )
        arrival_gradient = _parameter_gradient(
            arrival_loss, delay_parameter, retain_graph=True
        )
        task_gradient = _parameter_gradient(
            task_loss, delay_parameter, retain_graph=True
        )
        total_loss.backward()
        total_gradient = (
            None if delay_parameter is None
            else float(delay_parameter.grad.detach().reshape(()).item())
        )
        interface = exact_interface_metrics(
            output_spikes,
            target_spikes,
            labels,
            info["total_hidden_spikes"],
        )

        values = {
            "step": float(step),
            "total_loss": float(total_loss.detach().item()),
            "spike_loss": float(spike_loss.detach().item()),
            "envelope_loss": float(envelope_loss.detach().item()),
            "arrival_loss": float(arrival_loss.detach().item()),
            "delay_steps": float(current_delay.detach().item()),
            "arrival_centroid": float(arrival_centroid.detach().mean().item()),
            "target_arrival_centroid": float(target_arrival_centroid.detach().mean().item()),
            "task_delay_gradient": float(task_gradient or 0.0),
            "spike_delay_gradient": float(spike_gradient or 0.0),
            "envelope_delay_gradient": float(envelope_gradient or 0.0),
            "arrival_delay_gradient": float(arrival_gradient or 0.0),
            "total_delay_gradient": float(total_gradient or 0.0),
            "ih_weight_gradient_norm": _weight_grad_norm(model.syn_ih.weight),
            "ho_weight_gradient_norm": _weight_grad_norm(model.syn_ho.weight),
            "accuracy": float(interface["accuracy"]),
            "balanced_accuracy": float(interface["balanced_accuracy"]),
            "exact_interface_patterns": float(interface["exact_target_spike_train_matches"]),
            "hidden_active_patterns": float(interface["hidden_active_pattern_count"]),
            "silent_rate": float(interface["silent_rate"]),
            "collision_rate": float(interface["collision_rate"]),
        }
        for key, value in values.items():
            history[key].append(value)
        if step == 0:
            initial_arrival = arrival_trace.detach().cpu().numpy()
            target_arrival = target_arrival_trace.detach().cpu().numpy()
            initial_component_gradients = {
                "spike": spike_gradient,
                "envelope": envelope_gradient,
                "arrival": arrival_gradient,
                "task": task_gradient,
                "total": total_gradient,
            }
        if step < updates:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=float(protocol["optimization"]["gradient_clip_norm"]),
            )
            optimizer.step()

    model.eval()
    with torch.no_grad():
        _, final_info = model(
            spike_input, record=True, return_output_spike_train=True
        )
        final_delay = model.syn_ih.get_delays()[0, 0]
        final_arrival = shared_arrival_trace(
            spike_input, final_delay, d_max=d_max
        )
    assert initial_arrival is not None and target_arrival is not None
    final_interface = exact_interface_metrics(
        final_info["output_spike_train"],
        target_spikes,
        labels,
        final_info["total_hidden_spikes"],
    )
    initial_delay = float(history["delay_steps"][0])
    target_delay = float(spec["target_delay_steps"])
    initial_total_gradient = initial_component_gradients.get("total")
    directional = bool(spec["learned_delay"])
    direction_correct = (
        bool(
            initial_total_gradient is not None
            and abs(float(initial_total_gradient)) > 1e-12
            and float(initial_total_gradient) * (initial_delay - target_delay) > 0.0
        ) if directional else None
    )
    task_gradient = initial_component_gradients.get("task")
    arrival_gradient = initial_component_gradients.get("arrival")
    gradient_conflict = bool(
        directional
        and task_gradient is not None
        and arrival_gradient is not None
        and abs(float(task_gradient)) > 1e-12
        and abs(float(arrival_gradient)) > 1e-12
        and float(task_gradient) * float(arrival_gradient) < 0.0
    )
    interface_pass = per_cell_interface_pass(final_interface)
    final_delay_error = abs(float(final_delay.item()) - target_delay)
    learned_gate = bool(
        interface_pass
        and final_delay_error
        <= float(protocol["per_cell_gates"]["learned_only"]["final_delay_error_at_most_steps"])
        and direction_correct
    ) if directional else False

    final_record = {
        "input": spike_input.detach().cpu().numpy(),
        "hidden": final_info["hidden_spike_train"].detach().cpu().numpy(),
        "output": final_info["output_spike_train"].detach().cpu().numpy(),
        "output_pre_reset": final_info["output_membrane_train"].detach().cpu().numpy(),
        "output_current": final_info["output_synaptic_current_train"].detach().cpu().numpy(),
        "target_output": target_spikes.detach().cpu().numpy(),
        "initial_arrival": initial_arrival,
        "final_arrival": final_arrival.detach().cpu().numpy(),
        "target_arrival": target_arrival,
        "labels": labels.detach().cpu().numpy(),
        "A": A.detach().cpu().numpy(),
        "B": B.detach().cpu().numpy(),
        "final_ih_weight": model.syn_ih.weight.detach().cpu().numpy(),
        "final_ho_weight": model.syn_ho.weight.detach().cpu().numpy(),
    }
    return model, {
        "history": {key: np.asarray(value, dtype=np.float64) for key, value in history.items()},
        "final_interface": final_interface,
        "initial_component_gradients": initial_component_gradients,
        "initial_delay_steps": initial_delay,
        "final_delay_steps": float(final_delay.item()),
        "final_delay_error_steps": final_delay_error,
        "initial_total_delay_gradient_direction_correct": direction_correct,
        "initial_task_arrival_gradient_conflict": gradient_conflict,
        "interface_pass": interface_pass,
        "learned_delay_pass": learned_gate,
        "final_record": final_record,
    }


def _truth_records(record: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(record["output"].shape[0]):
        spike_positions = np.argwhere(record["output"][index] > 0.5)
        rows.append({
            "A": int(record["A"][index, 0]),
            "B": int(record["B"][index, 0]),
            "label": int(record["labels"][index, 0]),
            "output_spikes": [
                {"time": int(time), "neuron": int(neuron)}
                for time, neuron in spike_positions.tolist()
            ],
            "hidden_spikes": int(record["hidden"][index].sum()),
            "exact_target_match": bool(
                np.array_equal(record["output"][index], record["target_output"][index])
            ),
        })
    return rows


def _resource_ledger(
    model: SNNSimultaneousModel, record: dict[str, np.ndarray]
) -> dict[str, Any]:
    ledger = static_resource_ledger(model)
    ledger.update(dynamic_resource_ledger(
        model,
        mean_input_spikes=float(record["input"].sum(axis=(1, 2)).mean()),
        mean_hidden1_spikes=float(record["hidden"].sum(axis=(1, 2)).mean()),
        mean_output_spikes=float(record["output"].sum(axis=(1, 2)).mean()),
    ))
    return ledger


def _strict_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def save_diagnostic_panel(
    protocol: dict[str, Any], spec: dict[str, Any], result: dict[str, Any], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    history = result["history"]
    record = result["final_record"]
    steps = history["step"]
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Level 1A {spec['stage']} | {spec['condition']} | seed {spec['seed']}"
    )

    axes[0, 0].semilogy(steps, np.maximum(history["total_loss"], 1e-12), label="total")
    axes[0, 0].semilogy(steps, np.maximum(history["spike_loss"], 1e-12), label="spike")
    axes[0, 0].semilogy(
        steps,
        np.maximum(float(spec["voltage_envelope_weight"]) * history["envelope_loss"], 1e-12),
        label="weighted envelope",
    )
    axes[0, 0].semilogy(
        steps,
        np.maximum(float(spec["arrival_auxiliary_weight"]) * history["arrival_loss"], 1e-12),
        label="weighted arrival",
    )
    axes[0, 0].set(title="Loss components", xlabel="update", ylabel="loss")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(steps, history["exact_interface_patterns"], label="exact spike trains")
    axes[0, 1].plot(steps, history["hidden_active_patterns"], label="hidden-active")
    axes[0, 1].axhline(4, linestyle="--", color="#c44e52", label="4/4 gate")
    axes[0, 1].set(title="Truth-table interface", xlabel="update", ylabel="patterns", ylim=(-.1, 4.3))
    axes[0, 1].legend(frameon=False)

    axes[0, 2].plot(steps, history["delay_steps"], color="#1f8a9b")
    axes[0, 2].axhline(float(spec["target_delay_steps"]), linestyle="--", color="#c44e52")
    axes[0, 2].set(title="Shared input delay", xlabel="update", ylabel="steps")

    axes[0, 3].plot(steps, history["arrival_centroid"], color="#6a4c93")
    axes[0, 3].plot(steps, history["target_arrival_centroid"], "--", color="#c44e52")
    axes[0, 3].set(title="Arrival centroid", xlabel="update", ylabel="simulation step")

    axes[1, 0].plot(steps, history["total_delay_gradient"], label="total")
    axes[1, 0].plot(steps, history["task_delay_gradient"], label="task")
    axes[1, 0].plot(
        steps,
        float(spec["arrival_auxiliary_weight"]) * history["arrival_delay_gradient"],
        label="weighted arrival",
    )
    axes[1, 0].axhline(0, color="#777777", linewidth=.8)
    axes[1, 0].set(title="Delay-gradient components", xlabel="update", ylabel="dL/d(raw)")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].semilogy(steps, np.maximum(history["ih_weight_gradient_norm"], 1e-12), label="input-hidden")
    axes[1, 1].semilogy(steps, np.maximum(history["ho_weight_gradient_norm"], 1e-12), label="hidden-output")
    axes[1, 1].set(title="Weight-gradient norms", xlabel="update", ylabel="L2 norm")
    axes[1, 1].legend(frameon=False)

    input_rows = record["input"].transpose(0, 2, 1).reshape(-1, record["input"].shape[1])
    axes[1, 2].imshow(input_rows, aspect="auto", interpolation="nearest", cmap="Greys")
    axes[1, 2].set(title="Truth-table input raster", xlabel="time", ylabel="pattern x input")

    hidden_rows = record["hidden"].transpose(0, 2, 1).reshape(-1, record["hidden"].shape[1])
    axes[1, 3].imshow(hidden_rows, aspect="auto", interpolation="nearest", cmap="Greys")
    axes[1, 3].set(title="Truth-table hidden raster", xlabel="time", ylabel="pattern x hidden")

    output_rows = record["output"].transpose(0, 2, 1).reshape(-1, record["output"].shape[1])
    target_rows = record["target_output"].transpose(0, 2, 1).reshape(-1, record["target_output"].shape[1])
    display = output_rows + 2.0 * target_rows
    axes[2, 0].imshow(display, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=3)
    axes[2, 0].set(title="Output raster (target=2, spike=1)", xlabel="time", ylabel="pattern x output")

    voltage_rows = record["output_pre_reset"].transpose(0, 2, 1).reshape(-1, record["output_pre_reset"].shape[1])
    image = axes[2, 1].imshow(voltage_rows, aspect="auto", interpolation="nearest", cmap="coolwarm")
    axes[2, 1].set(title="Output pre-reset voltage", xlabel="time", ylabel="pattern x output")
    fig.colorbar(image, ax=axes[2, 1], fraction=.046)

    weights = axes[2, 2].imshow(record["final_ih_weight"], aspect="auto", interpolation="nearest", cmap="coolwarm")
    axes[2, 2].set(title="Final input-hidden weights", xlabel="hidden", ylabel="input")
    fig.colorbar(weights, ax=axes[2, 2], fraction=.046)

    time = np.arange(record["initial_arrival"].shape[1])
    axes[2, 3].plot(time, record["initial_arrival"].mean(axis=0), label="initial")
    axes[2, 3].plot(time, record["final_arrival"].mean(axis=0), label="final")
    axes[2, 3].plot(time, record["target_arrival"].mean(axis=0), "--", label="target")
    axes[2, 3].set(title="Unweighted arrival trace", xlabel="time", ylabel="events")
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
    cell_config = {
        "protocol_id": PROTOCOL_ID,
        **spec,
        "operation": "XOR",
        "K": 1,
        "hidden_neurons": int(protocol["model"]["hidden_neurons"]),
        "input_events_per_trial": int(protocol["encoding"]["events_per_trial"]),
        "target_output_step": target_output_step(protocol, float(spec["target_delay_steps"])),
        "full_batch_updates": int(protocol["optimization"]["full_batch_updates"]),
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
        "initial_delay_steps": float(result["initial_delay_steps"]),
        "final_delay_steps": float(result["final_delay_steps"]),
        "final_delay_error_steps": float(result["final_delay_error_steps"]),
        "initial_spike_loss_delay_gradient": result["initial_component_gradients"]["spike"],
        "initial_voltage_envelope_delay_gradient": result["initial_component_gradients"]["envelope"],
        "initial_arrival_auxiliary_delay_gradient": result["initial_component_gradients"]["arrival"],
        "initial_task_delay_gradient": result["initial_component_gradients"]["task"],
        "initial_total_delay_gradient": result["initial_component_gradients"]["total"],
        "initial_total_delay_gradient_direction_correct": result["initial_total_delay_gradient_direction_correct"],
        "initial_task_arrival_gradient_conflict": bool(result["initial_task_arrival_gradient_conflict"]),
        "final_input_hidden_weight_gradient_norm": float(result["history"]["ih_weight_gradient_norm"][-1]),
        "final_hidden_output_weight_gradient_norm": float(result["history"]["ho_weight_gradient_norm"][-1]),
        "interface_pass": bool(result["interface_pass"]),
        "learned_delay_pass": bool(result["learned_delay_pass"]),
        "complete": True,
    }
    ledger = _resource_ledger(model, record)
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
        [
            {key: float(values[index]) for key, values in result["history"].items()}
            for index in range(len(result["history"]["step"]))
        ],
    )
    _strict_write_json(directory / "exhaustive_truth_table_results.json", truth)
    _strict_write_json(directory / "resource_ledger.json", ledger)
    _strict_write_json(metrics_path, metrics)
    plots = directory / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        plots / "diagnostic_data.npz",
        **result["history"],
        **record,
    )
    save_diagnostic_panel(protocol, spec, result, plots / "diagnostic_panel.png")
    return metrics


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _stage_i_summary_plot(
    protocol: dict[str, Any], rows: list[dict[str, Any]], output: Path
) -> None:
    etas = [float(v) for v in protocol["losses"]["global_voltage_envelope"]["candidate_weights"]]
    lrs = [float(v) for v in protocol["optimization"]["candidate_weight_learning_rates"]]
    controls = [str(v["name"]) for v in protocol["stage_i_interface_gate"]["fixed_controls"]]
    matrix = np.zeros((len(etas), len(lrs)), dtype=float)
    for i, eta in enumerate(etas):
        for j, lr in enumerate(lrs):
            selected = [
                row for row in rows
                if float(row["voltage_envelope_weight"]) == eta
                and float(row["weight_learning_rate"]) == lr
                and row["condition"] in controls
            ]
            matrix[i, j] = sum(bool(row["interface_pass"]) for row in selected)
    fig, axis = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=10, cmap="viridis", aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{int(matrix[i,j])}/10", ha="center", va="center")
    axis.set_xticks(range(len(lrs)), [f"{v:g}" for v in lrs])
    axis.set_yticks(range(len(etas)), [f"{v:g}" for v in etas])
    axis.set(xlabel="weight learning rate", ylabel="voltage-envelope weight", title="Level 1A Stage-I exact-interface cells")
    fig.colorbar(image, ax=axis, label="passing cells")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def aggregate_stage_i(
    protocol: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    output = SUMMARY_ROOT / "stage_i"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output / "cells.csv")
    candidates = []
    controls = [str(v["name"]) for v in protocol["stage_i_interface_gate"]["fixed_controls"]]
    for eta in protocol["losses"]["global_voltage_envelope"]["candidate_weights"]:
        for lr_w in protocol["optimization"]["candidate_weight_learning_rates"]:
            selected = [
                row for row in rows
                if row["condition"] in controls
                and float(row["voltage_envelope_weight"]) == float(eta)
                and float(row["weight_learning_rate"]) == float(lr_w)
            ]
            passed = len(selected) == 10 and all(bool(row["interface_pass"]) for row in selected)
            candidates.append({
                "voltage_envelope_weight": float(eta),
                "weight_learning_rate": float(lr_w),
                "complete_cells": len(selected),
                "passing_cells": sum(bool(row["interface_pass"]) for row in selected),
                "candidate_pass": bool(passed),
            })
    passing = sorted(
        [item for item in candidates if item["candidate_pass"]],
        key=lambda item: (item["voltage_envelope_weight"], item["weight_learning_rate"]),
    )
    selected_candidate = passing[0] if passing else None
    decision = {
        "protocol_id": PROTOCOL_ID,
        "stage": "stage_i",
        "expected_cells": expected_stage_i_cells(protocol),
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == expected_stage_i_cells(protocol),
        "candidate_results": candidates,
        "selected_interface_candidate": selected_candidate,
        "stage_i_pass": selected_candidate is not None,
        "stage_ii_authorized": selected_candidate is not None,
        "test_split_opened": False,
    }
    _strict_write_json(output / "decision.json", decision)
    _stage_i_summary_plot(protocol, rows, output / "interface_gate_heatmap.png")
    return decision


def _stage_ii_summary_plot(
    protocol: dict[str, Any], rows: list[dict[str, Any]], output: Path
) -> None:
    lambdas = [float(v) for v in protocol["losses"]["shared_arrival_centroid"]["candidate_weights"]]
    lrs = [float(v) for v in protocol["optimization"]["candidate_delay_learning_rates"]]
    matrix = np.zeros((len(lambdas), len(lrs)), dtype=float)
    for i, lam in enumerate(lambdas):
        for j, lr in enumerate(lrs):
            selected = [
                row for row in rows
                if bool(row["learned_delay"])
                and float(row["arrival_auxiliary_weight"]) == lam
                and float(row["delay_learning_rate"]) == lr
            ]
            matrix[i, j] = sum(bool(row["learned_delay_pass"]) for row in selected)
    fig, axis = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=10, cmap="viridis", aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{int(matrix[i,j])}/10", ha="center", va="center")
    axis.set_xticks(range(len(lrs)), [f"{v:g}" for v in lrs])
    axis.set_yticks(range(len(lambdas)), [f"{v:g}" for v in lambdas])
    axis.set(xlabel="delay learning rate", ylabel="arrival auxiliary weight", title="Level 1A Stage-II learned-delay cells")
    fig.colorbar(image, ax=axis, label="passing cells")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def aggregate_stage_ii(
    protocol: dict[str, Any], rows: list[dict[str, Any]], stage_i_decision: dict[str, Any]
) -> dict[str, Any]:
    output = SUMMARY_ROOT / "stage_ii"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output / "cells.csv")
    candidates = []
    for lam in protocol["losses"]["shared_arrival_centroid"]["candidate_weights"]:
        for lr_d in protocol["optimization"]["candidate_delay_learning_rates"]:
            selected = [
                row for row in rows
                if bool(row["learned_delay"])
                and float(row["arrival_auxiliary_weight"]) == float(lam)
                and float(row["delay_learning_rate"]) == float(lr_d)
            ]
            passed = len(selected) == 10 and all(bool(row["learned_delay_pass"]) for row in selected)
            candidates.append({
                "arrival_auxiliary_weight": float(lam),
                "delay_learning_rate": float(lr_d),
                "complete_cells": len(selected),
                "passing_cells": sum(bool(row["learned_delay_pass"]) for row in selected),
                "candidate_pass": bool(passed),
            })
    passing = sorted(
        [item for item in candidates if item["candidate_pass"]],
        key=lambda item: (item["arrival_auxiliary_weight"], item["delay_learning_rate"]),
    )
    selected_candidate = passing[0] if passing else None
    decision = {
        "protocol_id": PROTOCOL_ID,
        "stage": "stage_ii",
        "expected_cells": expected_stage_ii_cells(protocol),
        "complete_cells": len(rows),
        "all_cells_complete": len(rows) == expected_stage_ii_cells(protocol),
        "stage_i_pass": bool(stage_i_decision.get("stage_i_pass")),
        "selected_interface_candidate": stage_i_decision.get("selected_interface_candidate"),
        "candidate_results": candidates,
        "selected_learned_delay_candidate": selected_candidate,
        "level1a_pass": bool(stage_i_decision.get("stage_i_pass")) and selected_candidate is not None,
        "level1b_preregistration_authorized": selected_candidate is not None,
        "K_greater_than_one_authorized": False,
        "test_split_opened": False,
    }
    _strict_write_json(output / "decision.json", decision)
    _stage_ii_summary_plot(protocol, rows, output / "learned_delay_gate_heatmap.png")
    return decision


def run_stage(
    protocol: dict[str, Any], *, stage: str, root: Path, device: str, smoke: bool
) -> dict[str, Any]:
    if stage == "i":
        specs = stage_i_specs(protocol)
        if smoke:
            specs = [specs[0], specs[-1]]
        rows = [run_cell(protocol, spec, root=root, device=device) for spec in specs]
        return {"smoke_cells": len(rows), "cells": rows} if smoke else aggregate_stage_i(protocol, rows)
    stage_i_path = SUMMARY_ROOT / "stage_i" / "decision.json"
    if smoke and not stage_i_path.exists():
        stage_i_decision = {
            "stage_i_pass": True,
            "stage_ii_authorized": True,
            "selected_interface_candidate": {
                "voltage_envelope_weight": 0.1,
                "weight_learning_rate": 0.003,
            },
        }
    else:
        if not stage_i_path.exists():
            raise RuntimeError("Stage II requires the complete Stage-I decision file")
        stage_i_decision = json.loads(stage_i_path.read_text(encoding="utf-8"))
    specs = stage_ii_specs(protocol, stage_i_decision)
    if smoke:
        specs = [specs[0], next(spec for spec in specs if bool(spec["learned_delay"]))]
    rows = [run_cell(protocol, spec, root=root, device=device) for spec in specs]
    return (
        {"smoke_cells": len(rows), "cells": rows}
        if smoke else aggregate_stage_ii(protocol, rows, stage_i_decision)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["i", "ii"], default="i")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    if expected_stage_i_cells(protocol) != int(
        protocol["stage_i_interface_gate"]["grid"]["deterministic_cells"]
    ):
        raise SystemExit("Stage-I declared cell count does not match the grid")
    if expected_stage_ii_cells(protocol) != int(
        protocol["stage_ii_learned_delay_gate"]["grid"]["deterministic_cells"]
    ):
        raise SystemExit("Stage-II declared cell count does not match the grid")
    if args.stage == "i" and not args.smoke and not args.dry_run:
        if protocol["stage_i_interface_gate"]["status"] != "preregistered_ready":
            raise SystemExit("Stage I is not launch-ready")
    if args.stage == "ii" and not args.smoke and not args.dry_run:
        if protocol["stage_ii_learned_delay_gate"]["status"] != "locked_pending_stage_i":
            raise SystemExit("unexpected Stage-II protocol status")
    if args.dry_run:
        cells = expected_stage_i_cells(protocol) if args.stage == "i" else expected_stage_ii_cells(protocol)
        print(json.dumps({
            "protocol_id": PROTOCOL_ID,
            "stage": args.stage,
            "formal_cells": cells,
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

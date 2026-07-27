"""Run the preregistered K=5/8 mixed-operation MLP surface preview.

The six stability cells are invalid for scientific claims.  The 72-cell
exploratory surface is launchable only after the smoke audit passes and the
machine-readable protocol is explicitly unlocked.  No sealed test data are
created or opened by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from contextlib import contextmanager
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.boolean_dataset import MarginallyBalancedFixedOperationQueryDataset
from data.encoding import encode_simultaneous_trial
from snn.model import SNNSpatialParallelModel, SNNSimultaneousModel
from train.eval import _reliability_metrics, evaluate_simultaneous, save_eval_results
from train.trainer import build_optimizer, window_class_balanced_bce
from utils.seed import set_seed
from utils.viz import save_run_diagnostic_plots


BASE = Path(__file__).resolve().parents[1]
PROTOCOL = "mixedop_spatial_temporal_surface_preview_v1"
CONFIG_PATH = BASE / "configs" / f"{PROTOCOL}.yaml"
CONDITIONS = (
    "spatial_independent_d0",
    "shared_d0",
    "shared_temporal_oracle",
    "shared_temporal_wad",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL:
        raise ValueError("protocol id does not match runner")
    if float(protocol["optimization"]["weight_learning_rate"]) != 0.01:
        raise ValueError("the frozen weight LR must be 0.01")
    if float(protocol["optimization"]["readout_learning_rate"]) != 0.01:
        raise ValueError("the frozen readout LR must be 0.01")
    if float(protocol["optimization"]["delay_learning_rate"]) != 0.01:
        raise ValueError("the frozen delay LR must be 0.01")
    return protocol


def query_ops(protocol: dict[str, Any], K: int) -> list[str]:
    workloads = protocol["workloads"]
    values = workloads.get(K, workloads.get(str(K)))
    if values is None or len(values) != K:
        raise ValueError(f"no fixed {K}-query workload")
    return list(values)


def total_latency(K: int, output_window_len: int, input_steps: int = 10) -> int:
    return int(input_steps + K * output_window_len)


def grid_specs(protocol: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    seed = int(protocol["surface"]["seed"])
    if stage == "smoke":
        smoke = protocol["smoke"]
        tuples = product(smoke["K_values"], smoke["conditions"])
        return [
            {
                "K": int(K),
                "condition": condition,
                "total_hidden": int(smoke["total_hidden_neurons"]),
                "output_window_len": int(smoke["output_window_length"]),
                "seed": seed,
                "updates": int(smoke["optimizer_updates"]),
                "stage": "smoke",
            }
            for K, condition in tuples
        ]
    if stage != "formal":
        raise ValueError(f"unknown stage: {stage}")
    surface = protocol["surface"]
    tuples = product(
        surface["K_values"], surface["conditions"],
        surface["total_hidden_neurons"], surface["output_window_lengths"],
    )
    return [
        {
            "K": int(K),
            "condition": condition,
            "total_hidden": int(hidden),
            "output_window_len": int(window),
            "seed": seed,
            "updates": int(protocol["optimization"]["optimizer_updates"]),
            "stage": "formal",
        }
        for K, condition, hidden, window in tuples
    ]


def build_config(
    protocol: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    K = int(spec["K"])
    condition = str(spec["condition"])
    if condition not in CONDITIONS:
        raise ValueError(f"unregistered condition: {condition}")
    total_hidden = int(spec["total_hidden"])
    if condition == "spatial_independent_d0" and total_hidden % K:
        raise ValueError("total spatial hidden neurons must be divisible by K")
    window = int(spec["output_window_len"])
    input_steps = int(protocol["surface"]["input_window_steps"])
    T = total_latency(K, window, input_steps)
    model_cfg = protocol["model"]
    lif = model_cfg["hidden_lif"]
    delay = model_cfg["temporal_delay"]
    opt = protocol["optimization"]
    encoding = protocol["encoding"]
    rate_start_step = int(encoding.get("rate_start_step", 0))
    rate_steps = int(encoding.get("rate_steps", input_steps - rate_start_step))
    oracle_base_delay = int(delay.get("oracle_base_delay_steps", 0))
    delay_support_margin = int(delay.get("delay_support_margin_steps", 0))
    shared_d0 = condition == "shared_d0"
    d_max = (
        0 if condition == "spatial_independent_d0" or shared_d0
        else oracle_base_delay + (K - 1) * window + delay_support_margin
    )
    protocol_id = str(protocol["protocol_id"])
    ops = query_ops(protocol, K)
    is_wad = condition == "shared_temporal_wad"
    spatial = condition == "spatial_independent_d0"
    oracle_schedule = [float(oracle_base_delay + q * window) for q in range(K)]
    tying_name = str(delay.get("wad_tying", "per_synapse"))
    delay_tying = {
        "per_synapse": "pair",
        "per_query": "pre_group",
    }.get(tying_name, tying_name)
    if not is_wad:
        delay_tying = "pair"
    delay_init_raw = float(delay["wad_init_raw"])
    delay_init_value = delay.get("wad_init_value_steps")
    if is_wad and delay_init_value is not None:
        fraction = float(delay_init_value) / float(d_max)
        if not 0.0 < fraction < 1.0:
            raise ValueError("wad_init_value_steps must be interior to delay support")
        delay_init_raw = math.log(fraction / (1.0 - fraction))
    return {
        "protocol_id": protocol_id,
        "protocol_stage": spec["stage"],
        "study_class": protocol["study_class"],
        "experiment": protocol_id,
        "name": condition,
        "model_name": condition,
        "condition": condition,
        "seed": int(spec["seed"]),
        "smoke": spec["stage"] == "smoke",
        "point_label": spec.get("point_label"),
        "candidate_label": spec.get("candidate_label"),
        "event_budget_label": spec.get("event_budget_label"),
        "training_arm": spec.get("training_arm"),
        "path_variant": spec.get("path_variant"),
        "surface_condition": spec.get("surface_condition", condition),
        "event8_aligned": bool(spec.get("event8_aligned", False)),
        "primary_surface_metric": spec.get(
            "primary_surface_metric", "worst_query_balanced_accuracy"
        ),
        "K": K,
        "query_ops": ops,
        "ops_list": ops,
        "operation_assignment": "fixed_by_query_position",
        "input_schedule": "simultaneous",
        "n_input": int(encoding.get("channels_per_query", 4)) * K,
        "surface_total_hidden": total_hidden,
        "surface_hidden_width": total_hidden,
        "hidden_per_query": total_hidden // K if spatial else None,
        "hidden_width_semantics": "total_physical_hidden_neurons",
        "n_hidden": total_hidden,
        "n_hidden_total": total_hidden,
        "topology_type": (
            "spatial_independent_shared_decoder" if spatial else "shared_dense"
        ),
        "win_len": input_steps,
        "read_len": K * window,
        "T": T,
        "output_window_len": window,
        "d_max": d_max,
        "train_mode": "weights_and_delays" if is_wad else "weights_only",
        "fixed_delay_value": None if is_wad else 0.0,
        "oracle_delay_schedule": (
            oracle_schedule if condition == "shared_temporal_oracle"
            else [0.0] * K if shared_d0 else None
        ),
        "oracle_base_delay_steps": oracle_base_delay,
        "delay_support_margin_steps": delay_support_margin,
        "delay_support": [0.0, float(d_max)],
        "delay_placement": "input_to_hidden_only",
        "delay_param_type": delay["wad_parameterization"],
        "delay_step": 1.0,
        "delay_tying": delay_tying,
        "delay_group_index": (
            [q for q in range(K) for _ in range(int(encoding.get("channels_per_query", 4)))]
            if delay_tying == "pre_group" else None
        ),
        "delay_init_mode": delay["wad_init_mode"],
        "delay_init_raw": delay_init_raw,
        "delay_init_value_steps": (
            float(delay_init_value) if delay_init_value is not None else None
        ),
        "delay_init_std": 0.0,
        "delay_gradient_mode": "right_linear",
        "optimization_schedule": "joint",
        "lif_tau_m": float(lif["tau_m_steps"]),
        "lif_threshold": float(lif["threshold_au"]),
        "lif_reset": float(lif["reset_au"]),
        "lif_refractory": int(lif["refractory_steps"]),
        "surrogate_beta": float(lif["surrogate_beta"]),
        "dt": 1.0,
        "readout_type": "mlp",
        "readout_endpoint": (
            "shared_mlp_per_independent_module"
            if spatial else "shared_mlp_per_output_window"
        ),
        "observation_mode": "all_time" if spatial else "windowed_shared",
        "use_output_spikes": False,
        "output_encoding": "nonspiking_mlp_logits",
        "lr_w": float(opt["weight_learning_rate"]),
        "lr_d": float(opt["delay_learning_rate"]),
        "lr_readout": float(opt["readout_learning_rate"]),
        "batch_size": int(opt["batch_size"]),
        "optimizer_updates": int(spec["updates"]),
        "epochs": int(spec["updates"]),
        "validation_interval_updates": int(opt["validation_interval_updates"]),
        "loss_reduction": "query_by_class_macro_balanced_BCE",
        "checkpoint_selection": opt["checkpoint_selection"],
        "grad_clip": float(opt["gradient_clip_norm"]),
        "r_on": float(spec.get("r_on_hz", encoding.get("r_on_hz", 400.0))),
        "r_off": float(spec.get("r_off_hz", encoding.get("r_off_hz", 10.0))),
        "rate_start_step": rate_start_step,
        "rate_steps": rate_steps,
        "spike_penalty": 0.0,
        "delay_penalty": 0.0,
        "routing_loss_weight": float(spec.get("routing_loss_weight", 0.0)),
        "routing_loss_temperature": float(spec.get("routing_loss_temperature", 0.75)),
        "routing_loss_kind": str(spec.get("routing_loss_kind", "arrival_mass_ce")),
        "delay_credit_mode": str(spec.get("delay_credit_mode", "joint")),
        "save_schedule_checkpoint": bool(spec.get("save_schedule_checkpoint", False)),
        "schedule_gate_max_error_steps": float(spec.get("schedule_gate_max_error_steps", 1.0)),
        "homeo_lambda": 0.0,
        "homeo_target": 0.0,
        "encoding_mode": encoding["mode"],
        "one_hot_phase": float(encoding["one_hot_phase"]),
        "one_hot_n_spikes": int(encoding["selected_value_channel_events"]),
        "burst_n_spikes_on": 1,
        "burst_n_spikes_off": 1,
        "burst_phase_on": 1.0,
        "burst_phase_off": 1.0,
        "burst_jitter_ms": int(encoding["jitter_steps"]),
        "n_ops": 0,
        "n_train": int(protocol["data"]["train_samples"]),
        "n_val": int(protocol["data"]["validation_samples"]),
        "train_dataset_seed": int(spec["seed"]) + int(protocol["data"]["train_seed_offset"]),
        "validation_dataset_seed": int(spec["seed"]) + int(protocol["data"]["validation_seed_offset"]),
        "validation_encoding_seed": int(spec["seed"]) + int(protocol["data"].get("validation_encoding_seed_offset", 4000)),
        "evaluation_split": "sampled_marginally_balanced_validation",
        "sampled_joint_workload": True,
        "test_split_opened": False,
        "publication_claim_authorized": False,
    }


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    common = dict(
        win_len=cfg["win_len"], read_len=cfg["read_len"], d_max=cfg["d_max"],
        train_mode=cfg["train_mode"], fixed_delay_value=cfg["fixed_delay_value"],
        lif_tau_m=cfg["lif_tau_m"], lif_threshold=cfg["lif_threshold"],
        lif_reset=cfg["lif_reset"], lif_refractory=cfg["lif_refractory"],
        dt=cfg["dt"], surrogate_beta=cfg["surrogate_beta"],
        readout_type=cfg["readout_type"],
    )
    if cfg["condition"] == "spatial_independent_d0":
        return SNNSpatialParallelModel(
            n_queries=cfg["K"], hidden_per_query=cfg["hidden_per_query"], **common
        )
    model = SNNSimultaneousModel(
        n_queries=cfg["K"], n_hidden=cfg["surface_total_hidden"],
        delay_param_type=cfg["delay_param_type"], delay_step=cfg["delay_step"],
        delay_tying=cfg["delay_tying"], delay_init_mode=cfg["delay_init_mode"],
        delay_init_raw=cfg["delay_init_raw"], delay_init_std=cfg["delay_init_std"],
        delay_gradient_mode=cfg["delay_gradient_mode"],
        delay_group_index=cfg.get("delay_group_index"),
        n_input_channels=cfg["n_input"], use_output_spikes=False,
        observation_mode=cfg["observation_mode"],
        output_window_len=cfg["output_window_len"], **common,
    )
    if cfg["condition"] in {"shared_temporal_oracle", "shared_d0"}:
        schedule = torch.tensor(
            cfg["oracle_delay_schedule"], dtype=model.syn_ih.weight.dtype
        )
        per_input = torch.repeat_interleave(schedule, repeats=4)
        with torch.no_grad():
            model.syn_ih.fixed_delay_tensor = per_input[:, None].expand(
                model.n_input, model.n_hidden
            ).clone()
    return model


def encode_fn(cfg: dict[str, Any]):
    return partial(
        encode_simultaneous_trial,
        encoding_mode=cfg["encoding_mode"],
        one_hot_phase=cfg["one_hot_phase"],
        one_hot_n_spikes=cfg["one_hot_n_spikes"],
        rate_start_step=cfg.get("rate_start_step", 0),
        rate_steps=cfg.get("rate_steps"),
        burst_n_spikes_on=cfg["burst_n_spikes_on"],
        burst_n_spikes_off=cfg["burst_n_spikes_off"],
        burst_phase_on=cfg["burst_phase_on"],
        burst_phase_off=cfg["burst_phase_off"],
        burst_jitter_ms=cfg["burst_jitter_ms"],
    )


def loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    train_data = MarginallyBalancedFixedOperationQueryDataset(
        cfg["n_train"], cfg["query_ops"], cfg["train_dataset_seed"]
    )
    validation_data = MarginallyBalancedFixedOperationQueryDataset(
        cfg["n_val"], cfg["query_ops"], cfg["validation_dataset_seed"]
    )
    generator = torch.Generator().manual_seed(cfg["seed"] + 3000)
    train = DataLoader(
        train_data, batch_size=cfg["batch_size"], shuffle=True,
        drop_last=False, generator=generator,
    )
    validation = DataLoader(
        validation_data, batch_size=cfg["batch_size"], shuffle=False,
        drop_last=False,
    )
    return train, validation


def _batch_input(
    batch: Iterable[torch.Tensor], cfg: dict[str, Any], device: str, encoder
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    A, B, op_ids, labels = batch
    A, B = A.to(device), B.to(device)
    op_ids, labels = op_ids.to(device), labels.to(device)
    spikes = encoder(
        A, B, win_len=cfg["win_len"], read_len=cfg["read_len"],
        r_on=cfg["r_on"], r_off=cfg["r_off"], dt=cfg["dt"],
        device=device, op_ids=op_ids, n_ops=0,
    )
    return A, B, op_ids, labels, spikes


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    values = [p.grad.detach().float().norm().pow(2) for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt().item()) if values else 0.0


def _all_finite(tensors: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def _checkpoint_better(metrics: dict[str, Any], best: tuple[float, float] | None) -> bool:
    candidate = (
        float(metrics["worst_query_balanced_accuracy"]),
        float(metrics["exact_trial_accuracy"]),
    )
    return best is None or candidate > best


def routing_alignment_loss(
    spikes: torch.Tensor, model: torch.nn.Module, cfg: dict[str, Any]
) -> torch.Tensor:
    """Explicit routing supervision for query-tied temporal delays."""
    if cfg.get("delay_tying") != "pre_group":
        raise ValueError("routing alignment is defined only for per-query delays")
    kind = str(cfg.get("routing_loss_kind", "arrival_mass_ce"))
    temperature = float(cfg.get("routing_loss_temperature", 0.75))
    if temperature <= 0:
        raise ValueError("routing loss temperature must be positive")
    delays = model.syn_ih.get_delays()
    times = torch.arange(
        spikes.shape[1], device=spikes.device, dtype=delays.dtype
    ).view(-1, 1, 1)
    channels_per_query = cfg["n_input"] // cfg["K"]
    losses = []
    for query in range(cfg["K"]):
        start_channel = query * channels_per_query
        stop_channel = start_channel + channels_per_query
        event_rate = spikes[:, :, start_channel:stop_channel].mean(dim=0).unsqueeze(-1)
        arrivals = times + 1.0 + delays[start_channel:stop_channel].unsqueeze(0)
        if kind == "arrival_centroid_huber":
            event_mass = (
                event_rate.sum() * arrivals.shape[-1]
            ).clamp_min(1e-12)
            arrival_centroid = (event_rate * arrivals).sum() / event_mass
            target_center = float(
                cfg["win_len"]
                + query * cfg["output_window_len"]
                + 0.5 * cfg["output_window_len"]
            )
            normalized_error = (
                arrival_centroid - target_center
            ) / float(cfg["output_window_len"])
            losses.append(torch.nn.functional.smooth_l1_loss(
                normalized_error, torch.zeros_like(normalized_error), beta=0.25
            ))
            continue
        if kind != "arrival_mass_ce":
            raise ValueError(f"unknown routing loss kind: {kind}")
        masses = []
        for window in range(cfg["K"]):
            start = float(cfg["win_len"] + window * cfg["output_window_len"])
            stop = float(start + cfg["output_window_len"])
            membership = torch.sigmoid((arrivals - start) / temperature) * torch.sigmoid(
                (stop - arrivals) / temperature
            )
            masses.append((event_rate * membership).sum())
        mass = torch.stack(masses).clamp_min(1e-12)
        losses.append(-torch.log(mass[query] / mass.sum()))
    return torch.stack(losses).mean()


@torch.no_grad()
def _validation_snapshot_impl(
    model: torch.nn.Module, loader: DataLoader, cfg: dict[str, Any],
    device: str, encoder, collect: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    model.eval()
    logits_values: list[torch.Tensor] = []
    labels_values: list[torch.Tensor] = []
    A_values: list[torch.Tensor] = []
    B_values: list[torch.Tensor] = []
    op_values: list[torch.Tensor] = []
    hidden_spikes: list[torch.Tensor] = []
    active_fraction: list[torch.Tensor] = []
    hidden_windows: list[torch.Tensor] = []
    input_events: list[torch.Tensor] = []
    for batch in loader:
        A, B, op_ids, labels, spikes = _batch_input(batch, cfg, device, encoder)
        record = bool(collect and cfg["condition"] == "spatial_independent_d0")
        logits, info = model(spikes, record=record)
        logits_values.append(logits.detach().cpu())
        labels_values.append(labels.detach().cpu())
        hidden_spikes.append(info["total_hidden_spikes"].detach().cpu())
        active_fraction.append(info["active_hidden_fraction"].detach().cpu())
        if collect:
            A_values.append(A.detach().cpu())
            B_values.append(B.detach().cpu())
            op_values.append(op_ids.detach().cpu())
            input_events.append(
                spikes.detach().reshape(
                    spikes.shape[0], spikes.shape[1], cfg["K"],
                    cfg["n_input"] // cfg["K"],
                ).sum(dim=(1, 3)).cpu()
            )
            if "hidden_window_counts" in info:
                hidden_windows.append(info["hidden_window_counts"].sum(dim=2).float())
            else:
                train = info["hidden_spike_train"].float()
                per_window = []
                for q in range(cfg["K"]):
                    start = cfg["win_len"] + q * cfg["output_window_len"]
                    stop = start + cfg["output_window_len"]
                    per_window.append(train[:, start:stop, :].sum(dim=(1, 2)))
                hidden_windows.append(torch.stack(per_window, dim=1))

    logits = torch.cat(logits_values)
    labels = torch.cat(labels_values)
    predictions = (logits > 0).float()
    reliability = _reliability_metrics(predictions, labels)
    metrics = {
        **reliability,
        "loss": float(window_class_balanced_bce(logits, labels).item()),
        "mean_hidden_spikes": float(torch.cat(hidden_spikes).mean().item()),
        "mean_active_hidden_fraction": float(torch.cat(active_fraction).mean().item()),
    }
    if not collect:
        return metrics, None

    window_values = torch.cat(hidden_windows).numpy().astype(np.float32)
    arrays = {
        "A": torch.cat(A_values).numpy().astype(np.float32),
        "B": torch.cat(B_values).numpy().astype(np.float32),
        "op_ids": torch.cat(op_values).numpy().astype(np.int64),
        "labels": labels.numpy().astype(np.float32),
        "logits": logits.numpy().astype(np.float32),
        "predictions": predictions.numpy().astype(np.float32),
        "hidden_window_spikes": window_values,
        "input_events_per_query": torch.cat(input_events).numpy().astype(np.float32),
    }
    metrics["per_output_window_hidden_spikes"] = window_values.mean(axis=0).tolist()
    metrics["per_output_window_hidden_activity_fraction"] = (
        (window_values > 0).mean(axis=0).tolist()
    )
    metrics["mean_input_events_per_query"] = arrays[
        "input_events_per_query"
    ].mean(axis=0).tolist()
    return metrics, arrays


@contextmanager
def _fixed_encoding_rng(cfg: dict[str, Any], device: str, offset: int = 0):
    """Make stochastic validation encodings fixed without consuming train RNG."""
    devices: list[int] = []
    if str(device).startswith("cuda") and torch.cuda.is_available():
        parsed = torch.device(device)
        devices = [parsed.index if parsed.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        seed = int(cfg.get("validation_encoding_seed", cfg["seed"] + 4000)) + int(offset)
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


def validation_snapshot(
    model: torch.nn.Module, loader: DataLoader, cfg: dict[str, Any],
    device: str, encoder, collect: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    with _fixed_encoding_rng(cfg, device):
        return _validation_snapshot_impl(
            model, loader, cfg, device, encoder, collect=collect
        )


def _delay_diagnostics(model: torch.nn.Module, cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg["condition"] == "spatial_independent_d0":
        arrays = [value.detach().cpu().reshape(-1) for value in model.get_delays().values()]
        delays = torch.cat(arrays)
        query_means = [0.0] * cfg["K"]
        query_stds = [0.0] * cfg["K"]
    else:
        matrix = model.syn_ih.get_delays().detach().cpu()
        delays = matrix.reshape(-1)
        query_means, query_stds = [], []
        for q in range(cfg["K"]):
            values = matrix[4 * q:4 * (q + 1)].reshape(-1).float()
            query_means.append(float(values.mean().item()))
            query_stds.append(float(values.std(unbiased=False).item()))
    target = (
        [float(value) for value in cfg["declared_delay_targets_steps"]]
        if cfg.get("declared_delay_targets_steps") is not None
        else
        [float(value) for value in cfg["oracle_delay_schedule"]]
        if cfg.get("oracle_delay_schedule") is not None
        else [float(cfg.get("oracle_base_delay_steps", 0) + q * cfg["output_window_len"])
              for q in range(cfg["K"])]
    )
    correspondence_mae = float(np.mean(np.abs(np.asarray(query_means) - target)))
    correspondence_max = float(np.max(np.abs(np.asarray(query_means) - target)))
    result = {
        "delay_min_steps": float(delays.min().item()),
        "delay_max_steps": float(delays.max().item()),
        "delay_mean_steps": float(delays.float().mean().item()),
        "delay_std_steps": float(delays.float().std(unbiased=False).item()),
        "delay_query_mean_steps": query_means,
        "delay_query_std_steps": query_stds,
        "declared_query_schedule_steps": target,
        "delay_query_schedule_mae_steps": correspondence_mae,
        "delay_query_schedule_max_abs_error_steps": correspondence_max,
        "oracle_schedule_exact": bool(
            cfg["condition"] != "shared_temporal_oracle"
            or correspondence_mae <= 1e-7
        ),
    }
    if cfg.get("event_centroid_before_delay_steps") is not None:
        event_centroid = float(cfg["event_centroid_before_delay_steps"])
        arrival_centroids = [event_centroid + value for value in query_means]
        window_centers = [
            float(cfg["win_len"] + (q + 0.5) * cfg["output_window_len"])
            for q in range(cfg["K"])
        ]
        per_query_errors = [
            abs(arrival - center)
            for arrival, center in zip(arrival_centroids, window_centers)
        ]
        result.update({
            "arrival_centroid_steps": arrival_centroids,
            "target_arrival_centroid_steps": window_centers,
            "arrival_centroid_error_per_query_steps": per_query_errors,
            "arrival_centroid_max_abs_error_steps": float(max(per_query_errors)),
            "arrival_centroid_window_containment": [
                bool(
                    cfg["win_len"] + q * cfg["output_window_len"]
                    <= arrival
                    < cfg["win_len"] + (q + 1) * cfg["output_window_len"]
                )
                for q, arrival in enumerate(arrival_centroids)
            ],
        })
    return result


def _cross_target_metrics(predictions: np.ndarray, labels: np.ndarray) -> tuple[list, float]:
    preds = torch.from_numpy(predictions)
    target = torch.from_numpy(labels)
    matrix: list[list[float | None]] = []
    for output_q in range(preds.shape[1]):
        row = []
        for target_q in range(target.shape[1]):
            row.append(_reliability_metrics(
                preds[:, output_q:output_q + 1], target[:, target_q:target_q + 1]
            )["balanced_accuracy"])
        matrix.append(row)
    diagonal = [matrix[q][q] for q in range(len(matrix))]
    off = [matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix)) if i != j]
    gap = float(np.mean(diagonal) - np.mean(off))
    return matrix, gap


def _run_directory(protocol: dict[str, Any], cfg: dict[str, Any]) -> Path:
    root_key = "smoke_root" if cfg["smoke"] else "formal_root"
    condition_root = BASE / protocol["execution"][root_key] / cfg["condition"]
    path_variant = cfg.get("path_variant") or cfg.get("training_arm")
    if path_variant:
        condition_root = condition_root / str(path_variant)
    return (
        condition_root / f"K{cfg['K']}"
        / f"T{cfg['T']}_N{cfg['surface_total_hidden']}_w{cfg['output_window_len']}_seed{cfg['seed']}"
    )


def _required_artifacts(protocol: dict[str, Any]) -> list[str]:
    return list(protocol["required_cell_artifacts"])


def _complete(run_dir: Path, protocol: dict[str, Any]) -> bool:
    return all((run_dir / relative).exists() for relative in _required_artifacts(protocol))


def run_cell(
    protocol: dict[str, Any], spec: dict[str, Any], device: str, dry_run: bool = False,
    config_builder=None,
) -> Path:
    cfg = (
        config_builder(protocol, spec)
        if config_builder is not None
        else build_config(protocol, spec)
    )
    run_dir = _run_directory(protocol, cfg)
    if dry_run:
        return run_dir
    if _complete(run_dir, protocol):
        return run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"incomplete existing cell requires audit: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "config.json", cfg)

    started = time.time()
    set_seed(cfg["seed"])
    model = build_model(cfg).to(device)
    train_loader, validation_loader = loaders(cfg)
    encoder = encode_fn(cfg)
    optimizer = build_optimizer(model, cfg)
    parameters = [p for p in model.parameters() if p.requires_grad]
    weight_parameters = list(model.weight_params()) + list(model.readout_params())
    delay_parameters = list(model.delay_params())

    update_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    viz_rows: list[dict[str, Any]] = []
    best_score: tuple[float, float] | None = None
    best_update = 0
    best_metrics: dict[str, Any] | None = None
    best_schedule_score: tuple[float, float, float] | None = None
    best_schedule_update = 0
    interval_loss: list[float] = []
    interval_acc: list[float] = []
    interval_grad_w: list[float] = []
    interval_grad_d: list[float] = []

    initial, _ = validation_snapshot(model, validation_loader, cfg, device, encoder)
    initial_checkpoint_eligible = True
    if cfg.get("save_schedule_checkpoint"):
        initial_delay = _delay_diagnostics(model, cfg)
        initial.update(initial_delay)
        initial_checkpoint_eligible = bool(
            float(initial_delay["delay_query_schedule_max_abs_error_steps"])
            <= float(cfg.get("schedule_gate_max_error_steps", 1.0))
        )
        best_schedule_score = (
            -float(initial_delay["delay_query_schedule_max_abs_error_steps"]),
            float(initial["worst_query_balanced_accuracy"]),
            float(initial["exact_trial_accuracy"]),
        )
        torch.save(model.state_dict(), run_dir / "best_schedule_model.pt")
    if initial_checkpoint_eligible or not cfg.get("checkpoint_requires_schedule_gate", False):
        best_score = (
            float(initial["worst_query_balanced_accuracy"]),
            float(initial["exact_trial_accuracy"]),
        )
        best_metrics = initial
        torch.save(model.state_dict(), run_dir / "best_model.pt")
    validation_rows.append({"update": 0, **initial})
    viz_rows.append({
        "epoch": 0, "train_loss": initial["loss"], "val_loss": initial["loss"],
        "train_acc": initial["pooled_accuracy"], "val_acc": initial["pooled_accuracy"],
        "val_worst_query_balanced_accuracy": initial["worst_query_balanced_accuracy"],
        "val_exact_trial_accuracy": initial["exact_trial_accuracy"],
        "weight_grad_norm": 0.0, "delay_grad_norm": 0.0,
    })

    iterator = iter(train_loader)
    for update in range(1, cfg["optimizer_updates"] + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        model.train()
        _, _, _, labels, spikes = _batch_input(batch, cfg, device, encoder)
        optimizer.zero_grad(set_to_none=True)
        logits, info = model(spikes)
        task_loss = window_class_balanced_bce(logits, labels)
        route_loss = torch.zeros((), device=spikes.device)
        if float(cfg.get("routing_loss_weight", 0.0)) > 0:
            route_loss = routing_alignment_loss(spikes, model, cfg)
        routing_weight = float(cfg.get("routing_loss_weight", 0.0))
        loss = task_loss + routing_weight * route_loss
        if cfg.get("delay_credit_mode", "joint") == "routing_only_for_delays":
            if routing_weight <= 0 or not delay_parameters:
                raise ValueError(
                    "routing_only_for_delays requires routing loss and trainable delays"
                )
            task_loss.backward(retain_graph=True)
            for parameter in delay_parameters:
                parameter.grad = None
            (routing_weight * route_loss).backward()
        elif cfg.get("delay_credit_mode", "joint") == "joint":
            loss.backward()
        else:
            raise ValueError(f"unknown delay credit mode: {cfg['delay_credit_mode']}")
        weight_grad_norm = _gradient_norm(weight_parameters)
        delay_grad_norm = _gradient_norm(delay_parameters)
        per_query_delay_grad_norms: dict[str, float] = {}
        if cfg["condition"] == "shared_temporal_wad":
            delay_grad = getattr(model.syn_ih.delay_raw, "grad", None)
            if delay_grad is not None and delay_grad.ndim >= 1:
                for query in range(cfg["K"]):
                    if cfg.get("delay_tying") == "pre_group":
                        values = delay_grad[query:query + 1]
                    else:
                        channels_per_query = cfg["n_input"] // cfg["K"]
                        values = delay_grad[
                            channels_per_query * query:channels_per_query * (query + 1)
                        ]
                    per_query_delay_grad_norms[f"delay_grad_norm_q{query}"] = float(
                        values.detach().float().norm().item()
                    )
                    per_query_delay_grad_norms[f"delay_grad_mean_q{query}"] = float(
                        values.detach().float().mean().item()
                    )
        gradients_finite = _all_finite(
            p.grad for p in parameters if p.grad is not None
        )
        global_grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            parameters, cfg["grad_clip"], error_if_nonfinite=False
        )
        global_grad_norm = float(global_grad_norm_tensor.item())
        clip_applied = bool(math.isfinite(global_grad_norm) and global_grad_norm > cfg["grad_clip"])
        optimizer.step()
        delay_values = [value for value in model.get_delays().values()]
        parameter_finite = _all_finite(p.detach() for p in model.parameters())
        delays_finite = _all_finite(delay_values)
        delay_min = min(float(value.detach().min().item()) for value in delay_values)
        delay_max = max(float(value.detach().max().item()) for value in delay_values)
        delay_legal = delay_min >= -1e-6 and delay_max <= cfg["d_max"] + 1e-6
        pooled = float(((logits > 0).float() == labels).float().mean().item())
        row = {
            "update": update,
            "train_loss": float(loss.detach().item()),
            "task_loss": float(task_loss.detach().item()),
            "routing_loss": float(route_loss.detach().item()),
            "train_pooled_accuracy": pooled,
            "mean_hidden_spikes": float(info["total_hidden_spikes"].detach().mean().item()),
            "weight_readout_grad_norm": weight_grad_norm,
            "delay_grad_norm": delay_grad_norm,
            "global_grad_norm_before_clip": global_grad_norm,
            "clip_applied": clip_applied,
            "loss_finite": bool(torch.isfinite(loss.detach()).item()),
            "gradients_finite": gradients_finite,
            "parameters_finite": parameter_finite,
            "delays_finite": delays_finite,
            "delays_legal": delay_legal,
            "delay_min_steps": delay_min,
            "delay_max_steps": delay_max,
            **per_query_delay_grad_norms,
        }
        update_rows.append(row)
        interval_loss.append(row["train_loss"])
        interval_acc.append(pooled)
        interval_grad_w.append(weight_grad_norm)
        interval_grad_d.append(delay_grad_norm)
        if update % cfg["validation_interval_updates"] == 0 or update == cfg["optimizer_updates"]:
            metrics, _ = validation_snapshot(model, validation_loader, cfg, device, encoder)
            if cfg.get("save_schedule_checkpoint"):
                validation_delay = _delay_diagnostics(model, cfg)
                metrics.update(validation_delay)
                schedule_score = (
                    -float(validation_delay["delay_query_schedule_max_abs_error_steps"]),
                    float(metrics["worst_query_balanced_accuracy"]),
                    float(metrics["exact_trial_accuracy"]),
                )
                if best_schedule_score is None or schedule_score > best_schedule_score:
                    best_schedule_score = schedule_score
                    best_schedule_update = update
                    torch.save(model.state_dict(), run_dir / "best_schedule_model.pt")
            checkpoint_eligible = bool(
                not cfg.get("checkpoint_requires_schedule_gate", False)
                or float(metrics.get("delay_query_schedule_max_abs_error_steps", 0.0))
                <= float(cfg.get("schedule_gate_max_error_steps", 1.0))
            )
            validation_rows.append({"update": update, **metrics})
            viz_rows.append({
                "epoch": update,
                "train_loss": float(np.mean(interval_loss)),
                "val_loss": metrics["loss"],
                "train_acc": float(np.mean(interval_acc)),
                "val_acc": metrics["pooled_accuracy"],
                "val_worst_query_balanced_accuracy": metrics["worst_query_balanced_accuracy"],
                "val_exact_trial_accuracy": metrics["exact_trial_accuracy"],
                "weight_grad_norm": float(np.mean(interval_grad_w)),
                "delay_grad_norm": float(np.mean(interval_grad_d)),
            })
            interval_loss.clear(); interval_acc.clear()
            interval_grad_w.clear(); interval_grad_d.clear()
            if checkpoint_eligible and _checkpoint_better(metrics, best_score):
                best_score = (
                    float(metrics["worst_query_balanced_accuracy"]),
                    float(metrics["exact_trial_accuracy"]),
                )
                best_update = update
                best_metrics = metrics
                torch.save(model.state_dict(), run_dir / "best_model.pt")

    torch.save(model.state_dict(), run_dir / "last_model.pt")
    mechanism_valid_checkpoint_found = best_score is not None
    if not mechanism_valid_checkpoint_found:
        best_update = cfg["optimizer_updates"]
        best_metrics = validation_rows[-1]
        torch.save(model.state_dict(), run_dir / "best_model.pt")
    model.load_state_dict(torch.load(
        run_dir / "best_model.pt", map_location=device, weights_only=True
    ))
    final_metrics, arrays = validation_snapshot(
        model, validation_loader, cfg, device, encoder, collect=True
    )
    assert arrays is not None
    with _fixed_encoding_rng(cfg, device):
        detailed = evaluate_simultaneous(
            model, validation_loader, cfg, device, encode_fn=encoder,
            return_trial_records=False,
        )
    cross_matrix, routing_gap = _cross_target_metrics(
        arrays["predictions"], arrays["labels"]
    )
    delays = _delay_diagnostics(model, cfg)
    results = {
        **detailed,
        **final_metrics,
        **delays,
        "protocol_id": cfg["protocol_id"],
        "condition": cfg["condition"],
        "training_arm": cfg.get("training_arm"),
        "surface_condition": cfg.get("surface_condition", cfg["condition"]),
        "K": cfg["K"],
        "query_ops": cfg["query_ops"],
        "total_hidden_neurons": cfg["surface_total_hidden"],
        "total_latency_steps": cfg["T"],
        "output_window_len": cfg["output_window_len"],
        "neuron_update_proxy_N_times_T": cfg["surface_total_hidden"] * cfg["T"],
        "selected_checkpoint": "best_model.pt",
        "selected_update": best_update,
        "training_optimizer_updates_total": cfg["optimizer_updates"],
        "selected_validation_metrics": best_metrics,
        "mechanism_valid_checkpoint_found": mechanism_valid_checkpoint_found,
        "cross_target_balanced_accuracy_matrix": cross_matrix,
        "routing_selectivity_gap": routing_gap,
        "evaluation_split": cfg["evaluation_split"],
        "sampled_joint_workload": True,
        "validation_input_event_fingerprint_sha256": hashlib.sha256(
            arrays["input_events_per_query"].tobytes()
        ).hexdigest(),
        "test_split_opened": False,
        "claim_status": "invalid_smoke" if cfg["smoke"] else "exploratory_single_seed",
        "wall_time_seconds": time.time() - started,
    }
    save_eval_results(results, str(run_dir / "validation_results.json"))
    _write_json(run_dir / "resource_ledger.json", results["resource_ledger"])
    _write_csv(run_dir / "update_log.csv", update_rows)
    _write_csv(run_dir / "validation_log.csv", validation_rows)
    np.savez_compressed(run_dir / "validation_predictions.npz", **arrays)

    if cfg.get("save_schedule_checkpoint"):
        model.load_state_dict(torch.load(
            run_dir / "best_schedule_model.pt", map_location=device, weights_only=True
        ))
        schedule_metrics, schedule_arrays = validation_snapshot(
            model, validation_loader, cfg, device, encoder, collect=True
        )
        assert schedule_arrays is not None
        with _fixed_encoding_rng(cfg, device):
            schedule_detailed = evaluate_simultaneous(
                model, validation_loader, cfg, device, encode_fn=encoder,
                return_trial_records=False,
            )
        schedule_cross, schedule_gap = _cross_target_metrics(
            schedule_arrays["predictions"], schedule_arrays["labels"]
        )
        schedule_results = {
            **schedule_detailed,
            **schedule_metrics,
            **_delay_diagnostics(model, cfg),
            "protocol_id": cfg["protocol_id"],
            "condition": cfg["condition"],
            "training_arm": cfg.get("training_arm"),
            "K": cfg["K"],
            "query_ops": cfg["query_ops"],
            "total_hidden_neurons": cfg["surface_total_hidden"],
            "total_latency_steps": cfg["T"],
            "output_window_len": cfg["output_window_len"],
            "selected_checkpoint": "best_schedule_model.pt",
            "selected_update": best_schedule_update,
            "cross_target_balanced_accuracy_matrix": schedule_cross,
            "routing_selectivity_gap": schedule_gap,
            "evaluation_split": cfg["evaluation_split"],
            "test_split_opened": False,
            "claim_status": "exploratory_single_seed",
        }
        save_eval_results(
            schedule_results, str(run_dir / "schedule_validation_results.json")
        )
        np.savez_compressed(
            run_dir / "schedule_validation_predictions.npz", **schedule_arrays
        )
        model.load_state_dict(torch.load(
            run_dir / "best_model.pt", map_location=device, weights_only=True
        ))

    sample = validation_loader.dataset[0]
    save_run_diagnostic_plots(
        model, cfg, viz_rows, results, str(run_dir), cfg["K"], "mixed", device,
        seed=cfg["seed"] + 10000, dataset_override=sample,
    )
    missing = [name for name in _required_artifacts(protocol)
               if name != "run_complete.json" and not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"cell artifacts incomplete: {missing}")
    _write_json(run_dir / "run_complete.json", {
        "protocol_id": cfg["protocol_id"],
        "stage": cfg["protocol_stage"],
        "completed": True,
        "required_artifacts_complete": True,
        "selected_update": best_update,
        "test_split_opened": False,
        "wall_time_seconds": results["wall_time_seconds"],
    })
    return run_dir


def audit_smoke(protocol: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in grid_specs(protocol, "smoke"):
        cfg = build_config(protocol, spec)
        run_dir = _run_directory(protocol, cfg)
        complete = _complete(run_dir, protocol)
        update_rows: list[dict[str, str]] = []
        if (run_dir / "update_log.csv").exists():
            with (run_dir / "update_log.csv").open(encoding="utf-8") as handle:
                update_rows = list(csv.DictReader(handle))
        results = json.loads((run_dir / "validation_results.json").read_text(encoding="utf-8")) if complete else {}
        finite = bool(update_rows) and all(
            row["loss_finite"] == "True" and row["gradients_finite"] == "True"
            and row["parameters_finite"] == "True" and row["delays_finite"] == "True"
            for row in update_rows
        )
        active = bool(results.get("mean_hidden_spikes", 0.0) > 0.0)
        oracle_exact = bool(
            cfg["condition"] != "shared_temporal_oracle"
            or results.get("oracle_schedule_exact") is True
        )
        wad_grad = bool(
            cfg["condition"] != "shared_temporal_wad"
            or max((float(row["delay_grad_norm"]) for row in update_rows), default=0.0) > 0.0
        )
        legal = bool(update_rows) and all(row["delays_legal"] == "True" for row in update_rows)
        last20 = update_rows[-20:]
        not_continuously_clipped = len(last20) == 20 and not all(
            row["clip_applied"] == "True" for row in last20
        )
        passed = all((complete, finite, active, oracle_exact, wad_grad, legal,
                      not_continuously_clipped))
        rows.append({
            "K": cfg["K"], "condition": cfg["condition"],
            "total_hidden": cfg["surface_total_hidden"], "T": cfg["T"],
            "artifacts_complete": complete, "finite": finite,
            "nonzero_hidden_activity": active, "oracle_schedule_exact": oracle_exact,
            "wad_delay_gradient_nonzero": wad_grad, "delays_legal": legal,
            "final_20_not_continuously_clipped": not_continuously_clipped,
            "accuracy_used_as_gate": False, "passed": passed,
        })
    decision = {
        "protocol_id": str(protocol["protocol_id"]),
        "stage": "stability_smoke",
        "invalid_for_claims": True,
        "cells_expected": 6,
        "cells_audited": len(rows),
        "passed": len(rows) == 6 and all(row["passed"] for row in rows),
        "accuracy_used_as_gate": False,
        "formal_action": "authorize_without_recipe_change" if all(row["passed"] for row in rows) else "stop",
        "rows": rows,
    }
    generated = BASE / protocol["execution"]["generated_root"]
    _write_json(generated / "smoke_decision.json", decision)
    _write_csv(generated / "smoke_gate_cells.csv", rows)
    return decision


def _formal_preconditions(protocol: dict[str, Any]) -> None:
    if protocol["authorization"].get("formal_launch") is not True:
        raise SystemExit("formal launch is locked in the preregistration")
    decision_path = BASE / protocol["execution"]["generated_root"] / "smoke_decision.json"
    if not decision_path.exists():
        raise SystemExit("formal launch requires a saved smoke decision")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("passed") is not True:
        raise SystemExit("formal launch stopped because the smoke gate did not pass")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    if args.stage == "smoke" and protocol["authorization"].get("smoke_launch") is not True:
        raise SystemExit("smoke launch is not authorized")
    if args.stage == "formal" and not args.dry_run:
        _formal_preconditions(protocol)
    specs = grid_specs(protocol, args.stage)
    expected = 6 if args.stage == "smoke" else 72
    if len(specs) != expected:
        raise RuntimeError(f"frozen matrix has {len(specs)} cells, expected {expected}")
    if args.dry_run:
        print(json.dumps({
            "protocol": PROTOCOL, "stage": args.stage, "cells": len(specs),
            "paths": [str(_run_directory(protocol, build_config(protocol, spec)).relative_to(BASE))
                      for spec in specs],
        }, indent=2))
        return
    for spec in specs:
        run_cell(protocol, spec, args.device)
    if args.stage == "smoke":
        decision = audit_smoke(protocol)
        print(json.dumps({"protocol": PROTOCOL, "stage": "smoke", "passed": decision["passed"]}))
    else:
        print(json.dumps({"protocol": PROTOCOL, "stage": "formal", "cells_complete": 72}))


if __name__ == "__main__":
    main()

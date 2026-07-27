"""Current-simulator control cells for the capacity/scaling protocol."""

from __future__ import annotations

import csv
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.boolean_dataset import MarginallyBalancedFixedOperationQueryDataset
from data.encoding import encode_simultaneous_trial
from snn.model import SNNSimultaneousModel, SNNSpatialParallelModel
from train.eval import _reliability_metrics
from train.trainer import window_class_balanced_bce
from utils.capacity_scaling import make_result_record, reliability_pass
from utils.resource_ledger import dynamic_resource_ledger, static_resource_ledger


def current_fixed_input_schedule(K: int, window_width: int) -> torch.Tensor:
    """Current simulator has a documented one-step buffer offset."""
    return torch.repeat_interleave(
        torch.tensor([3 + query * window_width for query in range(K)], dtype=torch.float32),
        4,
    )


def _loader(
    n: int, operations: list[str], seed: int, batch_size: int, shuffle: bool,
) -> DataLoader:
    dataset = MarginallyBalancedFixedOperationQueryDataset(n, operations, seed=seed)
    generator = torch.Generator().manual_seed(seed + 101)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def _encoder(spec: dict[str, Any]):
    return partial(
        encode_simultaneous_trial, encoding_mode="binary_one_hot_packet",
        packet_start_step=6, packet_steps=4, win_len=10,
        read_len=int(spec["output_budget_B"]), device=spec["device"],
    )


def build_current_model(spec: dict[str, Any]) -> torch.nn.Module:
    K = int(spec["K"])
    hidden = int(spec["N_hidden_total"])
    # Protocol-generated specs carry these explicitly. Defaults preserve the
    # current runner's direct-call/test interface while matching the frozen
    # capacity-scaling neuron dynamics.
    lif_kwargs = {
        "lif_tau_m": float(spec.get("lif_tau_m", 10.0)),
        "lif_threshold": float(spec.get("lif_threshold", 0.2)),
        "lif_reset": float(spec.get("lif_reset", 0.0)),
        "lif_refractory": int(spec.get("lif_refractory", 2)),
        "surrogate_beta": float(spec.get("surrogate_beta", 4.0)),
    }
    if spec["condition"] == "independent_spatial_d0":
        if hidden % K:
            raise ValueError("independent spatial total hidden width must divide by K")
        model = SNNSpatialParallelModel(
            n_queries=K, hidden_per_query=hidden // K, win_len=10,
            read_len=int(spec["output_budget_B"]), d_max=int(spec["T"]) - 1,
            train_mode="weights_only", fixed_delay_value=0.0, readout_type="mlp",
            delay_tying="pre_group",
            **lif_kwargs,
        )
        model.model_backend = "current"
        model.delay_granularity = "input_axon_per_channel"
        return model

    method = spec["delay_method"]
    model = SNNSimultaneousModel(
        n_queries=K, n_hidden=hidden, win_len=10,
        read_len=int(spec["output_budget_B"]), d_max=int(spec["T"]) - 1,
        train_mode="weights_and_delays", delay_param_type="direct",
        delay_tying="pre_group", delay_group_index=list(range(4 * K)),
        delay_init_raw=0.0, n_input_channels=4 * K, readout_type="mlp",
        observation_mode="windowed_shared", output_window_len=int(spec["window_width"]),
        **lif_kwargs,
    )
    model.model_backend = "current"
    model.delay_granularity = "input_axon_per_channel"
    model.delay_operator = "current_interpolated_synapse_with_input_axon_tying"
    if method == "fixed":
        model.syn_ih.delay_raw.data.copy_(
            current_fixed_input_schedule(K, int(spec["window_width"])).reshape(-1, 1)
        )
        model.syn_ih.delay_raw.requires_grad_(False)
    elif method == "d0":
        model.syn_ih.delay_raw.data.zero_()
        model.syn_ih.delay_raw.requires_grad_(False)
    elif method not in {"explicit_centroid", "task_only_current"}:
        raise ValueError(f"unsupported current delay method: {method}")
    return model


def explicit_centroid_loss(model: SNNSimultaneousModel, spec: dict[str, Any]) -> torch.Tensor:
    """Declared teacher loss; only explicit-centroid cells may call this."""
    delays = model.syn_ih.get_delays()
    losses = []
    for query in range(int(spec["K"])):
        query_delay = delays[4 * query:4 * query + 4].mean()
        arrival = 7.5 + 1.0 + query_delay
        target = 10.0 + (query + 0.5) * int(spec["window_width"])
        error = (arrival - target) / float(spec["window_width"])
        losses.append(torch.nn.functional.smooth_l1_loss(
            error, torch.zeros_like(error), beta=0.25
        ))
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_current(
    model: torch.nn.Module, loader: DataLoader, spec: dict[str, Any],
) -> dict[str, Any]:
    model.eval(); encode = _encoder(spec)
    predictions, labels_all, input_events, hidden_events, activities = [], [], [], [], []
    for A, B, _, labels in loader:
        A, B, labels = A.to(spec["device"]), B.to(spec["device"]), labels.to(spec["device"])
        spikes = encode(A, B)
        logits, info = model(spikes, record=True)
        predictions.append((logits > 0).float().cpu()); labels_all.append(labels.cpu())
        input_events.append(spikes.sum(dim=(1, 2)).cpu())
        hidden_events.append(info["total_hidden_spikes"].cpu())
        if "hidden_window_counts" in info:
            activities.append(info["hidden_window_counts"].sum(dim=2))
        elif "readout_features" in info:
            activities.append(info["readout_features"].sum(dim=2))
    predictions_tensor, labels_tensor = torch.cat(predictions), torch.cat(labels_all)
    activity = torch.cat(activities)
    metrics = _reliability_metrics(predictions_tensor, labels_tensor)
    metrics.update({
        "input_events_measured": float(torch.cat(input_events).float().mean()),
        "mean_hidden_spikes": float(torch.cat(hidden_events).float().mean()),
        "per_window_activity": activity.float().mean(dim=0).tolist(),
        "per_window_active_fraction": (activity > 0).float().mean(dim=0).tolist(),
    })
    return metrics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _intervention(
    model: torch.nn.Module, loader: DataLoader, spec: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    if spec["delay_method"] != "task_only_current" or int(spec["K"]) < 2:
        return {"performed": False, "reason": "K2 task-only current cells only"}
    raw = model.syn_ih.delay_raw
    saved = raw.detach().clone()
    output = {}
    for name, value in {
        "zero_delay": torch.zeros_like(saved),
        "query_permutation": saved.reshape(int(spec["K"]), 4, 1).roll(1, dims=0).reshape_as(saved),
    }.items():
        raw.data.copy_(value)
        metrics = evaluate_current(model, loader, spec)
        output[name] = {
            "worst_bacc_drop": float(
                baseline["worst_query_balanced_accuracy"]
                - metrics["worst_query_balanced_accuracy"]
            ),
            "exact_trial_drop": float(
                baseline["exact_trial_accuracy"] - metrics["exact_trial_accuracy"]
            ),
        }
    raw.data.copy_(saved)
    return {"performed": True, "variants": output}


@torch.no_grad()
def _save_diagnostic_panel(
    model: torch.nn.Module, loader: DataLoader, spec: dict[str, Any], output: Path,
) -> None:
    """Save one deterministic validation batch for later temporal-route plots."""
    A, B, _, labels = next(iter(loader))
    A, B = A.to(spec["device"]), B.to(spec["device"])
    spikes = _encoder(spec)(A, B)
    model.eval()
    logits, info = model(spikes, record=True)
    panel = {
        "A": A.cpu().numpy(),
        "B": B.cpu().numpy(),
        "labels": labels.numpy(),
        "input_spike_train": spikes.cpu().numpy(),
        "logits": logits.cpu().numpy(),
        "predictions": (logits > 0).cpu().numpy(),
        "hidden_spike_train": info["hidden_spike_train"].cpu().numpy(),
        "delay_values": model.syn_ih.get_delays().detach().cpu().numpy(),
        "window_edges": np.asarray(
            [10 + query * int(spec["window_width"]) for query in range(int(spec["K"]) + 1)],
            dtype=np.int64,
        ),
    }
    if "hidden_window_counts" in info:
        panel["hidden_window_counts"] = info["hidden_window_counts"].cpu().numpy()
    np.savez_compressed(output / "diagnostic_panel.npz", **panel)


def run_cell(
    protocol: dict[str, Any], spec: dict[str, Any], output: Path, *, device: str,
) -> dict[str, Any]:
    if spec["model_backend"] != "current":
        raise ValueError("current runner received a non-current cell")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    spec = {**spec, "device": device}
    (output / "resolved_config.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    torch.manual_seed(int(spec["seed"]))
    model = build_current_model(spec).to(device)
    operations = list(spec.get("query_ops", ["XOR"] * int(spec["K"])))
    batch_size = int(protocol["optimization"]["batch_size"])
    train_loader = _loader(
        int(protocol["data"]["train_samples"]), operations, int(spec["seed"]),
        batch_size, True,
    )
    validation_loader = _loader(
        int(protocol["data"]["validation_samples"]), operations,
        int(spec["seed"]) + 100000, batch_size, False,
    )
    groups = [
        {"params": model.weight_params(), "lr": 0.001},
        {"params": model.readout_params(), "lr": 0.001},
    ]
    if model.delay_params():
        groups.append({"params": model.delay_params(), "lr": 0.001})
    optimizer = torch.optim.Adam(groups)
    encode = _encoder(spec); iterator = iter(train_loader)
    history, best_score = [], (-math.inf, -math.inf)
    best_path = output / "best_model.pt"
    total = int(protocol["optimization"]["optimizer_updates"])
    interval = int(protocol["optimization"]["validation_interval_updates"])
    for update in range(1, total + 1):
        try:
            A, B, _, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader); A, B, _, labels = next(iterator)
        A, B, labels = A.to(device), B.to(device), labels.to(device)
        spikes = encode(A, B); model.train(); optimizer.zero_grad(set_to_none=True)
        logits, info = model(spikes)
        task_loss = window_class_balanced_bce(logits, labels)
        route_loss = torch.zeros((), device=device)
        if spec["delay_method"] == "explicit_centroid":
            route_loss = explicit_centroid_loss(model, spec)
            task_loss.backward(retain_graph=True)
            for parameter in model.delay_params():
                parameter.grad = None
            route_loss.backward()
            loss = task_loss + route_loss
        else:
            loss = task_loss; loss.backward()
        delay_gradient = math.sqrt(sum(
            float(parameter.grad.detach().square().sum())
            for parameter in model.delay_params() if parameter.grad is not None
        ))
        global_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if update % interval == 0 or update == total:
            validation = evaluate_current(model, validation_loader, spec)
            row = {
                "update": update, "train_loss": float(loss.detach()),
                "task_loss": float(task_loss.detach()),
                "routing_loss": float(route_loss.detach()),
                "delay_gradient_norm": delay_gradient,
                "global_gradient_norm_before_clip": global_norm,
                "val_worst_query_balanced_accuracy": validation["worst_query_balanced_accuracy"],
                "val_exact_trial_accuracy": validation["exact_trial_accuracy"],
            }
            history.append(row)
            score = (validation["worst_query_balanced_accuracy"], validation["exact_trial_accuracy"])
            if score > best_score:
                best_score = score
                torch.save({"model": model.state_dict(), "update": update}, best_path)
    _write_csv(output / "training_log.csv", history)
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    final = evaluate_current(model, validation_loader, spec)
    intervention = _intervention(model, validation_loader, spec, final)
    if spec.get("stage") == "temporal":
        _save_diagnostic_panel(model, validation_loader, spec, output)
    ledger = static_resource_ledger(model)
    ledger.update(dynamic_resource_ledger(
        model, mean_input_spikes=final["input_events_measured"],
        mean_hidden1_spikes=final["mean_hidden_spikes"],
    ))
    result = make_result_record(
        model_backend="current", delay_method=spec["delay_method"],
        delay_granularity="input_axon_per_channel", encoding_mode="binary_one_hot_packet",
        output_interface="windowed_shared_mlp", K=int(spec["K"]),
        N_hidden_total=int(spec["N_hidden_total"]), output_budget_B=int(spec["output_budget_B"]),
        window_width=int(spec["window_width"]), T=int(spec["T"]),
        input_events_expected=int(spec["input_events_expected"]),
        input_events_measured=final["input_events_measured"],
        delay_parameter_count=sum(p.numel() for p in model.delay_params()),
        delay_storage_count=ledger["delay_value_storage_elements"],
        worst_query_balanced_accuracy=final["worst_query_balanced_accuracy"],
        exact_trial_accuracy=final["exact_trial_accuracy"],
        per_query_balanced_accuracy=final["per_query_balanced_accuracy"],
        per_window_activity={"mean_spikes": final["per_window_activity"],
                             "active_fraction": final["per_window_active_fraction"]},
        delay_intervention_metrics=intervention, resource_ledger=ledger,
    )
    result.update({
        "seed": int(spec["seed"]), "condition": spec["condition"],
        "pass_reliability": reliability_pass(result),
        "selected_checkpoint_update": int(checkpoint["update"]),
        "test_split_opened": False,
    })
    (output / "validation_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "resource_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return result

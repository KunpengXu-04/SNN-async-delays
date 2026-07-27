"""Task-only SLAYER S1/S2 cell runner for the registered capacity protocol.

This module contains no centroid or oracle loss. A fixed schedule may be
supplied only for the separately labelled positive-control interface cell.
"""

from __future__ import annotations

import csv
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.boolean_dataset import MarginallyBalancedFixedOperationQueryDataset
from data.encoding import encode_simultaneous_trial
from snn.slayer_backend import SlayerNativeWindowedModel
from train.eval import _reliability_metrics
from train.trainer import window_class_balanced_bce
from utils.capacity_scaling import make_result_record, reliability_pass
from utils.resource_ledger import dynamic_resource_ledger, static_resource_ledger


def fixed_input_axon_schedule(K: int, window_width: int) -> torch.Tensor:
    """Route packet start step 6 to each output-window start step 10+q*w."""
    return torch.repeat_interleave(
        torch.tensor([4 + query * window_width for query in range(K)], dtype=torch.float32),
        4,
    )


def _encoder(spec: dict[str, Any]):
    return partial(
        encode_simultaneous_trial,
        encoding_mode="binary_one_hot_packet",
        packet_start_step=6,
        packet_steps=4,
        win_len=10,
        read_len=int(spec["output_budget_B"]),
        device=spec["device"],
    )


def _loader(n: int, K: int, seed: int, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = MarginallyBalancedFixedOperationQueryDataset(
        n, ["XOR"] * K, seed=seed
    )
    generator = torch.Generator().manual_seed(seed + 101)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def _build_model(spec: dict[str, Any]) -> SlayerNativeWindowedModel:
    method = spec["delay_method"]
    fixed = None
    if method == "fixed":
        fixed = fixed_input_axon_schedule(int(spec["K"]), int(spec["window_width"]))
    return SlayerNativeWindowedModel(
        n_queries=int(spec["K"]), n_hidden=int(spec["N_hidden_total"]),
        win_len=10, read_len=int(spec["output_budget_B"]),
        d_max=int(spec["T"]) - 1, delay_method=method,
        delay_init=float(spec.get("delay_init", 0.0)), fixed_delays=fixed,
    ).to(spec["device"])


@torch.no_grad()
def evaluate_cell(
    model: SlayerNativeWindowedModel, loader: DataLoader,
    spec: dict[str, Any], *, collect: bool = False,
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    model.eval()
    encode = _encoder(spec)
    predictions, labels_all, logits_all = [], [], []
    input_events, hidden_events = [], []
    window_activity = []
    for A, B, _, labels in loader:
        A, B, labels = A.to(spec["device"]), B.to(spec["device"]), labels.to(spec["device"])
        spikes = encode(A, B)
        logits, info = model(spikes)
        predictions.append((logits > 0).float().cpu())
        labels_all.append(labels.cpu())
        logits_all.append(logits.cpu())
        input_events.append(spikes.sum(dim=(1, 2)).cpu())
        hidden_events.append(info["total_hidden_spikes"].cpu())
        window_activity.append(info["hidden_window_counts"].sum(dim=2))
    predictions_tensor = torch.cat(predictions)
    labels_tensor = torch.cat(labels_all)
    input_tensor = torch.cat(input_events)
    hidden_tensor = torch.cat(hidden_events)
    activity_tensor = torch.cat(window_activity)
    metrics = _reliability_metrics(predictions_tensor, labels_tensor)
    metrics.update({
        "input_events_measured": float(input_tensor.mean()),
        "mean_hidden_spikes": float(hidden_tensor.float().mean()),
        "per_window_activity": activity_tensor.float().mean(dim=0).tolist(),
        "per_window_active_fraction": (activity_tensor > 0).float().mean(dim=0).tolist(),
    })
    arrays = None
    if collect:
        arrays = {
            "predictions": predictions_tensor,
            "labels": labels_tensor,
            "logits": torch.cat(logits_all),
        }
    return metrics, arrays


def _interventions(
    model: SlayerNativeWindowedModel, loader: DataLoader,
    spec: dict[str, Any], baseline: dict[str, Any],
) -> dict[str, Any]:
    if int(spec["K"]) < 2 or spec["delay_method"] != "task_only_slayer":
        return {"performed": False, "reason": "S2 task-only cells only"}
    saved = model.axon_delay.delay.detach().clone()
    variants: dict[str, Any] = {}
    for name, values in {
        "zero_delay": torch.zeros_like(saved),
        "query_permutation": saved.reshape(int(spec["K"]), 4).roll(1, dims=0).flatten(),
    }.items():
        model.axon_delay.delay.data.copy_(values)
        metrics, _ = evaluate_cell(model, loader, spec)
        variants[name] = {
            "worst_query_balanced_accuracy": metrics["worst_query_balanced_accuracy"],
            "exact_trial_accuracy": metrics["exact_trial_accuracy"],
            "worst_bacc_drop": float(
                baseline["worst_query_balanced_accuracy"]
                - metrics["worst_query_balanced_accuracy"]
            ),
            "exact_trial_drop": float(
                baseline["exact_trial_accuracy"] - metrics["exact_trial_accuracy"]
            ),
        }
    model.axon_delay.delay.data.copy_(saved)
    return {"performed": True, "variants": variants}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_cell(
    protocol: dict[str, Any], spec: dict[str, Any], output: Path,
    *, device: str,
) -> dict[str, Any]:
    """Train one immutable SLAYER cell and emit the common result schema."""
    if spec["model_backend"] != "slayer_native":
        raise ValueError("this runner accepts only slayer_native cells")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed or partial cell: {output}")
    output.mkdir(parents=True)
    spec = {**spec, "device": device, "delay_init": 0.0}
    (output / "resolved_config.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    torch.manual_seed(int(spec["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(spec["seed"]))
    model = _build_model(spec)
    batch_size = int(protocol["optimization"]["batch_size"])
    train_loader = _loader(
        int(protocol["data"]["train_samples"]), int(spec["K"]),
        int(spec["seed"]), batch_size, True,
    )
    validation_loader = _loader(
        int(protocol["data"]["validation_samples"]), int(spec["K"]),
        int(spec["seed"]) + 100000, batch_size, False,
    )
    lr_w = float(spec.get("weight_learning_rate", 0.001))
    lr_d = float(spec.get("delay_learning_rate", 0.001))
    groups = [
        {"params": model.weight_params(), "lr": lr_w},
        {"params": model.readout_params(), "lr": lr_w},
    ]
    if model.delay_params():
        groups.append({"params": model.delay_params(), "lr": lr_d})
    optimizer = torch.optim.Adam(groups)
    total_updates = int(protocol["optimization"]["optimizer_updates"])
    interval = int(protocol["optimization"]["validation_interval_updates"])
    clip = float(protocol["optimization"]["gradient_clip_norm"])
    iterator = iter(train_loader)
    history: list[dict[str, Any]] = []
    best_score = (-math.inf, -math.inf)
    best_path = output / "best_model.pt"
    encode = _encoder(spec)
    for update in range(1, total_updates + 1):
        try:
            A, B, _, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            A, B, _, labels = next(iterator)
        A, B, labels = A.to(device), B.to(device), labels.to(device)
        spikes = encode(A, B)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, info = model(spikes)
        loss = window_class_balanced_bce(logits, labels)
        loss.backward()
        delay_gradient = (
            float(model.axon_delay.delay.grad.detach().norm().cpu())
            if model.axon_delay.delay.grad is not None else 0.0
        )
        global_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip).cpu())
        optimizer.step()
        model.axon_delay.clamp()
        if update % interval == 0 or update == total_updates:
            validation, _ = evaluate_cell(model, validation_loader, spec)
            row = {
                "update": update, "train_loss": float(loss.detach().cpu()),
                "train_hidden_spikes": float(info["total_hidden_spikes"].mean().detach().cpu()),
                "delay_gradient_norm": delay_gradient,
                "global_gradient_norm_before_clip": global_norm,
                "val_worst_query_balanced_accuracy": validation["worst_query_balanced_accuracy"],
                "val_exact_trial_accuracy": validation["exact_trial_accuracy"],
            }
            history.append(row)
            score = (
                float(validation["worst_query_balanced_accuracy"]),
                float(validation["exact_trial_accuracy"]),
            )
            if score > best_score:
                best_score = score
                torch.save({"model": model.state_dict(), "spec": spec, "update": update}, best_path)
    _write_csv(output / "training_log.csv", history)
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    final, arrays = evaluate_cell(model, validation_loader, spec, collect=True)
    intervention = _interventions(model, validation_loader, spec, final)
    ledger = static_resource_ledger(model)
    ledger.update(dynamic_resource_ledger(
        model, mean_input_spikes=final["input_events_measured"],
        mean_hidden1_spikes=final["mean_hidden_spikes"],
    ))
    result = make_result_record(
        model_backend="slayer_native", delay_method=spec["delay_method"],
        delay_granularity="input_axon_per_channel",
        encoding_mode="binary_one_hot_packet",
        output_interface="windowed_shared_mlp", K=int(spec["K"]),
        N_hidden_total=int(spec["N_hidden_total"]),
        output_budget_B=int(spec["output_budget_B"]),
        window_width=int(spec["window_width"]), T=int(spec["T"]),
        input_events_expected=int(spec["input_events_expected"]),
        input_events_measured=final["input_events_measured"],
        delay_parameter_count=sum(p.numel() for p in model.delay_params()),
        delay_storage_count=ledger["delay_value_storage_elements"],
        worst_query_balanced_accuracy=final["worst_query_balanced_accuracy"],
        exact_trial_accuracy=final["exact_trial_accuracy"],
        per_query_balanced_accuracy=final["per_query_balanced_accuracy"],
        per_window_activity={
            "mean_spikes": final["per_window_activity"],
            "active_fraction": final["per_window_active_fraction"],
        },
        delay_intervention_metrics=intervention, resource_ledger=ledger,
    )
    result.update({
        "seed": int(spec["seed"]), "condition": spec["condition"],
        "pass_reliability": reliability_pass(result),
        "selected_checkpoint_update": int(checkpoint["update"]),
        "test_split_opened": False,
    })
    (output / "validation_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    torch.save(arrays, output / "validation_predictions.pt")
    (output / "resource_ledger.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8"
    )
    return result

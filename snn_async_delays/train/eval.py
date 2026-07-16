"""
Evaluation metrics for all three steps.

Key metrics
-----------
Step 1 : per-op accuracy, convergence epochs, spike count
Step 2/3: max_K at accuracy >= tau, energy-normalized throughput K/spikes,
          density metrics (ops/neuron/time), and mixed-op diagnostics.
"""

from __future__ import annotations
import json
import os
from typing import List, Dict

import torch
from torch.utils.data import DataLoader

from snn.model import SNNModel, SNNSimultaneousModel, SlotBoundaries
from data.encoding import encode_trial, encode_simultaneous_trial
from utils.resource_ledger import static_resource_ledger, dynamic_resource_ledger


def _binary_confusion(preds_flat: torch.Tensor, labels_flat: torch.Tensor) -> list[list[int]]:
    preds_b = preds_flat.int()
    labels_b = labels_flat.int()
    tn = int(((preds_b == 0) & (labels_b == 0)).sum().item())
    fp = int(((preds_b == 1) & (labels_b == 0)).sum().item())
    fn = int(((preds_b == 0) & (labels_b == 1)).sum().item())
    tp = int(((preds_b == 1) & (labels_b == 1)).sum().item())
    return [[tn, fp], [fn, tp]]


def _reliability_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict:
    """Compute position-sensitive reliability metrics for a [trial, query] task.

    Pooled binary accuracy is retained for backward compatibility, but it is
    not an adequate multiplexing endpoint: a model can pass it while failing
    its earliest query positions.  ``None`` denotes an undefined balanced
    accuracy (a split containing one class only), avoiding non-standard JSON
    ``NaN`` values.
    """
    correct = (preds == labels).float()
    per_query = correct.mean(dim=0)
    balanced_per_query: list[float | None] = []
    for query in range(labels.shape[1]):
        y = labels[:, query]
        c = correct[:, query]
        positive = y == 1
        negative = y == 0
        if not bool(positive.any()) or not bool(negative.any()):
            balanced_per_query.append(None)
            continue
        tpr = c[positive].mean()
        tnr = c[negative].mean()
        balanced_per_query.append(float(((tpr + tnr) / 2).item()))

    valid_balanced = [value for value in balanced_per_query if value is not None]
    return {
        "accuracy": float(correct.mean().item()),
        "pooled_accuracy": float(correct.mean().item()),
        "per_query_acc": per_query.tolist(),
        "worst_query_accuracy": float(per_query.min().item()),
        "exact_trial_accuracy": float(correct.all(dim=1).float().mean().item()),
        "per_query_balanced_accuracy": balanced_per_query,
        "balanced_accuracy": (
            float(sum(valid_balanced) / len(valid_balanced))
            if valid_balanced else None
        ),
        "worst_query_balanced_accuracy": (
            float(min(valid_balanced)) if valid_balanced else None
        ),
    }


@torch.no_grad()
def evaluate(
    model: SNNModel,
    loader: DataLoader,
    slots: List[SlotBoundaries],
    cfg: Dict,
    device: str = "cpu",
) -> Dict:
    """
    Full evaluation pass.

    Returns keys:
      accuracy, per_query_acc, mean_hidden_spikes, throughput_K_per_spk,
      ops_per_neuron_per_ms, mean_active_hidden_fraction,
      binary_confusion, op_accuracy (if mixed-op metadata available).
    """
    model.eval()
    K = len(slots)

    all_preds = []
    all_labels = []
    all_h_spk = []
    all_active_frac = []
    all_op_ids = []

    for A, B, op_ids, labels in loader:
        if labels.dim() == 1:
            A, B, op_ids, labels = (
                A.unsqueeze(1), B.unsqueeze(1),
                op_ids.unsqueeze(1), labels.unsqueeze(1),
            )

        A = A.to(device)
        B = B.to(device)
        op_ids = op_ids.to(device)
        labels = labels.to(device)

        spike_input = encode_trial(
            A,
            B,
            op_ids,
            slots,
            n_input=model.n_input,
            r_on=cfg["r_on"],
            r_off=cfg["r_off"],
            dt=cfg["dt"],
            device=device,
        )
        logits, info = model(spike_input, slots)

        preds = (logits > 0).float()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_h_spk.append(info["total_hidden_spikes"].cpu())
        all_active_frac.append(info["active_hidden_fraction"].cpu())
        all_op_ids.append(op_ids.cpu())

    preds = torch.cat(all_preds, dim=0)       # [N, K]
    labels = torch.cat(all_labels, dim=0)     # [N, K]
    h_spk = torch.cat(all_h_spk, dim=0)       # [N]
    active_frac = torch.cat(all_active_frac, dim=0)  # [N]
    op_ids_all = torch.cat(all_op_ids, dim=0)         # [N, K]

    reliability = _reliability_metrics(preds, labels)
    mean_h_spk = h_spk.mean().item()

    throughput = K / mean_h_spk if mean_h_spk > 0 else None

    T_steps = slots[-1].read_end
    trial_ms = T_steps * float(cfg["dt"])
    ops_per_neuron_per_ms = K / (max(model.n_hidden, 1) * max(trial_ms, 1e-9))

    preds_flat = preds.reshape(-1)
    labels_flat = labels.reshape(-1)
    binary_conf = _binary_confusion(preds_flat, labels_flat)

    op_accuracy = {}
    op_confusions = {}
    ops_list = cfg.get("ops_list")
    if ops_list is not None:
        op_ids_flat = op_ids_all.reshape(-1)
        correct_flat = (preds_flat == labels_flat).float()
        for op_idx, op_name in enumerate(ops_list):
            mask = (op_ids_flat == op_idx)
            n = int(mask.sum().item())
            if n == 0:
                continue
            acc = float(correct_flat[mask].mean().item())
            op_accuracy[op_name] = acc
            op_confusions[op_name] = _binary_confusion(preds_flat[mask], labels_flat[mask])

    return {
        **reliability,
        "mean_hidden_spikes": mean_h_spk,
        "throughput_K_per_spk": throughput,
        "ops_per_neuron_per_ms": ops_per_neuron_per_ms,
        "mean_active_hidden_fraction": float(active_frac.mean().item()),
        "binary_confusion": binary_conf,
        "op_accuracy": op_accuracy,
        "op_confusions": op_confusions,
        "K": K,
    }


def max_K_at_threshold(
    results_by_K: Dict[int, Dict],
    tau: float = 0.95,
    metric: str = "worst_query_accuracy",
) -> int:
    """
    Given a dict {K: eval_result}, return the largest K where the declared
    position-sensitive metric reaches ``tau``.  The default deliberately
    rejects historical pooled-accuracy summaries that lack the new field.
    Pass ``metric=\"accuracy\"`` only for explicitly labelled legacy analysis.
    """
    max_k = 0
    for K, res in sorted(results_by_K.items()):
        if metric not in res:
            raise KeyError(
                f"Result for K={K} lacks {metric!r}; do not silently fall back "
                "to pooled accuracy."
            )
        if res[metric] is not None and res[metric] >= tau:
            max_k = K
    return max_k


def save_eval_results(results: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        # JSON containing NaN is not portable and previously made tests and
        # downstream aggregation disagree.  Evaluation must emit a defined
        # value (normally ``None``) or fail loudly at the producing run.
        json.dump(results, f, indent=2, default=str, allow_nan=False)


def load_eval_results(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def evaluate_simultaneous(
    model: SNNSimultaneousModel,
    loader: DataLoader,
    cfg: Dict,
    device: str = "cpu",
    encode_fn=None,
    K_query: int | None = None,
    return_trial_records: bool = False,
) -> Dict:
    """
    Evaluation for SNNSimultaneousModel (true temporal multiplexing).

    Returns the same keys as evaluate() for direct comparison.

    K_query : actual input multiplexing load (number of sequential
        sub-windows), used for the K / throughput_K_per_spk /
        ops_per_neuron_per_ms fields. Defaults to model.n_queries (the
        readout output dimension), which is correct whenever input
        sub-windows and output logits are in 1:1 correspondence (Step 2/3).
        Aggregate-output topologies (many-query-one-out) decouple the two
        (model.n_queries=1 while K_query>1) and must pass it explicitly.
    """
    model.eval()
    K = K_query if K_query is not None else model.n_queries

    all_preds, all_labels, all_h_spk, all_active_frac = [], [], [], []
    all_input_spk: list = []
    all_output_spk: list = []
    all_h1_spk: list = []
    all_h2_spk: list = []
    all_output_window_collisions: list = []
    all_logits: list = []
    all_opponent_counts: list = []
    all_hidden_window_counts: list = []
    all_A: list = []
    all_B: list = []
    all_target_timing_hits: list = []
    all_target_timing_errors: list = []

    for A, B, op_ids, labels in loader:
        if labels.dim() == 1:
            A, B, op_ids, labels = (
                A.unsqueeze(1), B.unsqueeze(1),
                op_ids.unsqueeze(1), labels.unsqueeze(1),
            )

        A      = A.to(device)
        B      = B.to(device)
        labels = labels.to(device)

        _encode = encode_fn if encode_fn is not None else encode_simultaneous_trial
        spike_input = _encode(
            A, B,
            win_len=cfg["win_len"],
            read_len=cfg["read_len"],
            r_on=cfg["r_on"],
            r_off=cfg["r_off"],
            dt=cfg["dt"],
            device=device,
            op_ids=op_ids.to(device),
            n_ops=cfg.get("n_ops", 0),
        )
        target_timing_mode = cfg.get("opponent_target_timing_mode")
        logits, info = model(
            spike_input,
            return_output_spike_train=target_timing_mode is not None,
        )

        if target_timing_mode is not None:
            output_train = info["output_spike_train"]
            tolerance = int(cfg.get("target_timing_tolerance_steps", 2))
            window_len = int(cfg["output_window_len"])
            for query in range(labels.shape[1]):
                sequential = target_timing_mode in {"sequential_centers", "sequential_offsets"}
                offset = query * window_len if sequential else 0
                within_window = (
                    window_len // 2
                    if target_timing_mode in {"simultaneous_center", "sequential_centers"}
                    else int(cfg.get("output_target_offset_steps", 0))
                )
                target_time = int(cfg["win_len"]) + offset + within_window
                correct_channel = labels[:, query].long().clamp(0, 1) + 2 * query
                per_sample = output_train[
                    torch.arange(labels.shape[0], device=labels.device), :, correct_channel
                ]
                lo, hi = max(0, target_time - tolerance), min(model.T, target_time + tolerance + 1)
                all_target_timing_hits.append((per_sample[:, lo:hi].sum(dim=1) > 0).cpu())
                for sample in range(per_sample.shape[0]):
                    spike_times = torch.nonzero(per_sample[sample] > 0, as_tuple=True)[0]
                    if spike_times.numel():
                        all_target_timing_errors.append(
                            float((spike_times.float() - target_time).abs().min().item())
                        )

        preds = (logits > 0).float()
        all_preds.append(preds.cpu())
        all_logits.append(logits.detach().cpu())
        if return_trial_records:
            all_A.append(A.cpu()); all_B.append(B.cpu())
        all_labels.append(labels.cpu())
        all_h_spk.append(info["total_hidden_spikes"].cpu())
        all_active_frac.append(info["active_hidden_fraction"].cpu())
        all_input_spk.append(spike_input.sum(dim=(1, 2)).cpu())
        if "total_output_spikes" in info:
            all_output_spk.append(info["total_output_spikes"].cpu())
        if "hidden_window_counts" in info:
            all_hidden_window_counts.append(info["hidden_window_counts"])
        if "output_window_counts" in info:
            counts = info["output_window_counts"]
            all_opponent_counts.append(counts)
            all_output_window_collisions.append(
                ((counts[:, :, 0] > 0) & (counts[:, :, 1] > 0)).float().reshape(-1)
            )
        elif "output_pair_counts" in info:
            counts = info["output_pair_counts"]
            all_opponent_counts.append(counts)
            all_output_window_collisions.append(
                ((counts[:, :, 0] > 0) & (counts[:, :, 1] > 0)).float().reshape(-1)
            )

        if "layer1_hidden_spikes" in info:
            all_h1_spk.append(info["layer1_hidden_spikes"].cpu())
            all_h2_spk.append(info["layer2_hidden_spikes"].cpu())

    preds  = torch.cat(all_preds,  dim=0)   # [N, K]
    labels = torch.cat(all_labels, dim=0)   # [N, K]
    logits_all = torch.cat(all_logits, dim=0)
    h_spk  = torch.cat(all_h_spk,  dim=0)  # [N]
    active_frac = torch.cat(all_active_frac, dim=0)
    input_spk = torch.cat(all_input_spk, dim=0)

    reliability = _reliability_metrics(preds, labels)
    mean_h_spk   = h_spk.mean().item()
    throughput    = K / mean_h_spk if mean_h_spk > 0 else None

    T_steps    = model.T
    trial_ms   = T_steps * float(cfg["dt"])
    # Use total neuron count (both layers) for ops/neuron metric
    n_neurons_total = getattr(model, "n_hidden_total", model.n_hidden)
    ops_per_neuron_per_ms = K / (max(n_neurons_total, 1) * max(trial_ms, 1e-9))

    preds_flat  = preds.reshape(-1)
    labels_flat = labels.reshape(-1)
    binary_conf = _binary_confusion(preds_flat, labels_flat)

    result = {
        **reliability,
        "mean_hidden_spikes":      mean_h_spk,
        "throughput_K_per_spk":    throughput,
        "ops_per_neuron_per_ms":   ops_per_neuron_per_ms,
        "mean_active_hidden_fraction": float(active_frac.mean().item()),
        "binary_confusion":        binary_conf,
        "K":                       K,
    }
    result["output_window_collision_rate"] = (
        float(torch.cat(all_output_window_collisions).mean().item())
        if all_output_window_collisions else None
    )
    result["target_timing_hit_rate"] = (
        float(torch.cat(all_target_timing_hits).float().mean().item())
        if all_target_timing_hits else None
    )
    result["mean_abs_target_timing_error_steps"] = (
        float(sum(all_target_timing_errors) / len(all_target_timing_errors))
        if all_target_timing_errors else None
    )
    if all_hidden_window_counts:
        hidden_window_counts = torch.cat(all_hidden_window_counts, dim=0).float()
        hidden_window_totals = hidden_window_counts.sum(dim=2)
        result["per_query_hidden_window_spikes"] = (
            hidden_window_totals.mean(dim=0).tolist()
        )
        result["per_query_hidden_window_activity_fraction"] = (
            (hidden_window_totals > 0).float().mean(dim=0).tolist()
        )
    else:
        result["per_query_hidden_window_spikes"] = None
        result["per_query_hidden_window_activity_fraction"] = None
    if all_opponent_counts:
        counts = torch.cat(all_opponent_counts, dim=0).float()
        differences = counts[:, :, 1] - counts[:, :, 0]
        signed_targets = labels * 2 - 1
        result.update({
            "output_silent_rate": float((counts.sum(dim=2) == 0).float().mean().item()),
            "output_tie_rate": float((differences == 0).float().mean().item()),
            "output_collision_rate": float(((counts[:, :, 0] > 0) & (counts[:, :, 1] > 0)).float().mean().item()),
            "per_query_output_spikes": counts.sum(dim=2).mean(dim=0).tolist(),
            "per_query_temporal_margin": (signed_targets * differences).mean(dim=0).tolist(),
            "worst_query_temporal_margin": float((signed_targets * differences).mean(dim=0).min().item()),
        })
    else:
        result.update({"output_silent_rate": None, "output_tie_rate": None,
                       "output_collision_rate": None, "per_query_output_spikes": None,
                       "per_query_temporal_margin": None, "worst_query_temporal_margin": None})

    cross_balanced = []
    for output_idx in range(preds.shape[1]):
        row = []
        for target_idx in range(labels.shape[1]):
            metrics = _reliability_metrics(preds[:, output_idx:output_idx + 1],
                                           labels[:, target_idx:target_idx + 1])
            row.append(metrics["balanced_accuracy"])
        cross_balanced.append(row)
    diagonal = [cross_balanced[k][k] for k in range(min(len(cross_balanced), labels.shape[1]))]
    off_diagonal = [cross_balanced[i][j] for i in range(len(cross_balanced))
                    for j in range(labels.shape[1]) if i != j and cross_balanced[i][j] is not None]
    result["cross_target_balanced_accuracy_matrix"] = cross_balanced
    result["routing_selectivity_gap"] = (
        float(sum(diagonal) / len(diagonal) - sum(off_diagonal) / len(off_diagonal))
        if diagonal and off_diagonal and all(x is not None for x in diagonal) else None
    )
    if return_trial_records:
        A_all, B_all = torch.cat(all_A), torch.cat(all_B)
        result["trial_records"] = [
            {"A": A_all[i].tolist(), "B": B_all[i].tolist(),
             "labels": labels[i].tolist(), "logits": logits_all[i].tolist(),
             "predictions": preds[i].tolist(),
             "exact_correct": bool((preds[i] == labels[i]).all().item())}
            for i in range(len(labels))
        ]
        result["truth_table_patterns"] = len(labels)
        result["truth_table_patterns_exact_correct"] = int((preds == labels).all(dim=1).sum().item())
        result["exact_truth_table_completion"] = bool((preds == labels).all().item())
    if hasattr(model, "observation_metadata"):
        result.update(model.observation_metadata())

    # Layer-wise spike stats (only present for 2-layer models)
    if all_h1_spk:
        h1_t = torch.cat(all_h1_spk, dim=0)
        h2_t = torch.cat(all_h2_spk, dim=0)
        result["layer1_hidden_spikes"] = float(h1_t.mean().item())
        result["layer2_hidden_spikes"] = float(h2_t.mean().item())
        mean_h1 = result["layer1_hidden_spikes"]
        mean_h2 = result["layer2_hidden_spikes"]
    else:
        result["layer1_hidden_spikes"] = None
        result["layer2_hidden_spikes"] = None
        mean_h1 = mean_h_spk
        mean_h2 = None

    mean_output = (
        float(torch.cat(all_output_spk, dim=0).mean().item())
        if all_output_spk else None
    )
    ledger = static_resource_ledger(model)
    ledger.update(dynamic_resource_ledger(
        model,
        mean_input_spikes=float(input_spk.mean().item()),
        mean_hidden1_spikes=float(mean_h1),
        mean_hidden2_spikes=(float(mean_h2) if mean_h2 is not None else None),
        mean_output_spikes=mean_output,
    ))
    result["resource_ledger"] = ledger

    return result


def summarize_sweep(results: List[Dict], key_fields: List[str] = None) -> str:
    """
    Pretty-print a list of result dicts as a markdown-like table.
    key_fields : columns to show (default: all keys)
    """
    if not results:
        return "(empty)"
    if key_fields is None:
        key_fields = list(results[0].keys())

    header = " | ".join(f"{k:>20}" for k in key_fields)
    sep = "-+-".join("-" * 20 for _ in key_fields)
    rows = []
    for r in results:
        row = " | ".join(f"{str(r.get(k, ''))[:20]:>20}" for k in key_fields)
        rows.append(row)

    return "\n".join([header, sep] + rows)

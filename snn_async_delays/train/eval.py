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

from snn.model import SNNModel, SlotBoundaries
from data.encoding import encode_trial


def _binary_confusion(preds_flat: torch.Tensor, labels_flat: torch.Tensor) -> list[list[int]]:
    preds_b = preds_flat.int()
    labels_b = labels_flat.int()
    tn = int(((preds_b == 0) & (labels_b == 0)).sum().item())
    fp = int(((preds_b == 1) & (labels_b == 0)).sum().item())
    fn = int(((preds_b == 0) & (labels_b == 1)).sum().item())
    tp = int(((preds_b == 1) & (labels_b == 1)).sum().item())
    return [[tn, fp], [fn, tp]]


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

    correct = (preds == labels).float()
    overall_acc = correct.mean().item()
    per_query_acc = correct.mean(dim=0).tolist()
    mean_h_spk = h_spk.mean().item()

    throughput = K / mean_h_spk if mean_h_spk > 0 else float("nan")

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
        "accuracy": overall_acc,
        "per_query_acc": per_query_acc,
        "mean_hidden_spikes": mean_h_spk,
        "throughput_K_per_spk": throughput,
        "ops_per_neuron_per_ms": ops_per_neuron_per_ms,
        "mean_active_hidden_fraction": float(active_frac.mean().item()),
        "binary_confusion": binary_conf,
        "op_accuracy": op_accuracy,
        "op_confusions": op_confusions,
        "K": K,
    }


def max_K_at_threshold(results_by_K: Dict[int, Dict], tau: float = 0.95) -> int:
    """
    Given a dict {K: eval_result}, return the largest K
    where accuracy >= tau. Returns 0 if none found.
    """
    max_k = 0
    for K, res in sorted(results_by_K.items()):
        if res["accuracy"] >= tau:
            max_k = K
    return max_k


def save_eval_results(results: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)


def load_eval_results(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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

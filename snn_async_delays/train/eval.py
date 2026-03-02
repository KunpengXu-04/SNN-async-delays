"""
Evaluation metrics for all three steps.

Key metrics
-----------
Step 1 : per-op accuracy, convergence epochs, spike count
Step 2/3: max_K @ accuracy >= tau, energy-normalized throughput K/spikes,
          per-query latency (estimated)
"""

from __future__ import annotations
import json
import os
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from snn.model import SNNModel, SlotBoundaries
from data.encoding import encode_trial


# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:   SNNModel,
    loader:  DataLoader,
    slots:   List[SlotBoundaries],
    cfg:     Dict,
    device:  str = "cpu",
) -> Dict:
    """
    Full evaluation pass.

    Returns dict with keys:
        accuracy, mean_hidden_spikes, mean_output_spikes,
        throughput (K / mean_hidden_spikes),
        per_query_acc [list of K floats]
    """
    model.eval()
    K = len(slots)

    all_preds  = []   # [N, K]
    all_labels = []   # [N, K]
    all_h_spk  = []   # [N]

    for A, B, op_ids, labels in loader:
        if labels.dim() == 1:
            A, B, op_ids, labels = (
                A.unsqueeze(1), B.unsqueeze(1),
                op_ids.unsqueeze(1), labels.unsqueeze(1),
            )
        A = A.to(device); B = B.to(device)
        op_ids = op_ids.to(device); labels = labels.to(device)

        spike_input = encode_trial(
            A, B, op_ids, slots,
            n_input=model.n_input,
            r_on=cfg["r_on"],
            r_off=cfg["r_off"],
            dt=cfg["dt"],
            device=device,
        )
        logits, info = model(spike_input, slots)   # [B, K]

        preds = (logits > 0).float()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_h_spk.append(info["total_hidden_spikes"].cpu())

    preds  = torch.cat(all_preds,  dim=0)   # [N, K]
    labels = torch.cat(all_labels, dim=0)   # [N, K]
    h_spk  = torch.cat(all_h_spk,  dim=0)  # [N]

    correct = (preds == labels).float()
    overall_acc    = correct.mean().item()
    per_query_acc  = correct.mean(dim=0).tolist()    # length K
    mean_h_spk     = h_spk.mean().item()

    throughput = K / mean_h_spk if mean_h_spk > 0 else float("nan")

    return {
        "accuracy":           overall_acc,
        "per_query_acc":      per_query_acc,
        "mean_hidden_spikes": mean_h_spk,
        "throughput_K_per_spk": throughput,
        "K":                  K,
    }


# ---------------------------------------------------------------------------

def max_K_at_threshold(
    results_by_K: Dict[int, Dict],
    tau: float = 0.95,
) -> int:
    """
    Given a dict {K: eval_result}, return the largest K
    where accuracy >= tau.  Returns 0 if none found.
    """
    max_k = 0
    for K, res in sorted(results_by_K.items()):
        if res["accuracy"] >= tau:
            max_k = K
    return max_k


# ---------------------------------------------------------------------------

def save_eval_results(results: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_eval_results(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------

def summarize_sweep(
    results: List[Dict],
    key_fields: List[str] = None,
) -> str:
    """
    Pretty-print a list of result dicts as a markdown table.
    key_fields : columns to show (default: all keys)
    """
    if not results:
        return "(empty)"
    if key_fields is None:
        key_fields = list(results[0].keys())

    header = " | ".join(f"{k:>20}" for k in key_fields)
    sep    = "-+-".join("-" * 20 for _ in key_fields)
    rows   = []
    for r in results:
        row = " | ".join(f"{str(r.get(k, ''))[:20]:>20}" for k in key_fields)
        rows.append(row)

    return "\n".join([header, sep] + rows)

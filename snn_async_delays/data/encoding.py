"""
Rate-coding encoder: converts (A, B, op_id) batches into spike trains.

Encoding rules
--------------
Input window [win_start, win_end):
  - Channel 0 (A): Poisson rate r_on Hz if A=1, r_off Hz if A=0
  - Channel 1 (B): same rule
  - Channels 2..2+n_ops (Step 3 only): one-hot op encoding at r_on Hz

Outside the input window: no spikes (probability 0).

The total trial length is inferred from the last slot's read_end.
"""

from __future__ import annotations
from typing import List, Tuple

import torch

from snn.model import SlotBoundaries


def encode_trial(
    A_batch: torch.Tensor,          # [B, K]  float
    B_batch: torch.Tensor,          # [B, K]  float
    op_ids:  torch.Tensor,          # [B, K]  long   (ignored if n_ops=0)
    slots:   List[SlotBoundaries],
    n_input: int,
    r_on:    float = 400.0,         # Hz
    r_off:   float = 10.0,          # Hz
    dt:      float = 1.0,           # ms
    device:  str   = "cpu",
) -> torch.Tensor:
    """
    Build the full trial spike train for a batch.

    Returns
    -------
    spike_input : [B, T, n_input]
        T = slots[-1].read_end
    """
    T = slots[-1].read_end
    K = len(slots)
    B = A_batch.shape[0]
    n_ops = max(n_input - 2, 0)

    spike_input = torch.zeros(B, T, n_input, device=device)

    # Convert Hz to per-step probability
    p_on  = r_on  * dt / 1000.0
    p_off = r_off * dt / 1000.0

    for k, slot in enumerate(slots):
        wl = slot.win_end - slot.win_start   # input window length

        A_k = A_batch[:, k].to(device)       # [B]
        B_k = B_batch[:, k].to(device)

        pA = torch.where(A_k > 0.5,
                         torch.full_like(A_k, p_on),
                         torch.full_like(A_k, p_off))   # [B]
        pB = torch.where(B_k > 0.5,
                         torch.full_like(B_k, p_on),
                         torch.full_like(B_k, p_off))

        # Expand to [B, win_len] and sample Poisson
        pA_exp = pA.unsqueeze(1).expand(B, wl)
        pB_exp = pB.unsqueeze(1).expand(B, wl)

        spike_input[:, slot.win_start:slot.win_end, 0] = torch.bernoulli(pA_exp)
        spike_input[:, slot.win_start:slot.win_end, 1] = torch.bernoulli(pB_exp)

        # One-hot op encoding (Step 3)
        if n_ops > 0:
            op_k = op_ids[:, k].to(device)                    # [B]
            op_oh = torch.zeros(B, n_ops, device=device)
            op_oh.scatter_(1, op_k.unsqueeze(1), 1.0)         # [B, n_ops]

            p_op = op_oh * p_on                                # [B, n_ops]
            p_op_exp = p_op.unsqueeze(1).expand(B, wl, n_ops) # [B, wl, n_ops]
            spike_input[:, slot.win_start:slot.win_end, 2:] = torch.bernoulli(p_op_exp)

    return spike_input

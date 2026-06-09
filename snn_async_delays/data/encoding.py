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


def encode_simultaneous_trial(
    A_batch: torch.Tensor,   # [B, K] float
    B_batch: torch.Tensor,   # [B, K] float
    win_len: int,
    read_len: int,
    r_on: float = 400.0,
    r_off: float = 10.0,
    dt: float = 1.0,
    device: str = "cpu",
    **kwargs,   # absorbs op_ids/n_ops passed by Step-3-aware callers
) -> torch.Tensor:
    """
    Encode K queries simultaneously with 2K dedicated input channels.

    TRUE temporal multiplexing design: all K queries are injected at the
    same time into shared hidden neurons. Each query k gets its own pair
    of input channels [2k, 2k+1] = [A_k, B_k].

    Channel layout: [A_0, B_0, A_1, B_1, ..., A_{K-1}, B_{K-1}]

    Timeline:
      [0, win_len)              : Poisson spikes from all 2K channels.
      [win_len, win_len+read_len): silence -- readout accumulates delayed spikes.

    The test: with trainable delays, hidden neuron j can route A_k's input
    to a specific arrival time, effectively time-sharing its capacity across
    K queries. With d=0, all 2K inputs arrive simultaneously and must be
    separated by weights alone.

    Returns
    -------
    spike_input : [B, T, 2K]  where T = win_len + read_len
    """
    B = A_batch.shape[0]
    K = A_batch.shape[1]
    T = win_len + read_len

    p_on  = r_on  * dt / 1000.0
    p_off = r_off * dt / 1000.0

    # Build per-channel probabilities [B, 2K]
    # Interleave A and B: [A_0, B_0, A_1, B_1, ...]
    AB = torch.stack([A_batch.to(device), B_batch.to(device)], dim=2)  # [B, K, 2]
    AB = AB.reshape(B, 2 * K)                                            # [B, 2K]

    p_fire = torch.where(AB > 0.5,
                         torch.full_like(AB, p_on),
                         torch.full_like(AB, p_off))   # [B, 2K]

    # Sample Bernoulli for win_len steps in one call: [B, win_len, 2K]
    p_expanded = p_fire.unsqueeze(1).expand(B, win_len, 2 * K)
    win_spikes = torch.bernoulli(p_expanded)

    spike_input = torch.zeros(B, T, 2 * K, device=device)
    spike_input[:, :win_len, :] = win_spikes
    return spike_input


def encode_sequential_trial(
    A_batch: torch.Tensor,   # [B, K] float
    B_batch: torch.Tensor,   # [B, K] float
    win_len: int,
    read_len: int,
    r_on: float = 400.0,
    r_off: float = 10.0,
    dt: float = 1.0,
    device: str = "cpu",
    op_ids: "torch.Tensor | None" = None,   # [B, K] long  (Step 3 only)
    n_ops: int = 0,                          # >0 enables one-hot op channels
    **kwargs,
) -> torch.Tensor:
    """
    Encode K queries sequentially on shared input channels.

    Plan D — true temporal multiplexing via time-division:
      Query k fires on channels [A, B] during sub-window
      [k*sub_win, (k+1)*sub_win), where sub_win = win_len // K.

    Step 3 extension: when n_ops > 0, appends n_ops one-hot op channels,
    giving n_input_channels = 2 + n_ops total.

    Returns
    -------
    spike_input : [B, T, 2+n_ops]  where T = win_len + read_len
    """
    B = A_batch.shape[0]
    K = A_batch.shape[1]
    T = win_len + read_len
    sub_win = win_len // K
    n_ch = 2 + n_ops

    p_on  = r_on  * dt / 1000.0
    p_off = r_off * dt / 1000.0

    spike_input = torch.zeros(B, T, n_ch, device=device)

    for k in range(K):
        t_start = k * sub_win
        t_end = (k + 1) * sub_win if k < K - 1 else win_len
        wl = t_end - t_start

        A_k = A_batch[:, k].to(device)
        B_k = B_batch[:, k].to(device)

        pA = torch.where(A_k > 0.5,
                         torch.full_like(A_k, p_on),
                         torch.full_like(A_k, p_off))
        pB = torch.where(B_k > 0.5,
                         torch.full_like(B_k, p_on),
                         torch.full_like(B_k, p_off))

        spike_input[:, t_start:t_end, 0] = torch.bernoulli(pA.unsqueeze(1).expand(B, wl))
        spike_input[:, t_start:t_end, 1] = torch.bernoulli(pB.unsqueeze(1).expand(B, wl))

        if n_ops > 0 and op_ids is not None:
            op_k = op_ids[:, k].to(device)                     # [B]
            op_oh = torch.zeros(B, n_ops, device=device)
            op_oh.scatter_(1, op_k.unsqueeze(1), 1.0)          # [B, n_ops]
            p_op_exp = (op_oh * p_on).unsqueeze(1).expand(B, wl, n_ops)
            spike_input[:, t_start:t_end, 2:] = torch.bernoulli(p_op_exp)

    return spike_input

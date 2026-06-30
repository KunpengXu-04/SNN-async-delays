"""
Rate-coding and burst-pattern encoder: converts (A, B, op_id) batches into spike trains.

Encoding rules — rate mode (default)
-------------------------------------
Input window [win_start, win_end):
  - Channel 0 (A): Poisson rate r_on Hz if A=1, r_off Hz if A=0
  - Channel 1 (B): same rule
  - Channels 2..2+n_ops (Step 3 only): one-hot op encoding at r_on Hz

Encoding rules — burst / burst_jitter mode
-------------------------------------------
A/B channels use a sparse deterministic spike pattern instead of Poisson:
  - value=1: burst_n_spikes_on spikes centered at burst_phase_on * (sub_win-1)
  - value=0: burst_n_spikes_off spikes centered at burst_phase_off * (sub_win-1)
  burst_jitter mode adds per-sample uniform jitter ±burst_jitter_ms to each spike.
  Op channels (Step 3) always remain rate-coded regardless of encoding_mode.

Outside the input window: no spikes.

The total trial length is inferred from the last slot's read_end.
"""

from __future__ import annotations
from typing import List, Tuple

import torch

from snn.model import SlotBoundaries


def _burst_times(value: bool, sub_win: int,
                 n_on: int, n_off: int,
                 phase_on: float, phase_off: float) -> list:
    """Return deterministic spike time offsets (0-indexed) within a sub-window.

    Spikes are evenly spaced around a phase-defined center position.
    Clamped to [0, sub_win-1] and deduplicated.
    """
    center = round((phase_on if value else phase_off) * (sub_win - 1))
    n = n_on if value else n_off
    half = n // 2
    times = [center - half + i for i in range(n)]
    return sorted(set(max(0, min(sub_win - 1, t)) for t in times))


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
    **kwargs,                        # absorbs burst params; burst not yet supported here
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
    # burst params (ignored when encoding_mode="rate")
    encoding_mode: str = "rate",
    burst_n_spikes_on: int   = 2,
    burst_n_spikes_off: int  = 1,
    burst_phase_on: float    = 0.2,
    burst_phase_off: float   = 0.8,
    burst_jitter_ms: int     = 1,
    **kwargs,   # absorbs op_ids/n_ops passed by Step-3-aware callers
) -> torch.Tensor:
    """
    Encode K queries simultaneously with 2K dedicated input channels.

    Channel layout: [A_0, B_0, A_1, B_1, ..., A_{K-1}, B_{K-1}]

    Timeline:
      [0, win_len)              : spikes from all 2K channels.
      [win_len, win_len+read_len): silence -- readout accumulates delayed spikes.

    encoding_mode controls A/B spike generation:
      "rate"         : Poisson (r_on / r_off Hz Bernoulli, default)
      "burst"        : deterministic burst at fixed phase positions
      "burst_jitter" : burst with per-sample uniform jitter ±burst_jitter_ms

    Returns
    -------
    spike_input : [B, T, 2K]  where T = win_len + read_len
    """
    B = A_batch.shape[0]
    K = A_batch.shape[1]
    T = win_len + read_len

    # Interleave A and B: [A_0, B_0, A_1, B_1, ...]
    AB = torch.stack([A_batch.to(device), B_batch.to(device)], dim=2)  # [B, K, 2]
    AB = AB.reshape(B, 2 * K)                                            # [B, 2K]

    spike_input = torch.zeros(B, T, 2 * K, device=device)

    if encoding_mode == "rate":
        p_on  = r_on  * dt / 1000.0
        p_off = r_off * dt / 1000.0
        p_fire = torch.where(AB > 0.5,
                             torch.full_like(AB, p_on),
                             torch.full_like(AB, p_off))   # [B, 2K]
        p_expanded = p_fire.unsqueeze(1).expand(B, win_len, 2 * K)
        spike_input[:, :win_len, :] = torch.bernoulli(p_expanded)
    else:
        jitter = burst_jitter_ms if encoding_mode == "burst_jitter" else 0
        times_on  = _burst_times(True,  win_len, burst_n_spikes_on, burst_n_spikes_off,
                                 burst_phase_on, burst_phase_off)
        times_off = _burst_times(False, win_len, burst_n_spikes_on, burst_n_spikes_off,
                                 burst_phase_on, burst_phase_off)
        for ch in range(2 * K):
            mask_on  = (AB[:, ch] > 0.5)   # [B]
            mask_off = ~mask_on
            for t_base, mask in ([(t, mask_on) for t in times_on]
                                  + [(t, mask_off) for t in times_off]):
                if jitter > 0:
                    t_jit = (t_base + torch.randint(-jitter, jitter + 1, (B,), device=spike_input.device)).clamp(0, win_len - 1)
                    b_idx = mask.nonzero(as_tuple=True)[0]
                    spike_input[b_idx, t_jit[b_idx], ch] = 1.0
                else:
                    spike_input[mask, t_base, ch] = 1.0

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
    # burst params (ignored when encoding_mode="rate")
    encoding_mode: str = "rate",
    burst_n_spikes_on: int   = 2,
    burst_n_spikes_off: int  = 1,
    burst_phase_on: float    = 0.2,
    burst_phase_off: float   = 0.8,
    burst_jitter_ms: int     = 1,
    **kwargs,
) -> torch.Tensor:
    """
    Encode K queries sequentially on shared input channels.

    Plan D — true temporal multiplexing via time-division:
      Query k fires on channels [A, B] during sub-window
      [k*sub_win, (k+1)*sub_win), where sub_win = win_len // K.

    Step 3 extension: when n_ops > 0, appends n_ops one-hot op channels
    (always rate-coded, regardless of encoding_mode).

    encoding_mode controls A/B spike generation:
      "rate"         : Poisson (r_on / r_off Hz Bernoulli, default, backward compat)
      "burst"        : deterministic burst at fixed phase positions within sub-window
      "burst_jitter" : burst with per-sample uniform jitter ±burst_jitter_ms

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

    # Pre-compute burst base times (same for every sub-window, re-clamped per wl below)
    if encoding_mode != "rate":
        jitter = burst_jitter_ms if encoding_mode == "burst_jitter" else 0

    for k in range(K):
        t_start = k * sub_win
        t_end = (k + 1) * sub_win if k < K - 1 else win_len
        wl = t_end - t_start

        A_k = A_batch[:, k].to(device)
        B_k = B_batch[:, k].to(device)

        if encoding_mode == "rate":
            pA = torch.where(A_k > 0.5,
                             torch.full_like(A_k, p_on),
                             torch.full_like(A_k, p_off))
            pB = torch.where(B_k > 0.5,
                             torch.full_like(B_k, p_on),
                             torch.full_like(B_k, p_off))
            spike_input[:, t_start:t_end, 0] = torch.bernoulli(pA.unsqueeze(1).expand(B, wl))
            spike_input[:, t_start:t_end, 1] = torch.bernoulli(pB.unsqueeze(1).expand(B, wl))
        else:
            # Burst / burst_jitter: place deterministic spike clusters in sub-window
            times_on  = _burst_times(True,  wl, burst_n_spikes_on, burst_n_spikes_off,
                                     burst_phase_on, burst_phase_off)
            times_off = _burst_times(False, wl, burst_n_spikes_on, burst_n_spikes_off,
                                     burst_phase_on, burst_phase_off)
            for ch, val_tensor in enumerate([A_k, B_k]):
                mask_on  = (val_tensor > 0.5)   # [B]
                mask_off = ~mask_on
                for t_base, mask in ([(t, mask_on) for t in times_on]
                                      + [(t, mask_off) for t in times_off]):
                    if jitter > 0:
                        t_jit = (t_base + torch.randint(-jitter, jitter + 1, (B,), device=spike_input.device)).clamp(0, wl - 1)
                        b_idx = mask.nonzero(as_tuple=True)[0]
                        spike_input[b_idx, t_start + t_jit[b_idx], ch] = 1.0
                    else:
                        spike_input[mask, t_start + t_base, ch] = 1.0

        # One-hot op encoding (Step 3) — always rate-coded
        if n_ops > 0 and op_ids is not None:
            op_k = op_ids[:, k].to(device)                     # [B]
            op_oh = torch.zeros(B, n_ops, device=device)
            op_oh.scatter_(1, op_k.unsqueeze(1), 1.0)          # [B, n_ops]
            p_op_exp = (op_oh * p_on).unsqueeze(1).expand(B, wl, n_ops)
            spike_input[:, t_start:t_end, 2:] = torch.bernoulli(p_op_exp)

    return spike_input

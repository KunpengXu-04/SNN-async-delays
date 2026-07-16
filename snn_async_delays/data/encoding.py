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


def _jittered_burst_times(
    base_times: list[int], batch_size: int, jitter: int, width: int, device: str
) -> torch.Tensor:
    """Jitter a burst without silently deleting spikes through time collisions.

    The previous implementation independently jittered each spike and wrote
    into a binary tensor.  When two jittered times coincided, assignment merged
    them, so a nominal two-spike ``1`` sometimes became a one-spike input.  The
    resulting perturbation changed both timing *and* spike count.  Here each
    event is sampled from its legal jitter neighbourhood excluding times already
    occupied by the same burst.  A configuration requiring more unique events
    than time bins is rejected explicitly.
    """
    if len(base_times) > width:
        raise ValueError("Burst has more events than available time bins")
    result = torch.empty(batch_size, len(base_times), dtype=torch.long, device=device)
    for b in range(batch_size):
        used: set[int] = set()
        for index, base in enumerate(base_times):
            candidates = [
                t for t in range(max(0, base - jitter), min(width - 1, base + jitter) + 1)
                if t not in used
            ]
            if not candidates:
                # This can only occur under an unusually dense burst.  Fall
                # back to any unused bin rather than silently changing count.
                candidates = [t for t in range(width) if t not in used]
            choice = candidates[torch.randint(len(candidates), (1,), device=device).item()]
            result[b, index] = choice
            used.add(choice)
    return result


def _write_burst(
    spike_input: torch.Tensor,
    mask: torch.Tensor,
    base_times: list[int],
    start: int,
    channel: int,
    jitter: int,
    width: int,
) -> None:
    """Write one binary burst per selected sample, preserving event count."""
    if not bool(mask.any()):
        return
    if jitter == 0:
        for t in base_times:
            spike_input[mask, start + t, channel] = 1.0
        return
    times = _jittered_burst_times(
        base_times, spike_input.shape[0], jitter, width, str(spike_input.device)
    )
    batch_indices = mask.nonzero(as_tuple=True)[0]
    for event in range(len(base_times)):
        spike_input[batch_indices, start + times[batch_indices, event], channel] = 1.0


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
    one_hot_phase: float     = 1.0,
    one_hot_n_spikes: int    = 1,
    **kwargs,   # absorbs op_ids/n_ops passed by Step-3-aware callers
) -> torch.Tensor:
    """
    Encode K queries simultaneously with dedicated input channels.

    The historical rate/burst modes use 2K channels laid out as
    `[A_0,B_0,A_1,B_1,...]`. ``binary_one_hot`` uses 4K channels laid out as
    `[A0,A1,B0,B1]` per query and emits exactly `2*one_hot_n_spikes` events per
    query, independent of the bit values.

    Timeline:
      [0, win_len)              : spikes from all 2K channels.
      [win_len, win_len+read_len): silence -- readout accumulates delayed spikes.

    encoding_mode controls A/B spike generation:
      "rate"         : Poisson (r_on / r_off Hz Bernoulli, default)
      "burst"        : deterministic burst at fixed phase positions
      "burst_jitter" : burst with per-sample uniform jitter ±burst_jitter_ms

    ``binary_one_hot`` places both selected value-channel events at the same
    declared phase. It is the neutral spatial-vs-temporal Pareto input code:
    value identity is spatial while query scheduling remains the treatment.

    Returns
    -------
    spike_input : [B, T, 2K]  where T = win_len + read_len
    """
    B = A_batch.shape[0]
    K = A_batch.shape[1]
    T = win_len + read_len

    if encoding_mode == "binary_one_hot":
        spike_input = torch.zeros(B, T, 4 * K, device=device)
        if int(one_hot_n_spikes) <= 0:
            raise ValueError("one_hot_n_spikes must be positive")
        event_times = _burst_times(
            True, win_len, int(one_hot_n_spikes), int(one_hot_n_spikes),
            float(one_hot_phase), float(one_hot_phase),
        )
        batch_index = torch.arange(B, device=device)
        for query in range(K):
            a_channel = 4 * query + A_batch[:, query].to(device).long().clamp(0, 1)
            b_channel = 4 * query + 2 + B_batch[:, query].to(device).long().clamp(0, 1)
            for event_time in event_times:
                spike_input[batch_index, event_time, a_channel] = 1.0
                spike_input[batch_index, event_time, b_channel] = 1.0
        return spike_input

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
            _write_burst(spike_input, mask_on, times_on, 0, ch, jitter, win_len)
            _write_burst(spike_input, mask_off, times_off, 0, ch, jitter, win_len)

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
                _write_burst(spike_input, mask_on, times_on, t_start, ch, jitter, wl)
                _write_burst(spike_input, mask_off, times_off, t_start, ch, jitter, wl)

        # One-hot op encoding (Step 3) — always rate-coded
        if n_ops > 0 and op_ids is not None:
            op_k = op_ids[:, k].to(device)                     # [B]
            op_oh = torch.zeros(B, n_ops, device=device)
            op_oh.scatter_(1, op_k.unsqueeze(1), 1.0)          # [B, n_ops]
            p_op_exp = (op_oh * p_on).unsqueeze(1).expand(B, wl, n_ops)
            spike_input[:, t_start:t_end, 2:] = torch.bernoulli(p_op_exp)

    return spike_input

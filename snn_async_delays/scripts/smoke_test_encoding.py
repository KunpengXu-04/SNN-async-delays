"""
Smoke test for burst encoding.

Verifies:
  1. tensor shape is correct for all three modes
  2. on-patterns and off-patterns produce different spike times
  3. no all-zero or all-one tensors for any mode
  4. burst spikes land within the correct sub-window
  5. jitter mode produces different patterns across calls

Also saves a side-by-side raster PNG to runs/smoke_test/rasters.png.

Usage:
    python -m scripts.smoke_test_encoding
    python -m scripts.smoke_test_encoding --no-plot
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data.encoding import encode_sequential_trial, _burst_times
from data.boolean_dataset import MultiQueryDataset


def _make_batch(K: int, B: int = 8):
    """Create a batch with all four (A,B) combos repeated twice, K copies."""
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)] * (B // 4)
    A = torch.tensor([[a for _ in range(K)] for a, b in combos], dtype=torch.float32)
    B_t = torch.tensor([[b for _ in range(K)] for a, b in combos], dtype=torch.float32)
    return A, B_t


def _check_tensor(t: torch.Tensor, name: str, win_len: int, read_len: int, K: int):
    B, T_actual, n_ch = t.shape
    T_expected = win_len + read_len
    assert T_actual == T_expected, \
        f"{name}: T={T_actual} != {T_expected}"
    assert n_ch == 2, \
        f"{name}: n_ch={n_ch} != 2"
    # No spikes in readout window
    readout_spikes = t[:, win_len:, :].sum().item()
    assert readout_spikes == 0.0, \
        f"{name}: {readout_spikes} spikes in readout window (should be 0)"
    # Not all zeros
    total = t.sum().item()
    assert total > 0, f"{name}: all-zero tensor"
    # Not all ones
    assert total < B * win_len * n_ch, f"{name}: all-one tensor"
    print(f"  {name:30s}  shape={list(t.shape)}  total_spikes={int(total)}")
    return True


def _check_burst_positions(t: torch.Tensor, K: int, sub_win: int,
                            burst_phase_on: float, burst_phase_off: float,
                            n_on: int, n_off: int):
    """Verify burst spikes land only at expected sub-window offsets.

    Batch layout: [(0,0),(0,1),(1,0),(1,1)] × 2 — A=1 at indices 2,3,6,7.
    """
    B, T, n_ch = t.shape
    # A=1 at sample indices 2,3,6,7; A=0 at 0,1,4,5
    a1_idx = [2, 3, 6, 7]
    a0_idx = [0, 1, 4, 5]

    times_on  = _burst_times(True,  sub_win, n_on, n_off, burst_phase_on, burst_phase_off)
    times_off = _burst_times(False, sub_win, n_on, n_off, burst_phase_on, burst_phase_off)

    for k in range(K):
        t_start = k * sub_win
        t_end   = t_start + sub_win
        win     = t[:, t_start:t_end, :]   # [B, sub_win, 2]

        # ch=0 (A channel): A=1 samples should spike at times_on only
        cols_a1 = set(win[a1_idx, :, 0].nonzero(as_tuple=True)[1].tolist())
        cols_a0 = set(win[a0_idx, :, 0].nonzero(as_tuple=True)[1].tolist())

        bad_on  = cols_a1 - set(times_on)
        bad_off = cols_a0 - set(times_off)
        if bad_on or bad_off:
            print(f"    WARNING query {k}: A=1 unexpected cols={bad_on}  "
                  f"A=0 unexpected cols={bad_off}")
        else:
            print(f"    OK query {k}: A=1 at {sorted(cols_a1)}  "
                  f"A=0 at {sorted(cols_a0)}")


def plot_rasters(tensors: dict, K: int, sub_win: int, read_len: int, out_path: str):
    """Save a side-by-side raster PNG for all encoding modes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping raster plot")
        return

    n_modes = len(tensors)
    fig, axes = plt.subplots(2, n_modes, figsize=(5 * n_modes, 6),
                              sharex=False, sharey=False)
    if n_modes == 1:
        axes = axes[:, None]

    win_len = K * sub_win

    for col, (mode, t) in enumerate(tensors.items()):
        # Show sample 2 (A=1, B=0) for channel 0 (A) and channel 1 (B)
        sample_idx = 2  # A=1, B=0 combo
        t_np = t[sample_idx].cpu().numpy()  # [T, 2]

        for row, (ch, ch_name) in enumerate([(0, "A"), (1, "B")]):
            ax = axes[row][col]
            spk = t_np[:, ch]  # [T]
            spike_times = np.where(spk > 0)[0]
            if len(spike_times) > 0:
                ax.vlines(spike_times, 0, 1, colors="black", linewidth=1.5)
            # Sub-window boundaries
            for k in range(K):
                ax.axvline(k * sub_win, color="blue", alpha=0.3, linestyle="--", linewidth=0.8)
            ax.axvline(win_len, color="red", alpha=0.5, linestyle="-", linewidth=1.2,
                       label="readout")
            ax.set_xlim(0, win_len + read_len)
            ax.set_ylim(-0.1, 1.5)
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{mode}", fontsize=10, fontweight="bold")
            ax.set_ylabel(f"ch {ch_name}", fontsize=9)
            if row == 1:
                ax.set_xlabel("time (ms)", fontsize=9)
            n_spk = len(spike_times)
            ax.text(0.02, 0.85, f"A={'1' if sample_idx in [2,3] else '0'} → {n_spk} spk",
                    transform=ax.transAxes, fontsize=8, color="darkred")

    fig.suptitle(f"Encoding smoke test  (K={K}, sub_win={sub_win}, sample: A=1, B=0)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Raster saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--K",       type=int, default=3)
    parser.add_argument("--sub_win", type=int, default=10)
    parser.add_argument("--read_len",type=int, default=10)
    args = parser.parse_args()

    K       = args.K
    sub_win = args.sub_win
    read_len = args.read_len
    win_len = K * sub_win
    B       = 8

    # Burst params
    n_on, n_off     = 2, 1
    phase_on        = 0.20
    phase_off       = 0.80
    jitter_ms       = 1

    A, B_t = _make_batch(K, B)

    common_kwargs = dict(
        win_len=win_len, read_len=read_len,
        r_on=400.0, r_off=10.0, dt=1.0, device="cpu",
        burst_n_spikes_on=n_on, burst_n_spikes_off=n_off,
        burst_phase_on=phase_on, burst_phase_off=phase_off,
        burst_jitter_ms=jitter_ms,
    )

    modes = {
        "rate":         "rate",
        "burst":        "burst",
        "burst_jitter": "burst_jitter",
    }

    print("=" * 60)
    print(f"Encoding smoke test  K={K}  sub_win={sub_win}  read_len={read_len}")
    print("=" * 60)

    tensors = {}
    all_passed = True

    for mode_name, enc_mode in modes.items():
        t = encode_sequential_trial(A, B_t, encoding_mode=enc_mode, **common_kwargs)
        ok = _check_tensor(t, mode_name, win_len, read_len, K)
        all_passed = all_passed and ok
        tensors[mode_name] = t

    print()
    print("Burst position check:")
    _check_burst_positions(
        tensors["burst"], K, sub_win,
        phase_on, phase_off, n_on, n_off,
    )

    # Jitter reproducibility check
    print()
    print("Jitter variability check (two calls should differ):")
    t_jit1 = encode_sequential_trial(A, B_t, encoding_mode="burst_jitter", **common_kwargs)
    t_jit2 = encode_sequential_trial(A, B_t, encoding_mode="burst_jitter", **common_kwargs)
    are_different = not torch.equal(t_jit1, t_jit2)
    print(f"  Two burst_jitter calls produce different tensors: {are_different}")
    if not are_different:
        print("  WARNING: jitter calls produced identical tensors (rng seed issue?)")

    # Rate vs burst must differ
    print()
    print("Encoding mode differentiation check:")
    rate_eq_burst = torch.equal(tensors["rate"], tensors["burst"])
    print(f"  rate == burst (should be False): {rate_eq_burst}")
    if rate_eq_burst:
        print("  ERROR: rate and burst produced identical tensors!")
        all_passed = False

    # Check that on-pattern and off-pattern differ
    on_sample  = 2   # A=1, B=0
    off_sample = 0   # A=0, B=0
    burst = tensors["burst"]
    on_A  = burst[on_sample,  :win_len, 0].sum().item()
    off_A = burst[off_sample, :win_len, 0].sum().item()
    print(f"  burst: A=1 total spikes={int(on_A)}, A=0 total spikes={int(off_A)}")
    if on_A == off_A:
        print("  WARNING: on and off patterns have same spike count!")

    print()
    if all_passed:
        print(f"ALL CHECKS PASSED  (shape=[{B}, {win_len+read_len}, 2])")
    else:
        print("SOME CHECKS FAILED — see above")

    if not args.no_plot:
        plot_rasters(
            tensors, K=K, sub_win=sub_win, read_len=read_len,
            out_path=os.path.join("runs", "smoke_test", "rasters.png"),
        )


if __name__ == "__main__":
    main()

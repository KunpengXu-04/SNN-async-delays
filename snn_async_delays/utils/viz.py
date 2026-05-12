"""
Visualization utilities.

Functions
---------
plot_training_curves
plot_delay_distribution
plot_delay_histogram
plot_K_accuracy
plot_throughput
plot_metric_triplet
plot_spike_raster
plot_confusion_matrix
plot_opwise_accuracy
"""

from __future__ import annotations
import os
from typing import List, Dict, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_training_curves(log_rows: List[Dict], save_path: str):
    epochs = [r["epoch"] for r in log_rows]
    train_acc = [r["train_acc"] for r in log_rows]
    val_acc = [r["val_acc"] for r in log_rows]
    train_loss = [r["train_loss"] for r in log_rows]
    val_loss = [r["val_loss"] for r in log_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.95, color="gray", linestyle="--", linewidth=0.8)
    axes[1].legend()
    axes[1].set_title("Accuracy")

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_delay_distribution(delays: np.ndarray, title: str, save_path: str):
    """Heatmap for delay matrix [N_pre, N_post]."""
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(delays, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="delay (steps)")
    ax.set_xlabel("Post-synaptic neuron")
    ax.set_ylabel("Pre-synaptic neuron")
    ax.set_title(title)
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_delay_histogram(
    delays: np.ndarray,
    save_path: str,
    title: str = "Delay Distribution",
    bins: int = 30,
):
    """Histogram (and light KDE-style smoothing via line overlay) for delay values."""
    vals = delays.reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=bins, alpha=0.7, color="#4C72B0", density=True)

    # Lightweight smooth curve from histogram centers (no scipy dependency)
    hist, edges = np.histogram(vals, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(hist) >= 5:
        kernel = np.array([1, 4, 6, 4, 1], dtype=float)
        kernel = kernel / kernel.sum()
        smooth = np.convolve(hist, kernel, mode="same")
        ax.plot(centers, smooth, color="#C44E52", linewidth=2)

    ax.set_xlabel("Delay (steps)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_K_accuracy(
    K_values: List[int],
    results_by_mode: Dict[str, List[float]],
    tau: float = 0.95,
    save_path: str = "k_accuracy.png",
):
    fig, ax = plt.subplots(figsize=(7, 4))

    for mode, accs in results_by_mode.items():
        ax.plot(K_values, accs, marker="o", label=mode)

    ax.axhline(tau, color="gray", linestyle="--", linewidth=0.8, label=f"tau={tau}")
    ax.set_xlabel("K (queries per trial)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(K_values)
    ax.legend()
    ax.set_title("Accuracy vs K")
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_throughput(
    K_values: List[int],
    results_by_mode: Dict[str, List[float]],
    save_path: str = "throughput.png",
):
    fig, ax = plt.subplots(figsize=(7, 4))

    for mode, thr in results_by_mode.items():
        ax.plot(K_values, thr, marker="s", label=mode)

    ax.set_xlabel("K (queries per trial)")
    ax.set_ylabel("K / total_hidden_spikes")
    ax.set_xticks(K_values)
    ax.legend()
    ax.set_title("Energy-normalized throughput vs K")
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_metric_triplet(
    K_values: List[int],
    acc_by_mode: Dict[str, List[float]],
    thr_by_mode: Dict[str, List[float]],
    dens_by_mode: Dict[str, List[float]],
    tau_list: Optional[List[float]] = None,
    save_path: str = "metric_triplet.png",
):
    """
    Three-panel figure:
      1) Accuracy vs K
      2) K/spike vs K
      3) ops/neuron/ms vs K
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for mode, vals in acc_by_mode.items():
        axes[0].plot(K_values, vals, marker="o", label=mode)
    if tau_list:
        for tau in tau_list:
            axes[0].axhline(tau, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("Accuracy vs K")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(K_values)

    for mode, vals in thr_by_mode.items():
        axes[1].plot(K_values, vals, marker="s", label=mode)
    axes[1].set_title("K/spike vs K")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("K / hidden_spikes")
    axes[1].set_xticks(K_values)

    for mode, vals in dens_by_mode.items():
        axes[2].plot(K_values, vals, marker="^", label=mode)
    axes[2].set_title("Ops/neuron/time vs K")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("ops / neuron / ms")
    axes[2].set_xticks(K_values)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_spike_raster(
    spike_train: np.ndarray,
    title: str,
    save_path: str,
    slot_boundaries: Optional[List] = None,
):
    T, N = spike_train.shape
    fig, ax = plt.subplots(figsize=(10, 4))

    for n in range(N):
        times = np.where(spike_train[:, n] > 0)[0]
        ax.scatter(times, np.full_like(times, n), s=2, c="black", marker="|")

    if slot_boundaries is not None:
        for slot in slot_boundaries:
            ax.axvspan(slot.win_start, slot.win_end, alpha=0.1, color="blue")
            ax.axvspan(slot.read_start, slot.read_end, alpha=0.1, color="red")

    ax.set_xlabel("Timestep (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(
    conf: List[List[int]],
    save_path: str,
    title: str = "Confusion Matrix",
    labels: Optional[List[str]] = None,
):
    labels = labels or ["0", "1"]
    arr = np.array(conf, dtype=float)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(arr, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_multilayer_raster(
    spike_input: np.ndarray,
    hidden_spikes: np.ndarray,
    save_path: str,
    win_len: int,
    read_len: int,
    title: str = "Spike Raster",
    K: int = 1,
    sub_win: Optional[int] = None,
):
    """
    Two-panel spike raster: Input layer (top) and Hidden layer (bottom).

    Parameters
    ----------
    spike_input   : [T, n_input]   binary spike train for input neurons
    hidden_spikes : [T, n_hidden]  binary spike train for hidden neurons
    win_len       : length of input injection window
    read_len      : length of readout window
    K             : number of queries
    sub_win       : per-query sub-window length (Plan D); None for Plan C
    """
    T, n_input  = spike_input.shape
    _,  n_hidden = hidden_spikes.shape

    COLORS = plt.cm.tab10.colors

    # Height ratio: one row per neuron (min 1 px each)
    hr = [max(n_input, 2), max(n_hidden, 4)]
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 1.5 + 0.2 * (n_input + n_hidden)),
        sharex=True, gridspec_kw={"height_ratios": hr}
    )
    ax_in, ax_hid = axes

    # ── Input raster ──────────────────────────────────────────────────
    ax_in.set_title(title, fontsize=11)
    ax_in.set_ylabel("Input neuron", fontsize=9)

    for n in range(n_input):
        ts = np.where(spike_input[:, n] > 0)[0]
        if len(ts) == 0:
            continue
        # Plan D: 2 shared channels → all same colour but annotated by window
        # Plan C: 2K channels → colour by query index
        if n_input == 2:
            col = "#333333"
        else:
            col = COLORS[(n // 2) % len(COLORS)]
        ax_in.scatter(ts, np.full_like(ts, n), s=5, color=col, marker="|", linewidths=0.8)

    ax_in.set_ylim(-0.5, n_input - 0.5)
    ax_in.set_yticks(range(n_input))
    if n_input == 2:
        ax_in.set_yticklabels(["A", "B"], fontsize=8)
    else:
        ax_in.set_yticklabels(
            [f"A{n//2}" if n % 2 == 0 else f"B{n//2}" for n in range(n_input)],
            fontsize=6,
        )

    # ── Hidden raster ─────────────────────────────────────────────────
    ax_hid.set_ylabel("Hidden neuron", fontsize=9)
    ax_hid.set_xlabel("Timestep (ms)", fontsize=9)

    for n in range(n_hidden):
        ts = np.where(hidden_spikes[:, n] > 0)[0]
        if len(ts) == 0:
            continue
        ax_hid.scatter(ts, np.full_like(ts, n), s=5, color="steelblue",
                       marker="|", linewidths=0.8)

    ax_hid.set_ylim(-0.5, n_hidden - 0.5)
    ax_hid.set_yticks(range(0, n_hidden, max(1, n_hidden // 10)))

    # ── Annotations: windows & query boundaries ───────────────────────
    from matplotlib.patches import Patch
    for ax in axes:
        ax.axvspan(0, win_len, alpha=0.07, color="royalblue", zorder=0)
        ax.axvspan(win_len, win_len + read_len, alpha=0.10, color="tomato", zorder=0)
        ax.set_xlim(-0.5, T - 0.5)

        if sub_win is not None and K > 1:
            for k in range(K):
                t0 = k * sub_win
                t1 = (k + 1) * sub_win if k < K - 1 else win_len
                ax.axvline(t0, color=COLORS[k % len(COLORS)],
                           linestyle="--", linewidth=0.9, alpha=0.8, zorder=1)
                ax.axvline(t1, color=COLORS[k % len(COLORS)],
                           linestyle="--", linewidth=0.9, alpha=0.4, zorder=1)
                mid = (t0 + t1) / 2
                ymax = ax.get_ylim()[1]
                ax.text(mid, ymax * 0.97, f"Q{k}", ha="center", va="top",
                        fontsize=7, color=COLORS[k % len(COLORS)],
                        fontweight="bold", zorder=2)

    legend_patches = [
        Patch(facecolor="royalblue", alpha=0.3, label="input window"),
        Patch(facecolor="tomato",    alpha=0.4, label="readout window"),
    ]
    ax_in.legend(handles=legend_patches, loc="upper right", fontsize=8,
                 framealpha=0.8)

    fig.tight_layout(h_pad=0.3)
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_layer_flow(
    spike_input: np.ndarray,
    hidden_spikes: np.ndarray,
    save_path: str,
    win_len: int,
    read_len: int,
    delays: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    title: str = "Layer-to-Layer Spike Flow",
    K: int = 1,
    sub_win: Optional[int] = None,
    n_arrows: int = 8,
):
    """
    Layer-to-layer directed spike flow diagram.

    Y-axis represents the neural hierarchy (Input → Hidden → Readout).
    Within each layer-band spikes are scattered by neuron index.
    Delay arrows are drawn for the n_arrows synapses with largest |weight|.

    Parameters
    ----------
    spike_input   : [T, n_input]   binary spike train
    hidden_spikes : [T, n_hidden]  binary spike train
    delays        : [n_input, n_hidden]  learned delay values (optional)
    weights       : [n_input, n_hidden]  synaptic weights    (optional)
    n_arrows      : how many delay arrows to draw
    """
    T,  n_input  = spike_input.shape
    _,  n_hidden = hidden_spikes.shape

    COLORS = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(13, 5))

    # ── Band layout ──────────────────────────────────────────────────
    # Input  band : y ∈ [0,  1)  — each neuron spread across 0..1
    # Hidden band : y ∈ [1.5, 2.5)
    # Readout line: y = 3.2

    BAND_IN_BOT,  BAND_IN_TOP  = 0.0, 1.0
    BAND_HID_BOT, BAND_HID_TOP = 1.5, 2.5
    READOUT_Y = 3.2

    def input_y(n):
        return BAND_IN_BOT + (n + 0.5) / n_input * (BAND_IN_TOP - BAND_IN_BOT)

    def hidden_y(n):
        return BAND_HID_BOT + (n + 0.5) / n_hidden * (BAND_HID_TOP - BAND_HID_BOT)

    # ── Input spikes ─────────────────────────────────────────────────
    for n in range(n_input):
        ts = np.where(spike_input[:, n] > 0)[0]
        if len(ts) == 0:
            continue
        col = COLORS[(n // 2) % len(COLORS)] if n_input > 2 else "#333333"
        ys  = np.full(len(ts), input_y(n))
        ax.scatter(ts, ys, s=6, color=col, marker="|", linewidths=1.0, zorder=3)

    # ── Hidden spikes ─────────────────────────────────────────────────
    for n in range(n_hidden):
        ts = np.where(hidden_spikes[:, n] > 0)[0]
        if len(ts) == 0:
            continue
        ys = np.full(len(ts), hidden_y(n))
        ax.scatter(ts, ys, s=6, color="steelblue", marker="|",
                   linewidths=1.0, zorder=3)

    # ── Readout marker ────────────────────────────────────────────────
    ax.axhline(READOUT_Y, xmin=(win_len / T), xmax=1.0,
               color="tomato", linewidth=2, alpha=0.6, zorder=2)
    ax.text(win_len + (T - win_len) / 2, READOUT_Y + 0.08,
            "Readout window (accumulate → logits)",
            ha="center", va="bottom", fontsize=8, color="tomato")

    # ── Delay arrows (most important synapses) ────────────────────────
    if delays is not None and n_arrows > 0:
        if weights is not None:
            importance = np.abs(weights).flatten()
        else:
            importance = np.ones(n_input * n_hidden)
        flat_idx  = np.argsort(importance)[::-1][:n_arrows]
        pre_idxs  = flat_idx // n_hidden
        post_idxs = flat_idx %  n_hidden

        for pre, post in zip(pre_idxs, post_idxs):
            d = delays[pre, post]
            # Find first spike from pre-synaptic neuron
            ts_pre = np.where(spike_input[:, pre] > 0)[0]
            if len(ts_pre) == 0:
                continue
            t_spike = ts_pre[0]
            t_arrive = t_spike + d
            if t_arrive >= T:
                continue
            y0 = input_y(pre)
            y1 = hidden_y(post)
            col = COLORS[(pre // 2) % len(COLORS)] if n_input > 2 else COLORS[pre]
            ax.annotate(
                "", xy=(t_arrive, y1), xytext=(t_spike, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=col, alpha=0.55,
                    lw=0.9, mutation_scale=8,
                    connectionstyle="arc3,rad=0.15",
                ),
                zorder=4,
            )

    # ── Band backgrounds & labels ─────────────────────────────────────
    ax.axhspan(BAND_IN_BOT - 0.1,  BAND_IN_TOP + 0.1,
               alpha=0.06, color="royalblue", zorder=0)
    ax.axhspan(BAND_HID_BOT - 0.1, BAND_HID_TOP + 0.1,
               alpha=0.06, color="steelblue", zorder=0)
    ax.text(-0.8, (BAND_IN_BOT + BAND_IN_TOP) / 2,
            "Input", va="center", ha="right", fontsize=9, fontweight="bold",
            color="royalblue")
    ax.text(-0.8, (BAND_HID_BOT + BAND_HID_TOP) / 2,
            "Hidden", va="center", ha="right", fontsize=9, fontweight="bold",
            color="steelblue")
    ax.text(-0.8, READOUT_Y,
            "Readout", va="center", ha="right", fontsize=9, fontweight="bold",
            color="tomato")

    # ── Window shading & query lines ──────────────────────────────────
    ax.axvspan(0, win_len, alpha=0.05, color="royalblue", zorder=0)
    ax.axvspan(win_len, win_len + read_len, alpha=0.08, color="tomato", zorder=0)

    if sub_win is not None and K > 1:
        for k in range(K):
            t0 = k * sub_win
            ax.axvline(t0, color=COLORS[k % len(COLORS)],
                       linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)
            ax.text(t0 + 0.3, BAND_IN_BOT - 0.05, f"Q{k}",
                    fontsize=7, color=COLORS[k % len(COLORS)],
                    va="top", fontweight="bold")

    # ── Axes formatting ───────────────────────────────────────────────
    ax.set_xlim(-1, T)
    ax.set_ylim(BAND_IN_BOT - 0.25, READOUT_Y + 0.35)
    ax.set_yticks([])
    ax.set_xlabel("Timestep (ms)", fontsize=10)
    ax.set_title(title, fontsize=11)

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="royalblue", alpha=0.3, label="input window"),
        Patch(facecolor="tomato",    alpha=0.4, label="readout window"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.8)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_opwise_accuracy(
    op_accuracy: Dict[str, float],
    save_path: str,
    title: str = "Operation-wise Accuracy",
):
    if not op_accuracy:
        return

    ops = list(op_accuracy.keys())
    vals = [op_accuracy[k] for k in ops]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(ops)), vals, color="#55A868")
    ax.set_xticks(np.arange(len(ops)))
    ax.set_xticklabels(ops, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)

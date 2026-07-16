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
plot_multilayer_raster
plot_layer_flow
plot_K_vs_neurons
plot_neuron_efficiency
plot_depth_ablation_curves
plot_weight_heatmaps
plot_delay_heatmaps
plot_spike_raster_layers
plot_layer_to_layer_spike_flow
plot_diagnostic_panel
save_run_diagnostic_plots
plot_accuracy_heatmap
plot_accuracy_curves
plot_capacity_clean
plot_neuron_connection_raster
plot_truth_table_raster
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


# ─────────────────────────────────────────────────────────────────────────────
# Supervisor-requested plots (2026-06)
# ─────────────────────────────────────────────────────────────────────────────

def plot_K_vs_neurons(
    K_values: List[int],
    min_h_with_delay: Dict[int, Optional[int]],
    min_h_no_delay: Dict[int, Optional[int]],
    tau: float = 0.90,
    save_path: str = "k_vs_neurons.png",
):
    """
    X-axis: K (queries per trial).
    Y-axis: minimum hidden neurons to achieve >= tau accuracy on Plan D.

    Pass None as the value for a K where the minimum h is unknown (>tested range).
    Those points will be plotted at max_known + 20% with a '>N' label.
    """
    COLORS = {"delay": "#2196F3", "no_delay": "#F44336"}
    MARKERS = {"delay": "o", "no_delay": "s"}
    LABELS = {"delay": "Trainable delays (w&d)", "no_delay": "No delay (d=0)"}

    fig, ax = plt.subplots(figsize=(7, 5))

    all_known = [v for d in (min_h_with_delay, min_h_no_delay)
                 for v in d.values() if v is not None]
    h_max_tested = max(all_known) if all_known else 100
    y_cap = h_max_tested * 1.5

    for key, data in [("delay", min_h_with_delay), ("no_delay", min_h_no_delay)]:
        ks_known, hs_known = [], []
        ks_capped, hs_capped = [], []

        for k in sorted(data.keys()):
            h = data[k]
            if h is None:
                ks_capped.append(k)
                hs_capped.append(y_cap)
            else:
                ks_known.append(k)
                hs_known.append(h)

        # Always draw label (even if only capped points exist)
        label_added = False
        if ks_known:
            ax.plot(ks_known, hs_known, f"{MARKERS[key]}-",
                    color=COLORS[key], linewidth=2, markersize=8,
                    label=LABELS[key], zorder=3)
            label_added = True
        if ks_capped:
            ax.plot(ks_capped, hs_capped, f"{MARKERS[key]}",
                    color=COLORS[key], markersize=9, markerfacecolor="white",
                    markeredgewidth=2, zorder=3,
                    label=LABELS[key] if not label_added else "_nolegend_")
            for k, h in zip(ks_capped, hs_capped):
                ax.annotate(f">{h_max_tested}?", xy=(k, h),
                            xytext=(k + 0.05, h + y_cap * 0.03),
                            fontsize=8, color=COLORS[key], va="bottom")

        # Dashed connector between known and capped regions
        if ks_known and ks_capped:
            ax.plot([ks_known[-1], ks_capped[0]],
                    [hs_known[-1], hs_capped[0]],
                    "--", color=COLORS[key], linewidth=1, alpha=0.5, zorder=2)

    ax.set_xlabel("K  (queries per trial)", fontsize=12)
    ax.set_ylabel(f"Min. hidden neurons  (≥{tau*100:.0f}% accuracy)", fontsize=11)
    ax.set_title(f"Neuron Budget vs Query Load  (τ = {tau:.0%}, Plan D)", fontsize=13)
    ax.set_xticks(K_values)
    ax.set_ylim(0, y_cap * 1.15)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_neuron_efficiency(
    K_values: List[int],
    min_h_with_delay: Dict[int, Optional[int]],
    min_h_no_delay: Dict[int, Optional[int]],
    save_path: str = "neuron_efficiency.png",
):
    """
    Two-panel efficiency figure — both panels use line/curve style.

    Top panel  — Efficiency curve: h_no_delay(K) / h_with_delay(K).
                 Should rise with K as delays become more advantageous.
    Bottom panel — Neurons per query h/K for both conditions.
                 Delay line should be sub-linear; no-delay line linear.
    """
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    Ks_common = sorted(k for k in K_values
                       if min_h_with_delay.get(k) and min_h_no_delay.get(k))

    # ── Top: efficiency curve ──
    ax = axes[0]
    if Ks_common:
        ratios = [min_h_no_delay[k] / min_h_with_delay[k] for k in Ks_common]
        ax.plot(Ks_common, ratios, "o-", color="#2E7D32", linewidth=2.5,
                markersize=9, markerfacecolor="white", markeredgewidth=2.5,
                zorder=3)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2,
                   label="Ratio = 1  (no benefit)")
        for k, r in zip(Ks_common, ratios):
            ax.text(k, r + 0.05, f"{r:.1f}×", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#2E7D32")
        ax.set_ylim(0, max(ratios) * 1.25)
    else:
        ax.text(0.5, 0.5,
                "Run h-sweep to fill data\n(both conditions must reach threshold)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray", style="italic")
        ax.set_ylim(0, 5)
    ax.set_xlabel("K  (queries per trial)", fontsize=11)
    ax.set_ylabel("Efficiency  =  h(d=0) / h(delay)", fontsize=11)
    ax.set_title("Delay Efficiency Curve", fontsize=12)
    ax.set_xticks(K_values)
    if Ks_common:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Bottom: neurons per query ──
    ax = axes[1]
    COLORS = {"delay": "#2196F3", "no_delay": "#F44336"}
    LABELS = {"delay": "Trainable delays (sub-linear)", "no_delay": "No delay (linear)"}
    for key, data in [("delay", min_h_with_delay), ("no_delay", min_h_no_delay)]:
        ks = sorted(k for k in K_values if data.get(k))
        npq = [data[k] / k for k in ks]
        ax.plot(ks, npq, "o-", color=COLORS[key], linewidth=2, markersize=7,
                label=LABELS[key])

    ax.set_xlabel("K  (queries per trial)", fontsize=11)
    ax.set_ylabel("Neurons per query  (h / K)", fontsize=11)
    ax.set_title("Amortized Neuron Cost per Query", fontsize=12)
    ax.set_xticks(K_values)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(h_pad=1.5)
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# New diagnostic plots (2026-06): heatmap, degradation curves, clean capacity
# ─────────────────────────────────────────────────────────────────────────────

def _parse_sweep_results(summary: dict, H_VALS, K_VALS):
    """Extract per-seed accuracy from summary["<cond>"]["results"] dicts.

    Returns two dicts keyed by (h, K, seed) → accuracy for each condition.
    """
    import re as _re
    out = {}
    for cond in ("w_and_d", "d0"):
        data = {}
        for key, val in summary.get(cond, {}).get("results", {}).items():
            m = _re.match(r"h(\d+)_K(\d+)_seed(\d+)", key)
            if not m:
                continue
            h, K, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if h in H_VALS and K in K_VALS:
                data[(h, K, seed)] = val["accuracy"]
        out[cond] = data
    return out


def plot_depth_ablation_curves(all_results: list, save_path: str):
    """Two-panel summary for depth ablation experiment.

    Left : accuracy vs K per model variant, shaded over seeds.
    Right: throughput (K/spk) vs K per model variant.
    90% threshold dashed line on left panel.
    """
    from collections import defaultdict

    # Build {model_name: {K: [acc, ...]}} and {model_name: {K: [kpsp, ...]}}
    acc_by: dict  = defaultdict(lambda: defaultdict(list))
    spk_by: dict  = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        mname = r.get("model_name", "unknown")
        K     = r.get("K")
        if K is None:
            continue
        if r.get("accuracy") is not None:
            acc_by[mname][K].append(r["accuracy"])
        v = r.get("throughput_K_per_spk")
        if v is not None and v == v:   # nan check
            spk_by[mname][K].append(v)

    model_names = sorted(acc_by.keys())
    COLORS = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2",
              "#D32F2F", "#0288D1", "#558B2F", "#E64A19"]
    MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, mname in enumerate(model_names):
        color  = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        K_list = sorted(acc_by[mname].keys())

        # ── Accuracy panel ──
        means = [float(np.mean(acc_by[mname][K])) for K in K_list]
        lo    = [float(min(acc_by[mname][K])) for K in K_list]
        hi    = [float(max(acc_by[mname][K])) for K in K_list]
        axes[0].plot(K_list, means, color=color, linewidth=2,
                     marker=marker, markersize=8, label=mname)
        axes[0].fill_between(K_list, lo, hi, color=color, alpha=0.12)

        # ── Throughput panel ──
        K_spk = sorted(spk_by[mname].keys())
        if K_spk:
            means_spk = [float(np.mean(spk_by[mname][K])) for K in K_spk]
            lo_spk    = [float(min(spk_by[mname][K])) for K in K_spk]
            hi_spk    = [float(max(spk_by[mname][K])) for K in K_spk]
            axes[1].plot(K_spk, means_spk, color=color, linewidth=2,
                         marker=marker, markersize=8, label=mname)
            axes[1].fill_between(K_spk, lo_spk, hi_spk, color=color, alpha=0.12)

    # 90% threshold
    axes[0].axhline(0.90, color="#333", linestyle=":", linewidth=1.5)
    axes[0].text(min(K_list) + 0.05 if K_list else 2, 0.902, "τ = 90%",
                 va="bottom", fontsize=9, color="#333")

    K_all = sorted({r["K"] for r in all_results if r.get("K") is not None})
    for ax in axes:
        ax.set_xticks(K_all)
        ax.set_xlabel("K  (queries per trial)", fontsize=12)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Accuracy vs K", fontsize=13)
    axes[0].set_ylim(0.70, 1.01)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[0].legend(fontsize=9, loc="lower left", framealpha=0.9)

    axes[1].set_ylabel("Throughput  K / spikes", fontsize=12)
    axes[1].set_title("Energy-Normalised Throughput vs K", fontsize=13)
    axes[1].legend(fontsize=9, loc="upper right", framealpha=0.9)

    fig.suptitle("Depth Ablation — Plan D Sequential NAND  (shading = seed range)",
                 fontsize=13)
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_heatmap(summary: dict, save_path: str):
    """Two side-by-side heatmaps: w_and_d (left) and d0 (right).

    Color encodes mean accuracy over seeds at each (h, K) cell.
    A white dashed contour marks the 90% threshold on both panels.
    """
    H_VALS = [10, 20, 30, 50]
    K_VALS = [1, 2, 3, 4, 5]
    SEEDS = [42, 0]

    per_seed = _parse_sweep_results(summary, H_VALS, K_VALS)

    def _mean_matrix(cond):
        mat = np.full((len(H_VALS), len(K_VALS)), np.nan)
        for i, h in enumerate(H_VALS):
            for j, K in enumerate(K_VALS):
                vals = [per_seed[cond].get((h, K, s)) for s in SEEDS
                        if (h, K, s) in per_seed[cond]]
                if vals:
                    mat[i, j] = float(np.mean(vals))
        return mat

    mat_wd = _mean_matrix("w_and_d")
    mat_d0 = _mean_matrix("d0")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    vmin, vmax = 0.74, 1.0
    cmap = "RdYlGn"

    X_c = np.arange(len(K_VALS)) + 0.5
    Y_c = np.arange(len(H_VALS)) + 0.5
    XX, YY = np.meshgrid(X_c, Y_c)

    titles = ["Trainable delays (w & d)", "No delays (d = 0)"]
    for ax, mat, title in zip(axes, [mat_wd, mat_d0], titles):
        im = ax.pcolormesh(mat, cmap=cmap, vmin=vmin, vmax=vmax)

        # 90% contour (may not cross the d0 panel, which is fine)
        try:
            ax.contour(XX, YY, mat, levels=[0.90],
                       colors="white", linestyles="--", linewidths=1.8)
        except Exception:
            pass

        # Cell annotations
        for i in range(len(H_VALS)):
            for j in range(len(K_VALS)):
                v = mat[i, j]
                if not np.isnan(v):
                    txt_color = "white" if v < 0.78 or v > 0.97 else "black"
                    ax.text(j + 0.5, i + 0.5, f"{v * 100:.1f}%",
                            ha="center", va="center", fontsize=10,
                            fontweight="bold", color=txt_color)

        ax.set_xticks(X_c)
        ax.set_xticklabels([f"K={K}" for K in K_VALS], fontsize=10)
        ax.set_yticks(Y_c)
        ax.set_yticklabels([f"h={h}" for h in H_VALS], fontsize=10)
        ax.set_xlabel("Queries per trial (K)", fontsize=11)
        ax.set_ylabel("Hidden neurons (h)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=axes[1], shrink=0.85, label="Accuracy")
    fig.suptitle(
        "NAND Accuracy by Network Size and Query Load  (Plan D, MLP readout)",
        fontsize=13)
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_curves(summary: dict, save_path: str):
    """Accuracy vs K for every (condition, h) combination.

    Blue family = trainable delays (w&d); red family = no delays (d=0).
    Shaded band spans min/max over the two seeds.
    Horizontal dashed line marks the 90% threshold.
    """
    H_VALS = [10, 20, 30, 50]
    K_VALS = [1, 2, 3, 4, 5]
    SEEDS = [42, 0]

    per_seed = _parse_sweep_results(summary, H_VALS, K_VALS)

    COLORS_WD = ["#90CAF9", "#42A5F5", "#1565C0", "#0D2F6B"]
    COLORS_D0 = ["#FFCDD2", "#EF9A9A", "#C62828", "#7B1517"]
    MARKERS = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for h_idx, h in enumerate(H_VALS):
        for cond, colors in [("w_and_d", COLORS_WD), ("d0", COLORS_D0)]:
            accs_by_K = {}
            for K in K_VALS:
                vals = [per_seed[cond].get((h, K, s)) for s in SEEDS
                        if (h, K, s) in per_seed[cond]]
                if vals:
                    accs_by_K[K] = vals

            if not accs_by_K:
                continue

            K_plot = sorted(accs_by_K)
            means = [float(np.mean(accs_by_K[K])) for K in K_plot]
            lo = [float(min(accs_by_K[K])) for K in K_plot]
            hi = [float(max(accs_by_K[K])) for K in K_plot]
            color = colors[h_idx]
            ls = "-" if cond == "w_and_d" else "--"

            ax.plot(K_plot, means, color=color, linewidth=2,
                    marker=MARKERS[h_idx], markersize=7, linestyle=ls)
            ax.fill_between(K_plot, lo, hi, color=color, alpha=0.12)

            # Annotate h value at the end of each line
            ax.annotate(
                f"h={h}",
                xy=(K_plot[-1], means[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8, color=color, va="center",
            )

    ax.axhline(0.90, color="#333333", linestyle=":", linewidth=1.5)
    ax.text(K_VALS[-1] - 0.05, 0.902, "τ = 90%", ha="right", va="bottom",
            fontsize=9, color="#333333")

    ax.set_xlabel("K  (queries per trial)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("NAND Accuracy vs Query Load  (Plan D, MLP readout)", fontsize=13)
    ax.set_xticks(K_VALS)
    ax.set_xlim(0.7, K_VALS[-1] + 0.7)
    ax.set_ylim(0.72, 1.01)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Compact legend: condition families only
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#1565C0", label="Trainable delays (w & d)  — solid lines"),
        Patch(facecolor="#C62828", label="No delays (d = 0)  — dashed lines"),
        Line2D([0], [0], color="#555", linewidth=0.5,
               label="darker shade = more neurons (h=10→50)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower left",
              framealpha=0.92)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_clean(
    K_values: List[int],
    min_h_wd: Dict[int, Optional[int]],
    summary: dict,
    save_path: str,
    tau: float = 0.90,
):
    """Clean capacity curve for w_and_d only.

    Known threshold points shown as filled circles + solid line.
    Unknown points (K where threshold is not met) shown as open circles at
    h_max_tested, annotated with actual accuracy.
    A text box summarises the d0 ceiling so no floating markers are needed.
    Secondary y-axis shows neurons/query (h/K) for the known points.
    """
    import re as _re

    # Determine h values tested and h_max
    all_h = sorted({
        int(_re.match(r"h(\d+)_", k).group(1))
        for k in summary.get("w_and_d", {}).get("results", {})
        if _re.match(r"h(\d+)_", k)
    })
    h_max = max(all_h) if all_h else 50

    # Mean w_and_d accuracy at h_max for each K
    acc_at_hmax = {}
    for K in K_values:
        vals = [v["accuracy"]
                for key, v in summary.get("w_and_d", {}).get("results", {}).items()
                if _re.match(rf"h{h_max}_K{K}_seed\d+", key)]
        if vals:
            acc_at_hmax[K] = float(np.mean(vals))

    # Max d0 accuracy across all runs
    d0_accs = [v["accuracy"]
               for v in summary.get("d0", {}).get("results", {}).values()
               if isinstance(v, dict) and "accuracy" in v]
    max_d0_acc = max(d0_accs) if d0_accs else None

    BLUE = "#1976D2"
    fig, ax = plt.subplots(figsize=(7, 5))

    ks_known = sorted(k for k in K_values if min_h_wd.get(k) is not None)
    ks_open = sorted(k for k in K_values if min_h_wd.get(k) is None)

    if ks_known:
        hs_known = [min_h_wd[k] for k in ks_known]
        ax.plot(ks_known, hs_known, "o-", color=BLUE, linewidth=2.5,
                markersize=9, zorder=3,
                label=f"w&d: min h to reach {tau:.0%}")

    for k in ks_open:
        ax.plot(k, h_max, "o", color=BLUE, markersize=10,
                markerfacecolor="white", markeredgewidth=2.0, zorder=3)
        acc = acc_at_hmax.get(k)
        note = f"{acc * 100:.1f}%\n(below τ)" if acc is not None else "(below τ)"
        ax.annotate(note, xy=(k, h_max), xytext=(k + 0.05, h_max + 2),
                    fontsize=8.5, color=BLUE, va="bottom")

    # Dashed connector from last known to first open
    if ks_known and ks_open:
        ax.plot([ks_known[-1], ks_open[0]],
                [min_h_wd[ks_known[-1]], h_max],
                "--", color=BLUE, linewidth=1.2, alpha=0.5)

    # Secondary y-axis: neurons / query
    ax2 = ax.twinx()
    if ks_known:
        npq = [min_h_wd[k] / k for k in ks_known]
        ax2.plot(ks_known, npq, "s--", color="#78909C", linewidth=1.5,
                 markersize=7, alpha=0.75, label="h / K  (neurons/query)")
        ax2.set_ylabel("Neurons / query  (h / K)", fontsize=10, color="#546E7A")
        ax2.tick_params(axis="y", labelcolor="#546E7A")
        ax2.set_ylim(0, max(npq) * 3.0)

    # d=0 annotation box
    h_set_str = "{" + ", ".join(str(h) for h in all_h) + "}"
    if max_d0_acc is not None:
        d0_body = (f"d=0 (no delays):\nmax accuracy = {max_d0_acc * 100:.1f}%\n"
                   f"Never reaches τ = {tau:.0%}\n"
                   f"at any h ∈ {h_set_str}")
    else:
        d0_body = f"d=0 (no delays):\nNever reaches τ = {tau:.0%}"
    ax.text(0.97, 0.97, d0_body,
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color="#C62828",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE",
                      edgecolor="#E53935", linewidth=1.2))

    ax.set_xlabel("K  (queries per trial)", fontsize=12)
    ax.set_ylabel(f"Min. hidden neurons  (≥ {tau * 100:.0f}% accuracy)", fontsize=11)
    ax.set_title(f"Neuron Budget vs Query Load  (τ = {tau:.0%}, Plan D)", fontsize=13)
    ax.set_xticks(K_values)
    ax.set_ylim(0, (h_max + 10) * 1.65)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-run diagnostic visualization system (depth ablation, 2026-06)
# Compatible with 1-layer and 2-layer SNNSimultaneousModel.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_run_traces(model, cfg: dict, K: int, op: str, device: str,
                        seed: int = 999, dataset_override: tuple | None = None) -> tuple:
    """Record one sample forward pass; return (traces, weights_dict, delays_dict).

    Imports are local to avoid circular dependencies at module level.

    dataset_override : optional pre-built single-sample (A, B, op_ids, labels)
        tuple, bypassing the internal MultiQueryDataset construction. Needed
        by topologies whose input doesn't fit the K-sequential-sub-window
        dataset shape assumed below (e.g. one-query-many-op's
        BroadcastOpDataset, where A/B are a single shared query while labels
        span n_ops output heads).
    """
    import torch
    from data.boolean_dataset import MultiQueryDataset, FixedOperationQueryDataset
    from data.encoding import encode_sequential_trial, encode_simultaneous_trial

    model.eval()
    n_ops = cfg.get("n_ops", 0)
    # Burst encoding params — read from cfg so diagnostic plots match training encoding
    _enc_kwargs = dict(
        encoding_mode=cfg.get("encoding_mode", "rate"),
        burst_n_spikes_on=cfg.get("burst_n_spikes_on", 2),
        burst_n_spikes_off=cfg.get("burst_n_spikes_off", 1),
        burst_phase_on=cfg.get("burst_phase_on", 0.2),
        burst_phase_off=cfg.get("burst_phase_off", 0.8),
        burst_jitter_ms=cfg.get("burst_jitter_ms", 0),
        one_hot_phase=cfg.get("one_hot_phase", 1.0),
        one_hot_n_spikes=cfg.get("one_hot_n_spikes", 1),
    )
    if dataset_override is not None:
        A, B, op_ids, labels = dataset_override
        torch.manual_seed(seed)
        spike_input = encode_sequential_trial(
            A.unsqueeze(0), B.unsqueeze(0),
            win_len=cfg["win_len"], read_len=cfg["read_len"],
            r_on=cfg["r_on"], r_off=cfg["r_off"],
            dt=cfg["dt"], device=device,
            op_ids=op_ids.unsqueeze(0) if n_ops > 0 else None,
            n_ops=n_ops,
            **_enc_kwargs,
        )
    else:
        # Retry with increasing seeds until we get non-zero input spikes.
        # Needed because K=1 + short 10ms window + r_off=10Hz gives ~90%
        # chance of all-zero spikes when A=0 or B=0.
        for _try in range(10):
            _seed = seed + _try
            if cfg.get("input_schedule") == "simultaneous" and cfg.get("query_ops"):
                ds = FixedOperationQueryDataset(
                    n_samples=1, query_ops=cfg["query_ops"], seed=_seed,
                )
            elif op == "mixed":
                ds = MultiQueryDataset(K=K, n_samples=1, same_op=False,
                                       ops_list=cfg["ops_list"], seed=_seed,
                                       op_sampling=cfg.get("op_sampling", "uniform"))
            else:
                ds = MultiQueryDataset(K=K, n_samples=1, same_op=True,
                                       op_name=op, ops_list=cfg["ops_list"],
                                       seed=_seed)
            A, B, op_ids, labels = ds[0]
            torch.manual_seed(_seed)
            encoder = (encode_simultaneous_trial
                       if cfg.get("input_schedule") == "simultaneous"
                       else encode_sequential_trial)
            spike_input = encoder(
                A.unsqueeze(0), B.unsqueeze(0),
                win_len=cfg["win_len"], read_len=cfg["read_len"],
                r_on=cfg["r_on"], r_off=cfg["r_off"],
                dt=cfg["dt"], device=device,
                op_ids=op_ids.unsqueeze(0) if n_ops > 0 else None,
                n_ops=n_ops,
                **_enc_kwargs,
            )
            if spike_input.sum() > 0:
                break

    with torch.no_grad():
        logits, info = model(spike_input.to(device), record=True)

    T_val = spike_input.shape[1]
    traces: dict = {
        "input_spikes":   spike_input[0].cpu().numpy(),           # [T, n_input]
        "hidden1_spikes": info["hidden_spike_train"][0].numpy(),   # [T, h_last]
        "win_len":  cfg["win_len"],
        "read_len": cfg["read_len"],
        "sub_win":  cfg.get("sub_win"),
        "observation_mode": cfg.get("observation_mode", "late_window"),
        "output_window_len": cfg.get("output_window_len"),
        "input_schedule": cfg.get("input_schedule"),
        "query_ops": np.asarray(cfg.get("query_ops", [])),
        "readout_endpoint": cfg.get("readout_endpoint", ""),
        "topology_type": cfg.get(
            "topology_type", getattr(model, "topology_type", "shared_dense")
        ),
        "hidden_per_query": (
            cfg.get("surface_hidden_width")
            if hasattr(model, "syn_ih_modules") else None
        ),
        "encoding_mode": cfg.get("encoding_mode", "rate"),
        "one_hot_n_spikes": cfg.get("one_hot_n_spikes", 1),
        "opponent_target_timing_mode": cfg.get("opponent_target_timing_mode", ""),
        "output_target_offset_steps": cfg.get("output_target_offset_steps", -1),
        "target_filter_tau_steps": cfg.get("target_filter_tau_steps", -1),
        "K":        K,
        "T":        T_val,
        "output_logits": logits[0].detach().cpu().numpy(),        # [K] per-query logits
        "output_probabilities": torch.sigmoid(logits[0]).detach().cpu().numpy(),
        "output_predictions": (logits[0] > 0).to(torch.int64).detach().cpu().numpy(),
        "query_labels": labels.detach().cpu().numpy(),
        "output_semantics": "spiking_output" if "output_spike_train" in info else "decoder_decision",
    }
    if "hidden_membrane_train" in info:
        traces["hidden_membrane"] = info["hidden_membrane_train"][0].numpy()
    if "output_spike_train" in info:
        traces["output_spikes"] = info["output_spike_train"][0].cpu().numpy()  # [T, n_out]
    if "output_membrane_train" in info:
        traces["output_membrane"] = info["output_membrane_train"][0].cpu().numpy()
    if "output_window_counts" in info:
        traces["output_window_counts"] = info["output_window_counts"][0].numpy()
    if "output_pair_counts" in info:
        traces["output_pair_counts"] = info["output_pair_counts"][0].numpy()
    if "hidden1_spike_train" in info:
        # 2-layer: hidden_spike_train = h2; hidden1_spike_train = h1
        traces["hidden2_spikes"] = traces.pop("hidden1_spikes")      # rename h_last→h2
        traces["hidden1_spikes"] = info["hidden1_spike_train"][0].numpy()  # h1

    # Weights
    if hasattr(model, "syn_ih_modules"):
        hidden_per_query = int(model.hidden_per_query)
        combined_weights = np.zeros((model.n_input, model.n_hidden), dtype=np.float32)
        combined_delays = np.zeros_like(combined_weights)
        for query, layer in enumerate(model.syn_ih_modules):
            in_slice = slice(4 * query, 4 * (query + 1))
            hidden_slice = slice(
                hidden_per_query * query, hidden_per_query * (query + 1)
            )
            combined_weights[in_slice, hidden_slice] = (
                layer.weight.detach().cpu().numpy()
            )
            combined_delays[in_slice, hidden_slice] = (
                layer.get_delays().detach().cpu().numpy()
            )
        weights_dict: dict = {"ih": combined_weights}
    else:
        weights_dict = {
            "ih": model.syn_ih.weight.detach().cpu().numpy(),
        }
    if hasattr(model, "syn_h1h2"):
        weights_dict["h1h2"] = model.syn_h1h2.weight.detach().cpu().numpy()
    if hasattr(model, "syn_ho"):
        weights_dict["ho"] = model.syn_ho.weight.detach().cpu().numpy()
    if model.readout_type == "linear" and model.readout is not None:
        weights_dict["readout"] = model.readout.weight.detach().cpu().numpy()
    elif model.readout_type == "mlp" and model.readout is not None:
        linear_layers = [
            layer for layer in model.readout.modules()
            if isinstance(layer, torch.nn.Linear)
        ]
        if linear_layers:
            weights_dict["readout_in"] = (
                linear_layers[0].weight.detach().cpu().numpy()
            )
        if len(linear_layers) > 1:
            weights_dict["readout_out"] = (
                linear_layers[-1].weight.detach().cpu().numpy()
            )

    # Delays
    if hasattr(model, "syn_ih_modules"):
        delays_dict: dict = {"ih": combined_delays}
    else:
        raw_delays = model.get_delays()
        delays_dict = {k: v.detach().cpu().numpy() for k, v in raw_delays.items()}

    return traces, weights_dict, delays_dict


def _opponent_target_times(values: dict) -> list[int]:
    """Return declared target-spike timesteps for diagnostic overlays."""
    mode = values.get("opponent_target_timing_mode")
    if not mode:
        return []
    input_steps = int(values.get("win_len", 0))
    window_len = int(values.get("output_window_len") or 0)
    if window_len <= 0:
        return []
    if mode in {"simultaneous_center", "sequential_centers"}:
        within_window = window_len // 2
    elif mode in {"simultaneous_offset", "sequential_offsets"}:
        within_window = int(values.get("output_target_offset_steps", 0))
    else:
        return []
    queries = int(values.get("K", 1))
    if mode.startswith("sequential"):
        return [input_steps + q * window_len + within_window for q in range(queries)]
    return [input_steps + within_window]


def plot_weight_heatmaps(
    weights_dict: dict,
    save_path: str,
    title: str = "",
    axes=None,
):
    """Heatmaps for all synaptic weight matrices.

    Parameters
    ----------
    weights_dict : {"ih": W, "h1h2": W (opt), "readout": W (opt)}
    axes         : pre-allocated list of axes for embedding in diagnostic panel.
                   If None, a new figure is created and saved.
    """
    keys_order = [
        key for key in ("ih", "h1h2", "readout", "readout_in", "readout_out")
        if key in weights_dict
    ]
    n_panels   = len(keys_order)
    if n_panels == 0:
        return

    PANEL_TITLES = {
        "ih":      "Input → Hidden1  weights",
        "h1h2":    "Hidden1 → Hidden2  weights",
        "readout": "Hidden → Readout  weights",
        "readout_in": "Hidden → MLP hidden weights",
        "readout_out": "MLP hidden → logit weights",
    }

    own_fig = axes is None
    if own_fig:
        fig, axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5),
                                squeeze=False)
        axs = axs[0]
    else:
        axs = axes

    for ax, key in zip(axs, keys_order):
        W = weights_dict[key]
        vabs = max(float(np.abs(W).max()), 1e-6)
        im = ax.imshow(W, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto",
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Weight")
        ax.set_title(PANEL_TITLES.get(key, key), fontsize=10)
        ax.set_xlabel("Post-synaptic neuron", fontsize=9)
        ax.set_ylabel("Pre-synaptic neuron", fontsize=9)

    # Hide extra axes when fewer panels than allocated
    if not own_fig:
        for ax in axs[n_panels:]:
            ax.set_visible(False)

    if own_fig:
        if title:
            fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)


def plot_delay_heatmaps(
    delays_dict: dict,
    save_path: str,
    title: str = "",
    axes=None,
):
    """Heatmaps for all delay matrices (sequential colormap, vmin=0).

    Parameters
    ----------
    delays_dict : {"ih": D, "h1h2": D (opt)}
    axes        : pre-allocated axes list; None → own figure.
    """
    keys_order = [k for k in ("ih", "h1h2") if k in delays_dict]
    n_panels   = len(keys_order)
    if n_panels == 0:
        return

    PANEL_TITLES = {
        "ih":   "Input → Hidden1  delays",
        "h1h2": "Hidden1 → Hidden2  delays",
    }

    own_fig = axes is None
    if own_fig:
        fig, axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5),
                                squeeze=False)
        axs = axs[0]
    else:
        axs = axes

    for ax, key in zip(axs, keys_order):
        D = delays_dict[key]
        im = ax.imshow(D, cmap="viridis", aspect="auto", vmin=0,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Delay (steps)")
        ax.set_title(PANEL_TITLES.get(key, key), fontsize=10)
        ax.set_xlabel("Post-synaptic neuron", fontsize=9)
        ax.set_ylabel("Pre-synaptic neuron", fontsize=9)

    if not own_fig:
        for ax in axs[n_panels:]:
            ax.set_visible(False)

    if own_fig:
        if title:
            fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)


def plot_spike_raster_layers(
    traces: dict,
    save_path: str,
    title: str = "",
    win_len: Optional[int] = None,
    read_len: Optional[int] = None,
    sub_win: Optional[int] = None,
    K: int = 1,
):
    """Multi-layer spike raster for 1-layer or 2-layer models.

    Panels (top→bottom): Input | Hidden1 | Hidden2 (if present)
    """
    s_in  = traces["input_spikes"]   # [T, n_in]
    s_h1  = traces["hidden1_spikes"] # [T, h1]
    s_h2  = traces.get("hidden2_spikes")  # [T, h2] or None
    s_out = traces.get("output_spikes")   # [T, n_out] or None

    T, n_input  = s_in.shape
    _, h1        = s_h1.shape

    win_len  = win_len  or traces.get("win_len",  T)
    read_len = read_len or traces.get("read_len", 0)
    sub_win  = sub_win  or traces.get("sub_win")
    K        = K        or traces.get("K", 1)

    COLORS = list(plt.cm.tab10.colors)
    layers = [("Input",   s_in,  "#333333"),
              ("Hidden1", s_h1,  "steelblue")]
    if s_h2 is not None:
        layers.append(("Hidden2", s_h2, "darkorange"))
    if s_out is not None:
        layers.append(("Output", s_out, "#2ca02c"))

    heights = [max(ldata.shape[1], 2) for _, ldata, _ in layers]
    fig, axes = plt.subplots(
        len(layers), 1, sharex=True,
        figsize=(12, 1.5 + 0.18 * sum(heights)),
        gridspec_kw={"height_ratios": heights},
    )
    if len(layers) == 1:
        axes = [axes]

    for ax, (lname, ldata, col) in zip(axes, layers):
        n_neurons = ldata.shape[1]
        for n in range(n_neurons):
            ts = np.where(ldata[:, n] > 0)[0]
            if len(ts) == 0:
                continue
            if lname == "Input" and n_input > 2:
                c = COLORS[(n // 2) % len(COLORS)]
            else:
                c = col
            ax.scatter(ts, np.full(len(ts), n, dtype=float),
                       s=5, color=c, marker="|", linewidths=0.8)
        ax.set_ylim(-0.5, n_neurons - 0.5)
        ax.set_ylabel(lname, fontsize=9)
        ax.set_yticks(range(0, n_neurons, max(1, n_neurons // 8)))
        ax.axvspan(0, win_len, alpha=0.07, color="royalblue", zorder=0)
        ax.axvspan(win_len, win_len + read_len, alpha=0.10, color="tomato", zorder=0)
        ax.set_xlim(-0.5, T - 0.5)

        if sub_win is not None and K > 1:
            for k in range(K):
                t0 = k * sub_win
                ax.axvline(t0, color=COLORS[k % len(COLORS)],
                           linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
                ax.text(t0 + 0.3, n_neurons - 0.5, f"Q{k}",
                        fontsize=6, color=COLORS[k % len(COLORS)],
                        va="top", fontweight="bold", zorder=2)

    axes[-1].set_xlabel("Timestep (ms)", fontsize=9)
    if title:
        axes[0].set_title(title, fontsize=9)

    fig.subplots_adjust(hspace=0.08)
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_layer_to_layer_spike_flow(
    traces: dict,
    weights_dict: dict,
    delays_dict: dict,
    save_path: str,
    weight_threshold: float = 0.1,
    max_edges: int = 2000,
    alpha: float = 0.08,
    title: str = "",
):
    """Directed spike flow diagram using LineCollection for performance.

    Draws connections from every pre-synaptic spike to its post-synaptic
    arrival time t_pre + delay, filtered by |weight| > weight_threshold.
    """
    from matplotlib.collections import LineCollection

    s_in  = traces["input_spikes"]   # [T, n_in]
    s_h1  = traces["hidden1_spikes"] # [T, h1]
    s_h2  = traces.get("hidden2_spikes")  # [T, h2] or None
    s_out = traces.get("output_spikes")   # [T, n_out] or None

    T, n_input = s_in.shape
    _, h1 = s_h1.shape
    GAP = max(3, n_input // 4 + 1)

    # ── Y-axis offsets ──
    y0_in   = 0
    y0_h1   = n_input + GAP
    y0_h2   = (y0_h1 + h1 + GAP) if s_h2 is not None else None
    y_read  = (y0_h2 + s_h2.shape[1] + GAP if s_h2 is not None
               else y0_h1 + h1 + GAP)
    y0_out  = y_read + GAP if s_out is not None else None

    def y_in(n):  return float(y0_in + n)
    def y_h1(n):  return float(y0_h1 + n)
    def y_h2(n):  return float(y0_h2 + n) if y0_h2 is not None else 0.0
    def y_out(n): return float(y0_out + n) if y0_out is not None else 0.0

    win_len  = traces.get("win_len",  T)
    read_len = traces.get("read_len", 0)
    sub_win  = traces.get("sub_win")
    K        = traces.get("K", 1)
    COLORS   = list(plt.cm.tab10.colors)

    fig, ax = plt.subplots(figsize=(13, max(5, int(y_read * 0.28 + 3))))

    # ── Draw spikes ──
    for n in range(n_input):
        ts = np.where(s_in[:, n] > 0)[0]
        if len(ts):
            c = COLORS[(n // 2) % len(COLORS)] if n_input > 2 else "#333333"
            ax.scatter(ts, np.full(len(ts), y_in(n)),
                       s=6, color=c, marker="|", linewidths=1.0, zorder=3)

    for n in range(h1):
        ts = np.where(s_h1[:, n] > 0)[0]
        if len(ts):
            ax.scatter(ts, np.full(len(ts), y_h1(n)),
                       s=6, color="steelblue", marker="|", linewidths=1.0, zorder=3)

    if s_h2 is not None:
        for n in range(s_h2.shape[1]):
            ts = np.where(s_h2[:, n] > 0)[0]
            if len(ts):
                ax.scatter(ts, np.full(len(ts), y_h2(n)),
                           s=6, color="darkorange", marker="|", linewidths=1.0, zorder=3)

    if s_out is not None:
        for n in range(s_out.shape[1]):
            ts = np.where(s_out[:, n] > 0)[0]
            if len(ts):
                ax.scatter(ts, np.full(len(ts), y_out(n)),
                           s=8, color="#2ca02c", marker="|", linewidths=1.2, zorder=3)

    # ── Collect directed connections ──
    def _collect_edges(spike_train, pre_y_fn, post_y_fn, W, D):
        """Yield (t_pre, y_pre, t_post, y_post, w) for all fired pre-neurons."""
        edges = []
        n_pre, n_post = W.shape
        for i in range(n_pre):
            ts_pre = np.where(spike_train[:, i] > 0)[0]
            if len(ts_pre) == 0:
                continue
            for j in range(n_post):
                w = float(W[i, j])
                if abs(w) <= weight_threshold:
                    continue
                d = float(D[i, j])
                yp = pre_y_fn(i)
                yq = post_y_fn(j)
                for t in ts_pre:
                    arr = t + d
                    if 0 <= arr < T:
                        edges.append((float(t), yp, float(arr), yq, w))
        return edges

    all_edges = []
    if "ih" in weights_dict and "ih" in delays_dict:
        all_edges += _collect_edges(s_in, y_in, y_h1,
                                    weights_dict["ih"], delays_dict["ih"])
    if "h1h2" in weights_dict and "h1h2" in delays_dict and s_h2 is not None:
        all_edges += _collect_edges(s_h1, y_h1, y_h2,
                                    weights_dict["h1h2"], delays_dict["h1h2"])
    if s_out is not None and "ho" in weights_dict and "ho" in delays_dict:
        s_h_last = s_h2 if s_h2 is not None else s_h1
        y_h_last = y_h2 if s_h2 is not None else y_h1
        all_edges += _collect_edges(s_h_last, y_h_last, y_out,
                                    weights_dict["ho"], delays_dict["ho"])

    # Sort by |w| descending; keep top max_edges
    all_edges.sort(key=lambda e: -abs(e[4]))
    selected = all_edges[:max_edges]

    if selected:
        segs   = [[(e[0], e[1]), (e[2], e[3])] for e in selected]
        colors = [(1.0, 0.0, 0.0, alpha) if e[4] > 0 else (0.0, 0.0, 1.0, alpha)
                  for e in selected]
        lc = LineCollection(segs, colors=colors, linewidths=0.5, zorder=2)
        ax.add_collection(lc)

    # ── Band backgrounds & labels ──
    band_defs = [
        (y0_in - 0.6,  y0_in + n_input - 0.4,  "royalblue", "Input"),
        (y0_h1 - 0.6,  y0_h1 + h1 - 0.4,       "steelblue", "Hidden1"),
    ]
    if s_h2 is not None and y0_h2 is not None:
        band_defs.append((y0_h2 - 0.6, y0_h2 + s_h2.shape[1] - 0.4,
                          "sandybrown", "Hidden2"))
    if s_out is not None and y0_out is not None:
        band_defs.append((y0_out - 0.6, y0_out + s_out.shape[1] - 0.4,
                          "#2ca02c", "Output"))

    for yb, yt, col, lbl in band_defs:
        ax.axhspan(yb, yt, alpha=0.07, color=col, zorder=0)
        ax.text(-1.2, (yb + yt) / 2, lbl, ha="right", va="center",
                fontsize=9, fontweight="bold", color=col)

    # Readout line shown only when no spiking output layer (spiking output
    # replaces the readout conceptually — showing both is confusing).
    if s_out is None:
        ax.axhline(y_read, xmin=win_len / T, xmax=1.0,
                   color="tomato", linewidth=2.0, alpha=0.6, zorder=2)
        ax.text(win_len + (T - win_len) / 2, y_read + 0.3, "Readout",
                ha="center", va="bottom", fontsize=8, color="tomato")

    # ── Window shading ──
    ax.axvspan(0, win_len, alpha=0.05, color="royalblue", zorder=0)
    ax.axvspan(win_len, win_len + read_len, alpha=0.08, color="tomato", zorder=0)
    if sub_win is not None and K > 1:
        for k in range(K):
            ax.axvline(k * sub_win, color=COLORS[k % len(COLORS)],
                       linestyle=":", linewidth=1.0, alpha=0.6, zorder=1)
            ax.text(k * sub_win + 0.3, y0_in - 0.7, f"Q{k}",
                    fontsize=7, color=COLORS[k % len(COLORS)],
                    va="top", fontweight="bold")

    ax.set_xlim(-1.5, T)
    y_top = (y0_out + s_out.shape[1] + 1.2
             if (s_out is not None and y0_out is not None)
             else y_read + 1.2)
    ax.set_ylim(y0_in - 1.0, y_top)
    ax.set_yticks([])
    ax.set_xlabel("Timestep (ms)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=9)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=(1,0,0, min(1.0, alpha*5)), lw=1.5,
               label="Positive weight"),
        Line2D([0], [0], color=(0,0,1, min(1.0, alpha*5)), lw=1.5,
               label="Negative weight"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="upper right", framealpha=0.85)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def draw_mechanism_on_ax(ax, traces, weights_dict, delays_dict, title=True):
    """Draw the delay-routing mechanism into a single axes (used both as the
    diagnostic-panel bottom-right and as the standalone mechanism_sample0.png).

    Only actual SPIKES drive the arcs:
      - spiking-output run (has 'ho' + output_spikes): each HIDDEN spike -> output
        arrival via d_ho.
      - MLP-readout run (only d_ih): each HIDDEN spike is drawn, with causal
        input->hidden arcs (input arrival within a membrane window before the spike).
    ONLY when the hidden layer produced ZERO spikes do we fall back to drawing the
    full input->hidden d_ih ARRIVAL routing (top-N |w| per input spike), so the
    figure is never empty (e.g. burst @ low h fires ~0.2 spikes/trial).
    """
    from matplotlib.patches import FancyArrowPatch

    inp = traces["input_spikes"]
    T = int(traces.get("T", inp.shape[0]))
    h1 = traces["hidden1_spikes"]
    h2 = traces.get("hidden2_spikes")
    last_h = h2 if h2 is not None else h1
    win = int(traces.get("win_len", T))
    read = int(traces.get("read_len", 0))
    sub = traces.get("sub_win")
    K = int(traces.get("K", 1))
    out_spikes = traces.get("output_spikes")
    out_logits = traces.get("output_logits")
    observation_mode = traces.get("observation_mode", "late_window")
    output_window_len = traces.get("output_window_len")
    query_ops = list(traces.get("query_ops", []))
    has_ho = ("ho" in weights_dict) and ("ho" in delays_dict) and (out_spikes is not None)
    # arcs map to the layer the shown delays connect to: d_ho -> last hidden,
    # d_ih -> first hidden. (For 1-layer nets these coincide.)
    hid_arc = last_h if has_ho else h1
    n_in, n_h = inp.shape[1], hid_arc.shape[1]

    yin, yhid, yout = (0.0, 1.0), (2.0, 8.0), 9.3
    def yq(j):
        return yhid[0] + (yhid[1] - yhid[0]) * (j / max(1, n_h - 1))

    ax.axvspan(0, win, color="#e8eef7", alpha=0.7, zorder=0)
    ax.axvspan(win, T, color="#e7f4ea", alpha=0.9, zorder=0)
    ax.axvline(win, color="#2e7d32", ls="--", lw=1.2, zorder=1)
    for target_time in _opponent_target_times(traces):
        ax.axvline(target_time, color="#8e44ad", ls="-.", lw=1.1, zorder=1)
        ax.text(target_time, yout + .28, f"target t={target_time}", rotation=90,
                ha="right", va="bottom", fontsize=5.5, color="#8e44ad")
    ax.text(win / 2, yout + 0.5, "input", ha="center", fontsize=7, color="#4666a0")
    if observation_mode == "late_window":
        observed_label = "observed: final window"
        observed = lambda t: win <= t < T
    elif observation_mode == "all_time":
        observed_label = "observed: all time"
        observed = lambda t: 0 <= t < T
    elif observation_mode == "windowed_shared":
        observed_label = "observed: shared output windows"
        observed = lambda t: win <= t < T
    else:
        observed_label = "observed: time bins"
        observed = lambda t: 0 <= t < T
    ax.text((win + T) / 2, yout + 0.5, observed_label, ha="center", fontsize=7,
            color="#2e7d32", fontweight="bold")
    if K > 1 and sub:
        for k in range(1, K):
            ax.axvline(k * sub, color="#4666a0", ls=":", lw=0.8, alpha=0.6, zorder=1)
    if output_window_len and K > 1:
        for k in range(K + 1):
            ax.axvline(win + k * int(output_window_len), color="#2e7d32",
                       ls=":" if k not in (0, K) else "--", lw=.9, alpha=.75, zorder=1)
        for k in range(K):
            label = str(query_ops[k]) if k < len(query_ops) else f"Q{k}"
            ax.text(win + (k + .5) * int(output_window_len), yout + .05,
                    f"W{k}:{label}", ha="center", va="bottom", fontsize=6, color="#2e7d32")

    ti, ci = np.where(inp > 0)
    for t, c in zip(ti, ci):
        y = yin[0] + (yin[1] - yin[0]) * (c / max(1, n_in - 1))
        ax.plot([t], [y], marker="|", ms=10, mew=1.6, color="black", zorder=5)

    def _arc(x0, y0, x1, y1, color, alpha, lw, style="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                     connectionstyle="arc3,rad=-0.16", arrowstyle=style,
                     mutation_scale=8, lw=lw, color=color, alpha=alpha, zorder=3))

    th, jh = np.where(hid_arc > 0)
    landed = 0

    if len(th) > 0 and has_ho:                       # spike-based: hidden -> output
        d_ho = delays_dict["ho"]; w_ho = weights_dict["ho"]
        n_out = w_ho.shape[1]
        wmax = float(np.abs(w_ho).max()) + 1e-9
        def yo(o):
            return yout + (o - (n_out - 1) / 2) * min(.22, .8 / max(n_out, 1))
        for t, j in zip(th, jh):
            ax.plot([t], [yq(j)], "o", ms=4.5, color="#1f4e9c", zorder=6)
            for o in np.argsort(-np.abs(w_ho[j]))[:min(2, n_out)]:
                arr = t + d_ho[j, o]; col = "#c0392b" if w_ho[j, o] >= 0 else "#2c5fa8"
                _arc(t, yq(j), arr, yo(o), col, 0.3 + 0.6 * abs(w_ho[j, o]) / wmax,
                     0.8 + 1.5 * abs(w_ho[j, o]) / wmax, "-|>")
                inw = observed(arr); landed += int(inw)
                ax.plot([arr], [yo(o)], marker="v", ms=5,
                        mfc=("#c0392b" if inw else "none"), mec=col, mew=1.0, zorder=6)
        to, oo = np.where(out_spikes > 0)
        for t, o in zip(to, oo):
            ax.plot([t], [yo(o)], marker="*", ms=11, color="#f1c40f", mec="#b8860b", zorder=7)
        if n_out == 2:
            ax.text(T + .15, yo(0), "class 0", va="center", fontsize=5.5)
            ax.text(T + .15, yo(1), "class 1", va="center", fontsize=5.5)
        mode = "hidden→output $d_{ho}$ (spike-based)"

    elif len(th) > 0:                                # spike-based: input -> hidden spike
        w_ih, d_ih = weights_dict["ih"], delays_dict["ih"]
        wmax = float(np.abs(w_ih).max()) + 1e-9
        causal = max(int(sub or 10), 10) + 2
        for t, j in zip(th, jh):
            inw = observed(t); landed += int(inw)
            ax.plot([t], [yq(j)], "o", ms=5,
                    color=("#2e7d32" if inw else "#1f4e9c"), zorder=6)
            cand = []
            for tin, ch in zip(ti, ci):
                arr = tin + d_ih[ch, j]
                if 0 <= (t - arr) <= causal:
                    cand.append((abs(w_ih[ch, j]), tin, ch, w_ih[ch, j]))
            for aw, tin, ch, wv in sorted(cand, reverse=True)[:4]:
                yc = yin[0] + (yin[1] - yin[0]) * (ch / max(1, n_in - 1))
                col = "#c0392b" if wv >= 0 else "#2c5fa8"
                _arc(tin, yc, t, yq(j), col, 0.2 + 0.55 * aw / wmax, 0.6 + 1.2 * aw / wmax)
        mode = "input→hidden $d_{ih}$ (spike-based)"

    else:                                            # no spikes: arrival routing
        w_ih, d_ih = weights_dict["ih"], delays_dict["ih"]
        wmax = float(np.abs(w_ih).max()) + 1e-9
        top_n = max(1, min(n_h, 140 // max(1, len(ti))))
        for t, c in zip(ti, ci):
            yc = yin[0] + (yin[1] - yin[0]) * (c / max(1, n_in - 1))
            for j in np.argsort(-np.abs(w_ih[c]))[:top_n]:
                arr = t + d_ih[c, j]; col = "#c0392b" if w_ih[c, j] >= 0 else "#2c5fa8"
                _arc(t, yc, arr, yq(j), col, 0.15 + 0.5 * abs(w_ih[c, j]) / wmax,
                     0.6 + 1.0 * abs(w_ih[c, j]) / wmax)
                landed += int(observed(arr))
        mode = "input→hidden $d_{ih}$ (no spikes → arrival routing)"

    ax.text(-2.0, np.mean(yin), "In", ha="right", va="center", fontsize=7)
    ax.text(-2.0, np.mean(yhid), "Hid", ha="right", va="center", fontsize=7, color="#1f4e9c")
    if has_ho:
        ax.text(-2.0, yout, "Out", ha="right", va="center", fontsize=7, color="#b8860b")
    ax.set_xlim(-3, T + 1.5)
    ax.set_ylim(-1.0, yout + 1.0)
    ax.set_yticks([])
    ax.set_xlabel("time (ms)", fontsize=7)
    ax.tick_params(labelsize=6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(f"Delay routing mechanism\n{mode}  |  {landed} observed ({observation_mode})",
                     fontsize=8)
    return landed


def plot_diagnostic_panel(
    traces: dict,
    weights_dict: dict,
    delays_dict: dict,
    cfg: dict,
    log_rows: list,
    eval_results: dict,
    save_path: str,
):
    """Combined 4-row diagnostic figure (20×15 inches).

    Row 0: config text | training loss | training accuracy
    Row 1: weight heatmaps (up to 3 panels)
    Row 2: delay heatmaps (up to 2 panels)
    Row 3: spike raster (left) | layer-to-layer flow (right)
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 20), constrained_layout=False)
    gs  = GridSpec(5, 6, figure=fig, hspace=0.48, wspace=0.35)

    # ── Row 0: config | loss | accuracy ──────────────────────────────────
    ax_cfg  = fig.add_subplot(gs[0, :2])
    ax_loss = fig.add_subplot(gs[0, 2:4])
    ax_acc  = fig.add_subplot(gs[0, 4:6])

    # Config text
    ax_cfg.axis("off")
    hidden_sizes_str = str(cfg.get("hidden_sizes") or [cfg.get("n_hidden", "?")])
    lines = [
        f"n_input={cfg.get('n_input', 2)}  K={cfg.get('K', eval_results.get('K', '?'))}",
        f"hidden={hidden_sizes_str}",
        f"topology={cfg.get('topology_type','shared_dense')}",
        (f"surface_h={cfg.get('surface_hidden_width')} "
         f"({cfg.get('hidden_width_semantics')})"
         if cfg.get("surface_hidden_width") is not None else "surface_h=n/a"),
        f"input_code={cfg.get('encoding_mode','rate')}",
        f"train_mode={cfg.get('train_mode','?')}",
        f"readout={cfg.get('readout_type','linear')}",
        f"endpoint={cfg.get('readout_endpoint','count_decoder')}",
        f"observe={cfg.get('observation_mode','late_window')}",
        f"delay_param={cfg.get('delay_param_type','sigmoid')}",
        (f"fixed_d={cfg['fixed_delay_value']}"
         if cfg.get("fixed_delay_value") is not None else "fixed_d=trainable"),
        "",
        f"T={traces.get('T','?')}  win={cfg.get('win_len','?')}  "
        f"read={cfg.get('read_len','?')}  sw={cfg.get('sub_win','?')}",
        f"lr_w={cfg.get('lr_w','?')}  lr_d={cfg.get('lr_d','?')}  "
        f"lr_ro={cfg.get('lr_readout','?')}",
        f"epochs={cfg.get('epochs','?')}  batch={cfg.get('batch_size','?')}  "
        f"seed={cfg.get('seed','?')}",
        (f"target={cfg.get('opponent_target_timing_mode')} "
         f"t={','.join(map(str, _opponent_target_times({**traces, **cfg})))}"
         if cfg.get("opponent_target_timing_mode") else "target=none"),
        (f"intervention={eval_results.get('post_training_intervention', {}).get('name')}"
         if eval_results.get("post_training_intervention") else "intervention=none"),
        "",
        "-" * 30,
        f"acc    = {eval_results.get('accuracy', float('nan')):.1%}",
        f"worst  = {eval_results.get('worst_query_accuracy', float('nan')):.1%}",
        (f"worstB = {eval_results['worst_query_balanced_accuracy']:.1%}"
         if eval_results.get('worst_query_balanced_accuracy') is not None else "worstB = n/a"),
        f"exact  = {eval_results.get('exact_trial_accuracy', float('nan')):.1%}",
        (f"K/spk  = {eval_results['throughput_K_per_spk']:.3f}"
         if eval_results.get('throughput_K_per_spk') is not None else "K/spk  = n/a"),
        f"spk/tr = {eval_results.get('mean_hidden_spikes', float('nan')):.1f}",
        (f"time-hit={eval_results['target_timing_hit_rate']:.1%}"
         if eval_results.get('target_timing_hit_rate') is not None else "time-hit=n/a"),
    ]
    ax_cfg.text(0.05, 0.95, "\n".join(lines),
                transform=ax_cfg.transAxes, fontsize=7.3, linespacing=.95,
                va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                          edgecolor="#BDBDBD", linewidth=0.8))
    ax_cfg.set_title("Run Config + Results", fontsize=9)

    # Training curves
    if log_rows:
        epochs = [r["epoch"] for r in log_rows]
        ax_loss.plot(epochs, [r["train_loss"] for r in log_rows], label="train")
        ax_loss.plot(epochs, [r["val_loss"]   for r in log_rows], label="val")
        if "train_spike_count_loss" in log_rows[0]:
            ax_loss.plot(epochs, [r["train_spike_count_loss"] for r in log_rows],
                         label="train spike", linestyle="--", alpha=.8)
        if "train_target_spike_train_loss" in log_rows[0]:
            ax_loss.plot(epochs, [r["train_target_spike_train_loss"] for r in log_rows],
                         label="train timing", linestyle="-.", alpha=.85)
        if "train_output_membrane_loss" in log_rows[0]:
            ax_loss.plot(epochs, [r["train_output_membrane_loss"] for r in log_rows],
                         label="train membrane", linestyle=":", alpha=.9)
        ax_loss.set_xlabel("Epoch", fontsize=8)
        ax_loss.set_ylabel("Loss", fontsize=8)
        ax_loss.legend(fontsize=7)
        ax_loss.grid(True, alpha=0.25)
        ax_loss.tick_params(labelsize=7)

        ax_acc.plot(epochs, [r["train_acc"] for r in log_rows], label="train")
        ax_acc.plot(epochs, [r["val_acc"]   for r in log_rows], label="val")
        ax_acc.set_xlabel("Epoch", fontsize=8)
        ax_acc.set_ylabel("Accuracy", fontsize=8)
        ax_acc.set_ylim(0, 1.05)
        ax_acc.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax_acc.legend(fontsize=7)
        ax_acc.grid(True, alpha=0.25)
        ax_acc.tick_params(labelsize=7)
    else:
        for ax in (ax_loss, ax_acc):
            ax.text(0.5, 0.5, "No log data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
    ax_loss.set_title("Training Loss", fontsize=9)
    ax_acc.set_title("Training Accuracy", fontsize=9)

    # ── Row 1: weight heatmaps ────────────────────────────────────────────
    w_keys = [
        key for key in (
            "ih", "h1h2", "ho", "readout", "readout_in", "readout_out"
        ) if key in weights_dict
    ]
    n_wkeys = len(w_keys)
    w_titles = {"ih": "W: Input→Hidden",
                "h1h2": "W: H1→H2",
                "ho": "W: Hidden→Output",
                "readout": "W: H→Readout",
                "readout_in": "W: Hidden→MLP hidden",
                "readout_out": "W: MLP hidden→logit"}
    w_axes = [fig.add_subplot(gs[1, i*2:(i+1)*2]) for i in range(min(n_wkeys, 3))]
    for ax, key in zip(w_axes, w_keys):
        W = weights_dict[key]
        vabs = max(float(np.abs(W).max()), 1e-6)
        im = ax.imshow(W, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                       aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.65)
        ax.set_title(w_titles.get(key, key), fontsize=9)
        ax.set_xlabel("Post neuron", fontsize=7)
        ax.set_ylabel("Pre neuron", fontsize=7)
        ax.tick_params(labelsize=6)

    # ── Row 2: delay heatmaps ─────────────────────────────────────────────
    d_keys  = [k for k in ("ih", "h1h2", "ho") if k in delays_dict]
    n_dkeys = len(d_keys)
    d_titles = {"ih": "D: Input→Hidden",
                "h1h2": "D: H1→H2",
                "ho": "D: Hidden→Output"}
    d_axes = [fig.add_subplot(gs[2, i*3:(i+1)*3]) for i in range(min(n_dkeys, 2))]
    for ax, key in zip(d_axes, d_keys):
        D = delays_dict[key]
        im = ax.imshow(D, cmap="viridis", aspect="auto", vmin=0,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.65, label="steps")
        ax.set_title(d_titles.get(key, key), fontsize=9)
        ax.set_xlabel("Post neuron", fontsize=7)
        ax.set_ylabel("Pre neuron", fontsize=7)
        ax.tick_params(labelsize=6)

    # ── Row 3 left: spike raster (Input / Hidden / Output) ───────────────
    output_membrane = traces.get("output_membrane")
    membrane_grid = gs[3, :3].subgridspec(
        2 if output_membrane is not None else 1, 1, hspace=.42
    )
    ax_mem = fig.add_subplot(membrane_grid[0, 0])
    membrane = traces.get("hidden_membrane")
    if membrane is not None:
        im = ax_mem.imshow(membrane.T, aspect="auto", origin="lower", cmap="magma")
        plt.colorbar(im, ax=ax_mem, shrink=0.7, label="membrane potential")
        ax_mem.set_ylabel("Hidden neuron")
        if output_membrane is None:
            ax_mem.set_xlabel("Timestep")
        ax_mem.set_title("Hidden membrane dynamics (fixed diagnostic sample)")
        ax_mem.axvline(traces.get("win_len", membrane.shape[0]),
                       color="cyan", linestyle="--", linewidth=1)
        hidden_per_query = traces.get("hidden_per_query")
        if (hidden_per_query
                and traces.get("topology_type") == "spatial_independent_shared_decoder"):
            for boundary in range(
                int(hidden_per_query), membrane.shape[1], int(hidden_per_query)
            ):
                ax_mem.axhline(
                    boundary - .5, color="white", linestyle=":",
                    linewidth=.9, alpha=.85,
                )
        for target_time in _opponent_target_times(traces):
            ax_mem.axvline(target_time, color="#8e44ad", linestyle="-.", linewidth=1)
    else:
        ax_mem.text(0.5, 0.5, "Membrane trace unavailable", ha="center", va="center")
        ax_mem.set_axis_off()

    if output_membrane is not None:
        ax_omem = fig.add_subplot(membrane_grid[1, 0])
        for output_index in range(output_membrane.shape[1]):
            ax_omem.plot(output_membrane[:, output_index],
                         label=f"output {output_index}", linewidth=1.1)
        threshold = cfg.get("lif_output_threshold")
        if threshold is not None:
            ax_omem.axhline(float(threshold), color="black", linestyle="--",
                            linewidth=1, label=f"threshold={float(threshold):g}")
        win_start = int(traces.get("win_len", 0))
        window_len = traces.get("output_window_len")
        if window_len:
            for k in range(int(traces.get("K", 1)) + 1):
                ax_omem.axvline(win_start + k * int(window_len), color="#777",
                                linestyle=":", linewidth=.7)
        for target_time in _opponent_target_times(traces):
            ax_omem.axvline(target_time, color="#8e44ad", linestyle="-.", linewidth=1.2,
                            label=f"target t={target_time}")
        ax_omem.set_xlim(0, output_membrane.shape[0] - 1)
        ax_omem.set_xlabel("Timestep")
        ax_omem.set_ylabel("V")
        ax_omem.set_title("Pre-reset output membrane vs firing threshold")
        ax_omem.legend(fontsize=6, ncol=3, loc="upper left")
        ax_omem.grid(True, alpha=.2)

    ax_dec = fig.add_subplot(gs[3, 3:])
    logits_q = np.asarray(traces.get("output_logits", []), dtype=float)
    labels_q = np.asarray(traces.get("query_labels", []), dtype=int).reshape(-1)
    preds_q = np.asarray(traces.get("output_predictions", logits_q > 0), dtype=int).reshape(-1)
    q = np.arange(len(logits_q))
    colors_q = ["#55A868" if i < len(labels_q) and preds_q[i] == labels_q[i]
                else "#C44E52" for i in q]
    ax_dec.bar(q, logits_q, color=colors_q)
    ax_dec.axhline(0, color="black", linewidth=1)
    ax_dec.set_xticks(q)
    query_ops = list(cfg.get("query_ops", []))
    ax_dec.set_xticklabels([
        f"Q{i}{' '+query_ops[i] if i < len(query_ops) else ''}\n"
        f"y={labels_q[i] if i < len(labels_q) else '?'} p={preds_q[i]}"
        for i in q
    ])
    ax_dec.set_ylabel("Decoder logit")
    temporal_windows = cfg.get("observation_mode") == "windowed_shared" or cfg.get("opponent_output_mode") == "shared_windowed"
    ax_dec.set_title("Output-window decisions" if temporal_windows else "Operation-wise decoder decisions")
    counts_q = traces.get("output_window_counts", traces.get("output_pair_counts"))
    if counts_q is not None:
        counts_q = np.asarray(counts_q)
        for i in range(min(len(q), len(counts_q))):
            ax_dec.text(i, logits_q[i], f"  n-={counts_q[i,0]:.0f}, n+={counts_q[i,1]:.0f}",
                        rotation=90, va="bottom" if logits_q[i] >= 0 else "top", fontsize=6)

    s_in  = traces["input_spikes"]
    s_h1  = traces["hidden1_spikes"]
    s_h2  = traces.get("hidden2_spikes")
    T_val = traces.get("T", s_in.shape[0])
    wl    = traces.get("win_len", T_val)
    rl    = traces.get("read_len", 0)
    sw    = traces.get("sub_win")
    K_val = traces.get("K", 1)
    output_window_len = traces.get("output_window_len")
    out_logits   = traces.get("output_logits")   # [K] or None (old npz files)
    out_spikes   = traces.get("output_spikes")   # [T, n_out] real LIF output, or None
    has_real_out = out_spikes is not None         # true spiking output layer
    has_logit_out = (not has_real_out) and (out_logits is not None)

    layer_data = [("Input",  s_in, "#444444"),
                  ("Hidden", s_h1, "steelblue")]
    if s_h2 is not None:
        layer_data.append(("Hidden2", s_h2, "darkorange"))
    if has_real_out:
        layer_data.append(("Output", out_spikes, "#2ca02c"))

    # Decision-marker strip shown only when no real output spikes
    has_out   = has_logit_out
    n_lyr     = len(layer_data) + (1 if has_out else 0)
    heights_r = [max(ld[1].shape[1], 2) for ld in layer_data]
    if has_out:
        heights_r.append(max(K_val, 2))

    gs_r3 = gs[4, :3].subgridspec(n_lyr, 1,
                                   height_ratios=heights_r, hspace=0.06)
    rax = [fig.add_subplot(gs_r3[i, 0]) for i in range(n_lyr)]
    COLORS_t10 = list(plt.cm.tab10.colors)

    for ax, (lname, ldata, col) in zip(rax[:len(layer_data)], layer_data):
        n_nrn = ldata.shape[1]
        for n in range(n_nrn):
            ts = np.where(ldata[:, n] > 0)[0]
            if not len(ts):
                continue
            c = (COLORS_t10[(n//2) % len(COLORS_t10)]
                 if (lname == "Input" and ldata.shape[1] > 2) else col)
            ax.scatter(ts, np.full(len(ts), float(n)),
                       s=4, color=c, marker="|", linewidths=0.7)
        ax.set_ylim(-0.5, n_nrn - 0.5)
        ax.set_ylabel(lname, fontsize=7)
        ax.set_yticks([])
        ax.axvspan(0, wl, alpha=0.07, color="royalblue")
        ax.axvspan(wl, wl + rl, alpha=0.10, color="tomato")
        ax.set_xlim(-0.5, T_val - 0.5)
        ax.tick_params(labelsize=6)
        hidden_per_query = traces.get("hidden_per_query")
        if (lname == "Hidden" and hidden_per_query
                and traces.get("topology_type") == "spatial_independent_shared_decoder"):
            for boundary in range(int(hidden_per_query), n_nrn, int(hidden_per_query)):
                ax.axhline(boundary - .5, color="#555", ls=":", lw=.7, alpha=.75)
        if sw and K_val > 1:
            for k in range(K_val):
                ax.axvline(k * sw, color=COLORS_t10[k % len(COLORS_t10)],
                           ls="--", lw=0.7, alpha=0.7)
        if output_window_len and K_val > 1:
            for k in range(K_val + 1):
                boundary = wl + k * int(output_window_len)
                ax.axvline(boundary, color=COLORS_t10[k % len(COLORS_t10)],
                           ls=":" if k not in (0, K_val) else "--", lw=0.9, alpha=0.8)
            if lname == layer_data[0][0]:
                for k in range(K_val):
                    op_label = query_ops[k] if k < len(query_ops) else f"Q{k}"
                    ax.text(wl + (k + .5) * int(output_window_len), n_nrn - .4,
                            f"W{k}:{op_label}", ha="center", va="top", fontsize=5.5)
        for target_time in _opponent_target_times(traces):
            ax.axvline(target_time, color="#8e44ad", ls="-.", lw=1.0, alpha=.9)
        if lname == "Output" and cfg.get("opponent_output_mode"):
            if cfg.get("opponent_output_mode") == "shared_windowed" and n_nrn == 2:
                ax.set_yticks([0, 1]); ax.set_yticklabels(["class 0", "class 1"], fontsize=5.5)
            elif cfg.get("opponent_output_mode") == "parallel_pairs":
                labels_out = []
                for k in range(K_val):
                    op_label = query_ops[k] if k < len(query_ops) else f"Q{k}"
                    labels_out.extend([f"{op_label}:0", f"{op_label}:1"])
                ax.set_yticks(range(min(n_nrn, len(labels_out))))
                ax.set_yticklabels(labels_out[:n_nrn], fontsize=5)

    # Output strip: one row per query, decision marker at readout window centre.
    # len(out_logits) can be < K_val for aggregate-output topologies (one
    # readout head, K_val sub-windows) -- draw only the rows that exist.
    if has_out:
        ax_out = rax[len(layer_data)]
        for k in range(min(K_val, len(out_logits))):
            t_rc = (wl + (k + .5) * int(output_window_len)
                    if output_window_len and temporal_windows
                    else wl + max(rl // 2, 0))
            lv  = float(out_logits[k])
            col_k = "#2ca02c" if lv > 0 else "#d62728"
            mrk_k = "|"       if lv > 0 else "x"
            ax_out.scatter([t_rc], [float(k)], s=32, color=col_k,
                           marker=mrk_k, linewidths=1.5, zorder=4)
            ax_out.text(-0.5, float(k), f"Q{k}", ha="right", va="center",
                        fontsize=5.5, color="#555")
        ax_out.set_ylim(-0.5, K_val - 0.5)
        ax_out.set_ylabel("Decision", fontsize=7)
        ax_out.set_yticks([])
        ax_out.axvspan(0, wl, alpha=0.07, color="royalblue")
        ax_out.axvspan(wl, wl + rl, alpha=0.18, color="tomato")
        ax_out.set_xlim(-0.5, T_val - 0.5)
        ax_out.tick_params(labelsize=6)
        if output_window_len and temporal_windows:
            for k in range(K_val + 1):
                ax_out.axvline(wl + k * int(output_window_len), color="#555", ls=":", lw=.8)

    raster_title = ("Spike Raster: Input / Hidden / true Output spikes"
                    if has_real_out else
                    "Spike Raster + decoder decision markers (not output spikes)")
    rax[0].set_title(raster_title + "  (fixed sample)", fontsize=9)
    rax[-1].set_xlabel("Timestep (ms)", fontsize=7)
    for a in rax[:-1]:
        a.tick_params(labelbottom=False)

    # ── Row 3 right: delay-routing mechanism (replaces the dense flow graph) ──
    ax_flow = fig.add_subplot(gs[4, 3:])
    draw_mechanism_on_ax(ax_flow, traces, weights_dict, delays_dict, title=True)

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_diagnostic_data(plot_dir: str, traces: dict, weights_dict: dict,
                           delays_dict: dict) -> None:
    """Persist the raw arrays behind the diagnostic plots to
    `plot_dir/diagnostic_data.npz`, so plots can be regenerated later
    (new style/logic) without rerunning the model.
    """
    flat = {}
    for k, v in traces.items():
        flat[f"traces__{k}"] = np.asarray(v)
    for k, v in weights_dict.items():
        flat[f"weights__{k}"] = np.asarray(v)
    for k, v in delays_dict.items():
        flat[f"delays__{k}"] = np.asarray(v)
    np.savez_compressed(os.path.join(plot_dir, "diagnostic_data.npz"), **flat)


def load_diagnostic_data(plot_dir: str) -> tuple:
    """Inverse of `_save_diagnostic_data`. Returns (traces, weights_dict, delays_dict)."""
    npz = np.load(os.path.join(plot_dir, "diagnostic_data.npz"), allow_pickle=True)
    traces, weights_dict, delays_dict = {}, {}, {}
    for key in npz.files:
        group, _, name = key.partition("__")
        arr = npz[key]
        if arr.ndim == 0:
            arr = arr.item()
        target = {"traces": traces, "weights": weights_dict, "delays": delays_dict}[group]
        target[name] = arr
    return traces, weights_dict, delays_dict


def _find_richest_traces(model, cfg: dict, K: int, op: str, device: str,
                          n_seeds: int = 15) -> Optional[dict]:
    """Return the traces dict (from _extract_run_traces) with the most hidden spikes.

    Tries up to n_seeds different random seeds and picks the sample with the
    highest total hidden spike count — giving enhanced plots more visual content.
    Returns None if every seed fails.
    """
    best_tr, best_n = None, -1
    for s in range(n_seeds):
        try:
            tr, _, _ = _extract_run_traces(model, cfg, K, op, device, seed=50 * (s + 1))
            n = int(tr["hidden1_spikes"].sum())
            if n > best_n:
                best_tr, best_n = tr, n
        except Exception:
            continue
    return best_tr


def _make_diagnostic_plots(plot_dir: str, traces: dict, weights_dict: dict,
                            delays_dict: dict, title_base: str, cfg: dict,
                            K: int, log_rows: list, eval_results: dict) -> None:
    """Render all diagnostic PNGs from already-extracted data."""

    def _safe(fn, path, **kw):
        try:
            fn(**kw, save_path=path)
        except Exception as exc:
            import logging
            logging.getLogger("viz").warning(f"Plot failed {path}: {exc}")

    _safe(plot_weight_heatmaps, os.path.join(plot_dir, "weight_heatmaps.png"),
          weights_dict=weights_dict, title=title_base)

    _safe(plot_delay_heatmaps, os.path.join(plot_dir, "delay_heatmaps.png"),
          delays_dict=delays_dict, title=title_base)

    _safe(plot_spike_raster_layers, os.path.join(plot_dir, "spike_raster_sample0.png"),
          traces=traces, title=title_base,
          win_len=cfg.get("win_len"), read_len=cfg.get("read_len"),
          sub_win=cfg.get("sub_win"), K=K)

    _safe(plot_layer_to_layer_spike_flow,
          os.path.join(plot_dir, "layer_to_layer_spike_flow_sample0.png"),
          traces=traces, weights_dict=weights_dict, delays_dict=delays_dict,
          title=title_base)

    _safe(plot_diagnostic_panel,
          os.path.join(plot_dir, "diagnostic_panel.png"),
          traces=traces, weights_dict=weights_dict, delays_dict=delays_dict,
          cfg={**cfg, "K": K}, log_rows=log_rows, eval_results=eval_results)

    # Enhanced plots — same recorded traces as the original raster / flow plots
    _safe(plot_enhanced_raster_layers,
          os.path.join(plot_dir, "enhanced_raster.png"),
          traces=traces, title=title_base)
    _safe(plot_enhanced_spike_flow,
          os.path.join(plot_dir, "enhanced_flow.png"),
          traces=traces, weights_dict=weights_dict, delays_dict=delays_dict,
          title=title_base)


def replot_run_diagnostics(run_dir: str, cfg: dict, log_rows: list,
                            eval_results: dict, K: int, seed: int = 999) -> None:
    """Regenerate diagnostic PNGs from `plot_dir/diagnostic_data.npz`
    (saved by `save_run_diagnostic_plots`) without rerunning the model.
    Useful after changing plot styling/logic.
    """
    plot_dir = os.path.join(run_dir, "plots")
    traces, weights_dict, delays_dict = load_diagnostic_data(plot_dir)

    mname = eval_results.get("model_name", cfg.get("model_name", "model"))
    acc   = eval_results.get("accuracy", float("nan"))
    seed_ = cfg.get("seed", seed)
    title_base = (f"Plan D | {mname} | K={K} | seed={seed_} | "
                  f"acc={acc:.1%}")

    _make_diagnostic_plots(plot_dir, traces, weights_dict, delays_dict,
                            title_base, cfg, K, log_rows, eval_results)


def save_run_diagnostic_plots(
    model,
    cfg: dict,
    log_rows: list,
    eval_results: dict,
    run_dir: str,
    K: int,
    op: str,
    device: str,
    seed: int = 999,
    dataset_override: tuple | None = None,
):
    """Orchestrate all per-run diagnostic plots.

    Saves to `run_dir/plots/`:
      diagnostic_panel.png
      weight_heatmaps.png
      delay_heatmaps.png
      spike_raster_sample0.png
      layer_to_layer_spike_flow_sample0.png
      diagnostic_data.npz   (raw traces/weights/delays for replotting later)

    dataset_override : forwarded to _extract_run_traces() -- see its
        docstring. Pass K as the true sub-window count (K_query) for
        aggregate-output topologies, not model.n_queries.
    """
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    mname = eval_results.get("model_name", cfg.get("model_name", "model"))
    acc   = eval_results.get("accuracy", float("nan"))
    seed_ = cfg.get("seed", seed)
    title_base = (f"Plan D | {mname} | K={K} | seed={seed_} | "
                  f"acc={acc:.1%}")

    try:
        traces, weights_dict, delays_dict = _extract_run_traces(
            model, cfg, K, op, device, seed, dataset_override=dataset_override)
    except Exception as exc:
        import logging
        logging.getLogger("viz").warning(
            f"save_run_diagnostic_plots: trace extraction failed: {exc}")
        return

    _save_diagnostic_data(plot_dir, traces, weights_dict, delays_dict)
    _make_diagnostic_plots(plot_dir, traces, weights_dict, delays_dict,
                            title_base, cfg, K, log_rows, eval_results)


# ── Enhanced visualisations ──────────────────────────────────────────────────

def plot_enhanced_raster_layers(traces: dict, save_path: str,
                                 title: str = "") -> None:
    """Enhanced spike raster.

    Improvements over plot_spike_raster_layers:
      - Hidden neurons sorted by first-spike time (reveals temporal wave structure)
      - Spikes coloured by sub-window (Q0/Q1/Q2/readout)
      - Right-side sidebar shows readout-window spike count per neuron
    """
    s_in  = traces["input_spikes"]
    s_h1  = traces["hidden1_spikes"]
    T, n_in = s_in.shape
    _, h1   = s_h1.shape

    win_len  = traces.get("win_len", T)
    read_len = traces.get("read_len", 0)
    sub_win  = traces.get("sub_win") or win_len   # None-safe fallback
    K        = traces.get("K", 1)
    COLORS   = list(plt.cm.tab10.colors)

    def _window_color(t):
        for k in range(K):
            if k * sub_win <= t < (k + 1) * sub_win:
                return COLORS[k % len(COLORS)]
        return "darkred"

    fig, ax = plt.subplots(figsize=(12, max(4, 0.14 * (h1 + n_in) + 2)))

    # Background shading
    ax.axvspan(0,       win_len,            alpha=0.07, color="royalblue", zorder=0)
    ax.axvspan(win_len, win_len + read_len, alpha=0.10, color="tomato",    zorder=0)
    for k in range(K):
        ax.axvline(k * sub_win, color=COLORS[k % len(COLORS)],
                   ls="--", lw=0.9, alpha=0.65, zorder=1)
        ax.text(k * sub_win + 0.3, h1 + n_in - 0.3,
                f"Q{k}", fontsize=6.5, color=COLORS[k % len(COLORS)],
                va="top", fontweight="bold")

    # Sort hidden neurons by first-spike time
    first = np.full(h1, T + 1, dtype=float)
    for j in range(h1):
        ts = np.where(s_h1[:, j] > 0)[0]
        if len(ts):
            first[j] = ts[0]
    order = np.argsort(first)

    # Input spikes (original row order)
    for i in range(n_in):
        ts = np.where(s_in[:, i] > 0)[0]
        if len(ts):
            c = COLORS[(i // 2) % len(COLORS)] if n_in > 2 else "#333333"
            ax.scatter(ts, np.full(len(ts), h1 + i + 0.5),
                       s=7, color=c, marker="|", linewidths=1.0, zorder=3)

    # Hidden spikes (sorted, window-coloured)
    for rank, j in enumerate(order):
        ts = np.where(s_h1[:, j] > 0)[0]
        if len(ts) == 0:
            continue
        ax.scatter(ts, np.full(len(ts), rank),
                   s=7, c=[_window_color(t) for t in ts],
                   marker="|", linewidths=1.0, zorder=3)

    # Sidebar: readout-window spike count per neuron
    ro = s_h1[win_len:].sum(axis=0)
    max_ro = max(ro.max(), 1)
    for rank, j in enumerate(order):
        if ro[j] > 0:
            ax.barh(rank, ro[j] / max_ro * 3,
                    left=T + 0.3, height=0.8, color="darkred", alpha=0.55)

    ax.axhline(h1 + 0.1, color="gray", lw=0.7)
    ax.set_xlim(-0.5, T + 4.5)
    ax.set_ylim(-0.5, h1 + n_in + 0.5)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("Neuron (sorted by 1st spike)", fontsize=8)
    ax.set_yticks([0, h1 // 4, h1 // 2, 3 * h1 // 4, h1])
    ax.set_yticklabels([f"n=0", f"n={h1//4}", f"n={h1//2}",
                        f"n={3*h1//4}", "↑In"], fontsize=7)

    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=COLORS[k % len(COLORS)], label=f"Q{k} window")
                      for k in range(K)]
    legend_patches.append(mpatches.Patch(color="darkred", label="Readout window"))
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right", framealpha=0.85)

    total_spk = int(s_h1.sum())
    active    = int((s_h1.sum(axis=0) > 0).sum())
    ro_spk    = int(s_h1[win_len:].sum())
    ax.text(0.01, 0.01,
            f"total_spk={total_spk}  active={active}/{h1}  ro_spk={ro_spk}",
            transform=ax.transAxes, fontsize=6.5, va="bottom", color="gray")
    if title:
        ax.set_title(title, fontsize=8)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_enhanced_spike_flow(traces: dict, weights_dict: dict, delays_dict: dict,
                              save_path: str, title: str = "",
                              w_threshold: float = 0.08,
                              ro_threshold: float = 0.05) -> None:
    """Enhanced layer-to-layer spike flow.

    Three connection types:
      (A) Blue/red fan lines  : Input spike → delayed arrival at Hidden (excit./inhib.)
      (B) Orange spans + stars: Arrivals within tau_m → actual Hidden fire (gold/green ★)
      (C) Crimson/blue verticals: Hidden fires in readout window → Readout bar
          Only drawn when linear readout weights are available (weights_dict["readout"]).
    """
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    s_in  = traces["input_spikes"]
    s_h1  = traces["hidden1_spikes"]
    T, n_in = s_in.shape
    _, h1   = s_h1.shape

    win_len  = traces.get("win_len", T)
    read_len = traces.get("read_len", 0)
    sub_win  = traces.get("sub_win")
    K        = traces.get("K", 1)
    tau_m    = 10.0
    COLORS   = list(plt.cm.tab10.colors)

    W_ih = weights_dict.get("ih")
    D_ih = delays_dict.get("ih")
    W_ro = weights_dict.get("readout")   # None for MLP readout

    if W_ih is None or D_ih is None:
        return   # can't draw anything without input→hidden weights

    GAP    = max(3, n_in // 4 + 1)
    y0_in  = 0
    y0_h1  = n_in + GAP
    y_read = y0_h1 + h1 + GAP

    def yi(n): return float(y0_in + n)
    def yh(n): return float(y0_h1 + n)

    fig_h = max(7, int(y_read * 0.30 + 3))
    fig, ax = plt.subplots(figsize=(13, fig_h))

    # Background shading
    ax.axvspan(0,       win_len,            alpha=0.06, color="royalblue", zorder=0)
    ax.axvspan(win_len, win_len + read_len, alpha=0.09, color="tomato",    zorder=0)
    ax.axhspan(y0_in - 0.6, y0_in + n_in - 0.4, alpha=0.07, color="royalblue", zorder=0)
    ax.axhspan(y0_h1 - 0.6, y0_h1 + h1  - 0.4, alpha=0.05, color="steelblue", zorder=0)
    if sub_win and K > 1:
        for k in range(K):
            ax.axvline(k * sub_win, color=COLORS[k % len(COLORS)],
                       ls=":", lw=0.9, alpha=0.5, zorder=1)

    # (A) Fan lines: input spike → delayed arrival at hidden
    segs, seg_cols = [], []
    for i in range(n_in):
        ts_pre = np.where(s_in[:, i] > 0)[0]
        for j in range(h1):
            w = float(W_ih[i, j])
            if abs(w) <= w_threshold:
                continue
            c = (0.9, 0.1, 0.1, 0.07) if w > 0 else (0.1, 0.1, 0.9, 0.07)
            d = float(D_ih[i, j])
            for t in ts_pre:
                arr = t + d
                if 0 <= arr < T:
                    segs.append([(float(t), yi(i)), (float(arr), yh(j))])
                    seg_cols.append(c)
    if segs:
        ax.add_collection(LineCollection(segs, colors=seg_cols, linewidths=0.5, zorder=2))

    # Input spike markers
    for i in range(n_in):
        ts = np.where(s_in[:, i] > 0)[0]
        if len(ts):
            ax.scatter(ts, np.full(len(ts), yi(i)),
                       s=16, color="#222222", marker="|", linewidths=1.0, zorder=4)

    # (C) Vote lines drawn FIRST (low zorder) so stars appear on top
    if W_ro is not None:
        ro_w   = W_ro[0] if W_ro.ndim == 2 else W_ro
        max_wr = np.max(np.abs(ro_w)) + 1e-8
        for j in range(h1):
            wrj = float(ro_w[j])
            if abs(wrj) <= ro_threshold:
                continue
            t_fires_ro = np.where(s_h1[win_len:, j] > 0)[0] + win_len
            if len(t_fires_ro) == 0:
                continue
            col = "crimson" if wrj > 0 else "mediumblue"
            mag = min(abs(wrj) / max_wr, 1.0)
            lw  = 3.0 + 4.0 * mag
            for t_fire in t_fires_ro:
                ax.plot([t_fire, t_fire], [0, y_read],
                        color=col, lw=lw, alpha=0.85, zorder=3)
                ax.scatter(t_fire, y_read, s=100, color=col, marker="^",
                           alpha=1.0, zorder=13, linewidths=0)

    # (B) Arrival→fire spans + stars
    for j in range(h1):
        t_fires = np.where(s_h1[:, j] > 0)[0]
        if len(t_fires) == 0:
            continue
        for t_fire in t_fires:
            arrivals = []
            for i in range(n_in):
                for t in np.where(s_in[:, i] > 0)[0]:
                    arr = t + float(D_ih[i, j])
                    if (t_fire - tau_m) <= arr <= t_fire and abs(float(W_ih[i, j])) > w_threshold:
                        arrivals.append(arr)
            if arrivals:
                ax.plot([min(arrivals), t_fire], [yh(j), yh(j)],
                        color="darkorange", lw=1.8, alpha=0.7,
                        solid_capstyle="round", zorder=5)
                for arr in arrivals:
                    ax.scatter(arr, yh(j), s=15, color="darkorange",
                               marker="o", alpha=0.55, zorder=6, linewidths=0)
            star_col  = "limegreen" if t_fire >= win_len else "gold"
            star_edge = "darkgreen"  if t_fire >= win_len else "darkorange"
            ax.scatter(t_fire, yh(j), s=90, color=star_col, marker="*",
                       edgecolors=star_edge, linewidths=0.5, zorder=12)

    # Readout bar + labels
    ax.axhline(y_read, xmin=win_len / T, xmax=1.0,
               color="tomato", lw=2.0, alpha=0.6, zorder=2)
    ax.text(win_len + (T - win_len) / 2, y_read + 0.3, "Readout",
            ha="center", va="bottom", fontsize=7, color="tomato")
    for lbl, yl in [("Input", y0_in + n_in / 2), ("Hidden", y0_h1 + h1 / 2)]:
        ax.text(-1.5, yl, lbl, ha="right", va="center",
                fontsize=8, fontweight="bold", color="steelblue")

    ax.set_xlim(-2, T + 0.5)
    ax.set_ylim(y0_in - 1.0, y_read + 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Time (ms)", fontsize=10)

    legend_elems = [
        Line2D([0],[0], color=(0.9,0.1,0.1,0.5), lw=1.5, label="(A) arrival — excitatory"),
        Line2D([0],[0], color=(0.1,0.1,0.9,0.5), lw=1.5, label="(A) arrival — inhibitory"),
        mpatches.Patch(color="darkorange", alpha=0.75,
                       label="(B) integration window → fire"),
        Line2D([0],[0], marker="*", color="gold",      markeredgecolor="darkorange",
               markersize=9, lw=0, label="(B) fire — input window (not counted)"),
        Line2D([0],[0], marker="*", color="limegreen", markeredgecolor="darkgreen",
               markersize=9, lw=0, label="(B) fire — readout window (counted ✓)"),
    ]
    if W_ro is not None:
        legend_elems += [
            Line2D([0],[0], color="crimson",    lw=3, label="(C) → readout  pos. weight"),
            Line2D([0],[0], color="mediumblue", lw=3, label="(C) → readout  neg. weight"),
        ]
    else:
        legend_elems.append(
            mpatches.Patch(color="gray", alpha=0.4, label="(C) MLP readout — votes not shown"))

    ax.legend(handles=legend_elems, fontsize=7, loc="upper left", framealpha=0.9)
    if title:
        ax.set_title(title, fontsize=8)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_neuron_connection_raster(
    spike_input: np.ndarray,
    hidden_spikes: np.ndarray,
    save_path: str,
    win_len: int,
    read_len: int,
    delays_ih: Optional[np.ndarray] = None,
    weights_ih: Optional[np.ndarray] = None,
    weights_readout: Optional[np.ndarray] = None,
    title: str = "",
    K: int = 1,
    sub_win: Optional[int] = None,
    n_connections: int = 12,
):
    """
    Mechanism raster: neuron index (y) × time (x) with synaptic delay lines.

    Style matches the reference figure style:
      - Large filled circles for every spike event
      - Straight lines from pre-spike (t_fire, pre_y) to post-arrival
        (t_fire + delay, post_y), one line per sub-window per top synapse
      - Green dashed horizontal lines mark layer boundaries
      - Integer y-axis showing all neuron indices
      - Y-axis: input at bottom (y=0,1), hidden above, readout at top

    Parameters
    ----------
    n_connections : number of top synapses (by |weight|) to draw lines for
    weights_readout : [K, n_hidden] — used to colour hidden spikes by query
    """
    T, n_in = spike_input.shape
    _,  n_h = hidden_spikes.shape

    # ── Y positions: dense integers, no gap between layers ──
    # Input:  y = 0 .. n_in-1
    # Hidden: y = n_in .. n_in+n_h-1
    # Readout band at y = n_in+n_h (conceptual, no actual spikes)
    y_in  = np.arange(n_in)
    y_hid = np.arange(n_in, n_in + n_h)
    y_ro  = n_in + n_h          # top of the plot
    total = n_in + n_h + 1      # +1 for readout row

    # Layer boundary y-coordinates (between layers)
    bound_in_hid  = n_in - 0.5   # between input and hidden
    bound_hid_ro  = n_in + n_h - 0.5  # between hidden and readout

    # Figure height scales with number of neurons; cap at 16 inches
    fig_h = min(16, max(5, total * 0.30 + 1.5))
    fig, ax = plt.subplots(figsize=(13, fig_h))

    QUERY_COLORS = plt.cm.tab10.colors
    COL_INPUT   = "#1A237E"   # dark blue for input spikes
    COL_HIDDEN  = "#1565C0"   # mid blue for hidden (default)
    COL_READOUT = "#B71C1C"   # dark red for readout bar
    COL_BOUND   = "#2E7D32"   # green for boundary lines

    # ── Green dashed layer-boundary lines ──
    for bound_y in (bound_in_hid, bound_hid_ro):
        ax.axhline(bound_y, color=COL_BOUND, linestyle="--",
                   linewidth=1.5, alpha=0.85, zorder=2)

    # ── Layer labels (right side) ──
    ax.text(T + 0.5, (y_in[0] + y_in[-1]) / 2, "Input",
            va="center", ha="left", fontsize=9, fontweight="bold",
            color=COL_INPUT)
    ax.text(T + 0.5, (y_hid[0] + y_hid[-1]) / 2, "Hidden",
            va="center", ha="left", fontsize=9, fontweight="bold",
            color=COL_HIDDEN)
    ax.text(T + 0.5, y_ro, "Readout",
            va="center", ha="left", fontsize=9, fontweight="bold",
            color=COL_READOUT)

    # ── Y-tick setup: every integer, label every stride ticks ──
    stride = max(1, total // 15)   # keep at most ~15 labels
    all_ys = list(range(total))
    labels = []
    for y in all_ys:
        if y % stride == 0:
            if y < n_in:
                labels.append(["A", "B"][y] if n_in == 2
                               else (f"A{y//2}" if y % 2 == 0 else f"B{y//2}"))
            elif y < n_in + n_h:
                labels.append(str(y))   # plain integer index
            else:
                labels.append("Rdout")
        else:
            labels.append("")
    ax.set_yticks(all_ys)
    ax.set_yticklabels(labels, fontsize=7)
    ax.yaxis.set_tick_params(length=2)

    # ── Colour hidden spikes by dominant readout query ──
    hidden_q_color = [COL_HIDDEN] * n_h
    if weights_readout is not None:
        W = np.atleast_2d(weights_readout)        # [K_or_dim, n_h]
        dominant_q = np.argmax(np.abs(W), axis=0) # [n_h]
        for hh in range(n_h):
            hidden_q_color[hh] = QUERY_COLORS[dominant_q[hh] % len(QUERY_COLORS)]

    # ── Input spike circles ──
    for n in range(n_in):
        ts = np.where(spike_input[:, n] > 0)[0]
        if len(ts):
            ax.scatter(ts, np.full(len(ts), y_in[n]),
                       s=25, color=COL_INPUT, zorder=5, linewidths=0)

    # ── Hidden spike circles ──
    for n in range(n_h):
        ts = np.where(hidden_spikes[:, n] > 0)[0]
        if len(ts):
            ax.scatter(ts, np.full(len(ts), y_hid[n]),
                       s=25, color=hidden_q_color[n], zorder=5, linewidths=0)

    # ── Readout band ──
    t_ro_start, t_ro_end = win_len, win_len + read_len
    ax.axhspan(bound_hid_ro, y_ro + 0.5,
               xmin=t_ro_start / T, xmax=t_ro_end / T,
               alpha=0.12, color=COL_READOUT, zorder=0)
    ax.hlines(y_ro, t_ro_start, t_ro_end,
              colors=COL_READOUT, linewidth=3, alpha=0.8, zorder=4)
    ax.text((t_ro_start + t_ro_end) / 2, y_ro + 0.35,
            "Readout", ha="center", va="bottom",
            fontsize=7, color=COL_READOUT)

    # ── Propagation lines: pre-spike circle → post-spike circle ──
    # Each line connects two actual spike events: the pre-synaptic firing at
    # t_fire and the first post-synaptic firing at t_fire_post >= t_fire + delay.
    # Lines with no matching post-spike are skipped (avoids visual suspension).
    if delays_ih is not None and n_connections > 0:
        importance = (np.abs(weights_ih).flatten() if weights_ih is not None
                      else np.ones(n_in * n_h))
        top_flat   = np.argsort(importance)[::-1][:n_connections]
        pre_idxs   = top_flat // n_h
        post_idxs  = top_flat %  n_h

        sw = sub_win if (sub_win is not None and K > 1) else win_len
        w_max = importance.max() + 1e-8

        for k in range(K):
            t_win_start = k * sw
            t_win_end   = (k + 1) * sw if k < K - 1 else win_len
            q_col = QUERY_COLORS[k % len(QUERY_COLORS)]

            for pre, post in zip(pre_idxs, post_idxs):
                d_val = float(delays_ih[pre, post])
                w_val = float(weights_ih[pre, post]) if weights_ih is not None else 1.0
                alpha = 0.25 + 0.55 * (abs(w_val) / w_max)

                # First spike from pre-neuron within this sub-window
                mask_pre = ((spike_input[:, pre] > 0) &
                            (np.arange(T) >= t_win_start) &
                            (np.arange(T) < t_win_end))
                ts_pre = np.where(mask_pre)[0]
                if len(ts_pre) == 0:
                    continue
                t_fire   = float(ts_pre[0])
                t_arrive = t_fire + d_val

                # First post-synaptic spike at or after signal arrival
                ts_post = np.where(hidden_spikes[:, post] > 0)[0]
                ts_post_valid = ts_post[ts_post >= t_arrive]
                if len(ts_post_valid) == 0:
                    continue
                t_fire_post = float(ts_post_valid[0])

                ls = "-" if w_val >= 0 else "--"
                ax.plot([t_fire, t_fire_post],
                        [float(y_in[pre]), float(y_hid[post])],
                        ls, color=q_col, alpha=alpha, linewidth=1.1, zorder=3)

    # ── Window shading (light) ──
    ax.axvspan(0, win_len,          alpha=0.03, color="royalblue", zorder=0)
    ax.axvspan(win_len, win_len + read_len, alpha=0.06, color="tomato", zorder=0)

    # ── Query sub-window boundaries ──
    if sub_win is not None and K > 1:
        for k in range(K):
            t0 = k * sub_win
            ax.axvline(t0, color=QUERY_COLORS[k % len(QUERY_COLORS)],
                       linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)
            ax.text(t0 + 0.3, y_ro + 0.6, f"Q{k}",
                    fontsize=8, color=QUERY_COLORS[k % len(QUERY_COLORS)],
                    va="bottom", fontweight="bold")

    # ── Axes formatting ──
    ax.set_xlim(-1, T + 2)
    ax.set_ylim(-0.8, y_ro + 1.0)
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Layer neuron index", fontsize=10)
    ax.set_title(title or "Neuron Activity & Synaptic Connections", fontsize=11)
    ax.grid(True, axis="x", alpha=0.2, linewidth=0.5)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color=COL_BOUND,  lw=1.5, linestyle="--",
               label="Layer boundary"),
        Line2D([0], [0], color="gray", lw=1.2, linestyle="-",
               label="Excitatory syn (+w)"),
        Line2D([0], [0], color="gray", lw=1.2, linestyle="--",
               label="Inhibitory syn (−w)"),
    ]
    for k in range(K):
        legend_elements.append(
            Line2D([0], [0], color=QUERY_COLORS[k % len(QUERY_COLORS)],
                   lw=1.5, label=f"Q{k} propagation")
        )
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=7, framealpha=0.85, ncol=2)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_truth_table_raster(
    panels: List[Dict],
    save_path: str,
    win_len: int,
    read_len: int,
    delays_ih: Optional[np.ndarray] = None,
    weights_ih: Optional[np.ndarray] = None,
    weights_readout: Optional[np.ndarray] = None,
    K: int = 1,
    sub_win: Optional[int] = None,
    n_connections: int = 12,
    suptitle: str = "",
):
    """
    2×2 truth table raster: same network responding to all 4 input combinations.

    Each panel shares the same synapse set and y-layout, so differences in spike
    patterns and propagation lines directly reveal the learned mechanism.

    Parameters
    ----------
    panels : list of 4 dicts, each with keys:
        'spike_input'   : np.ndarray [T, n_in]
        'hidden_spikes' : np.ndarray [T, n_h]
        'label'         : str  (e.g. "A=0, B=0 → NAND=1  ✓")
    """
    assert len(panels) == 4, "Need exactly 4 panels for 2×2 truth table"

    T, n_in = panels[0]["spike_input"].shape
    _,  n_h = panels[0]["hidden_spikes"].shape

    y_in  = np.arange(n_in)
    y_hid = np.arange(n_in, n_in + n_h)
    y_ro  = n_in + n_h
    total = n_in + n_h + 1

    bound_in_hid = n_in - 0.5
    bound_hid_ro = n_in + n_h - 0.5

    QUERY_COLORS = plt.cm.tab10.colors
    COL_INPUT   = "#1A237E"
    COL_HIDDEN  = "#1565C0"
    COL_READOUT = "#B71C1C"
    COL_BOUND   = "#2E7D32"

    per_panel_h = min(8, max(4, total * 0.25 + 1.5))
    fig, axes = plt.subplots(2, 2, figsize=(14, per_panel_h * 2))

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.99)

    # Shared synapse selection (same top-N synapses across all panels)
    if delays_ih is not None and n_connections > 0:
        importance = (np.abs(weights_ih).flatten() if weights_ih is not None
                      else np.ones(n_in * n_h))
        top_flat  = np.argsort(importance)[::-1][:n_connections]
        pre_idxs  = top_flat // n_h
        post_idxs = top_flat %  n_h
        w_max = importance.max() + 1e-8
    else:
        pre_idxs = post_idxs = np.array([], dtype=int)
        importance = None
        w_max = 1.0

    # Shared hidden-neuron colour by dominant readout query
    hidden_q_color = [COL_HIDDEN] * n_h
    if weights_readout is not None:
        W = np.atleast_2d(weights_readout)
        dominant_q = np.argmax(np.abs(W), axis=0)
        for hh in range(n_h):
            hidden_q_color[hh] = QUERY_COLORS[dominant_q[hh] % len(QUERY_COLORS)]

    # Shared y-tick labels
    stride = max(1, total // 15)
    all_ys = list(range(total))
    tick_labels = []
    for y in all_ys:
        if y % stride == 0:
            if y < n_in:
                tick_labels.append(["A", "B"][y] if n_in == 2
                                   else (f"A{y//2}" if y % 2 == 0 else f"B{y//2}"))
            elif y < n_in + n_h:
                tick_labels.append(str(y))
            else:
                tick_labels.append("Rdout")
        else:
            tick_labels.append("")

    sw = sub_win if (sub_win is not None and K > 1) else win_len
    t_ro_start, t_ro_end = win_len, win_len + read_len

    for idx, panel in enumerate(panels):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        s_in  = panel["spike_input"]
        s_hid = panel["hidden_spikes"]
        lbl   = panel.get("label", f"Panel {idx}")

        # Green dashed layer-boundary lines
        for bound_y in (bound_in_hid, bound_hid_ro):
            ax.axhline(bound_y, color=COL_BOUND, linestyle="--",
                       linewidth=1.5, alpha=0.85, zorder=2)

        # Layer labels (right column only)
        if col == 1:
            ax.text(T + 0.5, (y_in[0] + y_in[-1]) / 2, "Input",
                    va="center", ha="left", fontsize=8, fontweight="bold",
                    color=COL_INPUT)
            ax.text(T + 0.5, (y_hid[0] + y_hid[-1]) / 2, "Hidden",
                    va="center", ha="left", fontsize=8, fontweight="bold",
                    color=COL_HIDDEN)
            ax.text(T + 0.5, y_ro, "Readout",
                    va="center", ha="left", fontsize=8, fontweight="bold",
                    color=COL_READOUT)

        # Y-ticks (labels only on left column)
        ax.set_yticks(all_ys)
        ax.set_yticklabels(tick_labels if col == 0 else [""] * len(all_ys),
                           fontsize=7)
        ax.yaxis.set_tick_params(length=2)

        # Input spikes
        for n in range(n_in):
            ts = np.where(s_in[:, n] > 0)[0]
            if len(ts):
                ax.scatter(ts, np.full(len(ts), y_in[n]),
                           s=30, color=COL_INPUT, zorder=5, linewidths=0)

        # Hidden spikes
        for n in range(n_h):
            ts = np.where(s_hid[:, n] > 0)[0]
            if len(ts):
                ax.scatter(ts, np.full(len(ts), y_hid[n]),
                           s=30, color=hidden_q_color[n], zorder=5, linewidths=0)

        # Readout band
        ax.axhspan(bound_hid_ro, y_ro + 0.5,
                   xmin=t_ro_start / T, xmax=t_ro_end / T,
                   alpha=0.12, color=COL_READOUT, zorder=0)
        ax.hlines(y_ro, t_ro_start, t_ro_end,
                  colors=COL_READOUT, linewidth=3, alpha=0.8, zorder=4)

        # Propagation lines: pre-spike circle → post-spike circle
        # Skips connections where the post-synaptic neuron never fires after arrival.
        if delays_ih is not None and len(pre_idxs) > 0:
            for k in range(K):
                t_win_start = k * sw
                t_win_end   = (k + 1) * sw if k < K - 1 else win_len
                q_col = QUERY_COLORS[k % len(QUERY_COLORS)]

                for pre, post in zip(pre_idxs, post_idxs):
                    d_val = float(delays_ih[pre, post])
                    w_val = float(weights_ih[pre, post]) if weights_ih is not None else 1.0
                    alpha = 0.30 + 0.55 * (abs(w_val) / w_max)

                    mask_pre = ((s_in[:, pre] > 0) &
                                (np.arange(T) >= t_win_start) &
                                (np.arange(T) < t_win_end))
                    ts_pre = np.where(mask_pre)[0]
                    if len(ts_pre) == 0:
                        continue
                    t_fire   = float(ts_pre[0])
                    t_arrive = t_fire + d_val

                    ts_post = np.where(s_hid[:, post] > 0)[0]
                    ts_post_valid = ts_post[ts_post >= t_arrive]
                    if len(ts_post_valid) == 0:
                        continue
                    t_fire_post = float(ts_post_valid[0])

                    ls = "-" if w_val >= 0 else "--"
                    ax.plot([t_fire, t_fire_post],
                            [float(y_in[pre]), float(y_hid[post])],
                            ls, color=q_col, alpha=alpha, linewidth=1.1, zorder=3)

        # Window shading
        ax.axvspan(0, win_len, alpha=0.03, color="royalblue", zorder=0)
        ax.axvspan(t_ro_start, t_ro_end, alpha=0.07, color="tomato", zorder=0)

        # Query sub-window boundaries for K > 1
        if sub_win is not None and K > 1:
            for k in range(K):
                t0 = k * sub_win
                ax.axvline(t0, color=QUERY_COLORS[k % len(QUERY_COLORS)],
                           linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)

        ax.set_title(lbl, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlim(-1, T + 2)
        ax.set_ylim(-0.8, y_ro + 1.0)
        ax.grid(True, axis="x", alpha=0.2, linewidth=0.5)

        if row == 1:
            ax.set_xlabel("Time (ms)", fontsize=9)
        if col == 0:
            ax.set_ylabel("Layer neuron index", fontsize=9)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COL_BOUND, lw=1.5, linestyle="--",
               label="Layer boundary"),
        Line2D([0], [0], color="gray", lw=1.2, linestyle="-",
               label="Excitatory (+w)"),
        Line2D([0], [0], color="gray", lw=1.2, linestyle="--",
               label="Inhibitory (−w)"),
        Line2D([0], [0], color="none", lw=0,
               label="Circles = spike events"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               fontsize=8, ncol=4, framealpha=0.85,
               bbox_to_anchor=(0.5, 0.01))

    fig.subplots_adjust(left=0.08, right=0.88, top=0.96, bottom=0.10,
                        hspace=0.38, wspace=0.06)
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

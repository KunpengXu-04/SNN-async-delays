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

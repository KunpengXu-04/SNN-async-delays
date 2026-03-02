"""
Visualisation utilities.

Functions
---------
plot_training_curves  : loss / accuracy over epochs
plot_delay_distribution : heatmap of learned delays
plot_K_accuracy       : accuracy vs K sweep curve
plot_spike_raster     : spike raster for one trial
"""

from __future__ import annotations
import os
from typing import List, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------

def plot_training_curves(log_rows: List[Dict], save_path: str):
    epochs     = [r["epoch"]      for r in log_rows]
    train_acc  = [r["train_acc"]  for r in log_rows]
    val_acc    = [r["val_acc"]    for r in log_rows]
    train_loss = [r["train_loss"] for r in log_rows]
    val_loss   = [r["val_loss"]   for r in log_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss,   label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].set_title("Loss")

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc,   label="val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.95, color="gray", linestyle="--", linewidth=0.8)
    axes[1].legend(); axes[1].set_title("Accuracy")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------

def plot_delay_distribution(delays: np.ndarray, title: str, save_path: str):
    """
    delays : [N_pre, N_post] array of delay values (ms or steps)
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(delays, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="delay (steps)")
    ax.set_xlabel("Post-synaptic neuron")
    ax.set_ylabel("Pre-synaptic neuron")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------

def plot_K_accuracy(
    K_values:   List[int],
    results_by_mode: Dict[str, List[float]],   # {mode: [acc for each K]}
    tau:        float = 0.95,
    save_path:  str   = "k_accuracy.png",
):
    fig, ax = plt.subplots(figsize=(7, 4))

    for mode, accs in results_by_mode.items():
        ax.plot(K_values, accs, marker="o", label=mode)

    ax.axhline(tau, color="gray", linestyle="--", linewidth=0.8, label=f"τ={tau}")
    ax.set_xlabel("K (queries per trial)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(K_values)
    ax.legend()
    ax.set_title("Accuracy vs K")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------

def plot_throughput(
    K_values:   List[int],
    results_by_mode: Dict[str, List[float]],   # {mode: [throughput for each K]}
    save_path:  str = "throughput.png",
):
    fig, ax = plt.subplots(figsize=(7, 4))

    for mode, thr in results_by_mode.items():
        ax.plot(K_values, thr, marker="s", label=mode)

    ax.set_xlabel("K (queries per trial)")
    ax.set_ylabel("K / total_hidden_spikes")
    ax.set_xticks(K_values)
    ax.legend()
    ax.set_title("Energy-normalised throughput vs K")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------

def plot_spike_raster(
    spike_train: np.ndarray,   # [T, N]
    title: str,
    save_path: str,
    slot_boundaries: Optional[List] = None,   # list of SlotBoundaries
):
    T, N = spike_train.shape
    fig, ax = plt.subplots(figsize=(10, 4))

    for n in range(N):
        times = np.where(spike_train[:, n] > 0)[0]
        ax.scatter(times, np.full_like(times, n), s=2, c="black", marker="|")

    if slot_boundaries is not None:
        for slot in slot_boundaries:
            ax.axvspan(slot.win_start,  slot.win_end,   alpha=0.1, color="blue")
            ax.axvspan(slot.read_start, slot.read_end,  alpha=0.1, color="red")

    ax.set_xlabel("Timestep (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)

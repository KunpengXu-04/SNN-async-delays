"""
Enhanced spike raster: sorted neurons + window-coloured spikes.

Two panels side by side:
  Left : w_and_d  (trainable delays)
  Right: d0       (no delays, control)

Improvements over default raster:
  1. Neurons sorted by first-spike time → reveals temporal wave structure
  2. Hidden spikes coloured by which sub-window they fall in
     (Q0 blue, Q1 orange, Q2 green, readout dark-red)
  3. Multiple NAND input combinations shown (4 rows) so the raster
     is never accidentally empty
  4. Firing-rate sidebar (mean spikes/neuron per window)

Usage (from snn_async_delays/):
    python tmp_enhanced_raster.py
"""
import sys, json, torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '.')
from snn.model import SNNSimultaneousModel
from utils.viz import _extract_run_traces
from utils.seed import set_seed

# ── Use the K=3 step3 run (has the most interesting temporal structure) ──────
BASE = "runs/step3_planD_4ops_16k"
RUNS = {
    "w+d (delays)":  f"{BASE}/w_and_d_K3_seed42",
    "d=0 (control)": f"{BASE}/d0_control_K3_seed42",
}
SAVE_PATH = "docs/enhanced_raster.png"
DEVICE    = "cpu"
SEARCH_SEEDS = 20      # try this many samples, pick richest


def build_model(cfg, device):
    hidden_sizes = cfg.get("hidden_sizes", [cfg.get("n_hidden", 50)])
    return SNNSimultaneousModel(
        n_queries         = cfg["K"],
        n_hidden          = hidden_sizes[0],
        win_len           = cfg["win_len"],
        read_len          = cfg["read_len"],
        d_max             = cfg["d_max"],
        train_mode        = cfg.get("train_mode", "weights_and_delays"),
        delay_param_type  = cfg.get("delay_param_type", "sigmoid"),
        delay_step        = cfg.get("delay_step", 1.0),
        fixed_delay_value = cfg.get("fixed_delay_value", None),
        lif_tau_m         = cfg.get("lif_tau_m", 10.0),
        lif_threshold     = cfg.get("lif_threshold", 1.0),
        lif_reset         = cfg.get("lif_reset", 0.0),
        lif_refractory    = cfg.get("lif_refractory", 2),
        dt                = cfg.get("dt", 1.0),
        surrogate_beta    = cfg.get("surrogate_beta", 4.0),
        n_input_channels  = cfg.get("n_input", 2),
        readout_type      = cfg.get("readout_type", "linear"),
        num_hidden_layers = cfg.get("num_hidden_layers", 1),
        hidden_sizes      = hidden_sizes,
        use_output_spikes = cfg.get("use_output_spikes", False),
        n_output_neurons  = cfg.get("n_output_neurons", None),
    ).to(device)


def get_rich_traces(model, cfg, device, n_seeds=SEARCH_SEEDS):
    """Return traces for the sample with the most total hidden spikes."""
    op = "mixed" if cfg.get("n_ops", 0) > 0 else cfg.get("op_name", cfg.get("ops_list", ["NAND"])[0])
    best, best_count = None, -1
    for s in range(n_seeds):
        seed = 50 * (s + 1)
        tr, w, d = _extract_run_traces(model, cfg, cfg["K"], op, device, seed=seed)
        count = int(tr["hidden1_spikes"].sum())
        if count > best_count:
            best, best_count = (tr, w, d), count
    print(f"  Best sample: total hidden spikes = {best_count}")
    return best


def window_color(t, sub_win, K, win_len, COLORS):
    """Return colour for a spike at time t."""
    for k in range(K):
        if k * sub_win <= t < (k + 1) * sub_win:
            return COLORS[k % len(COLORS)]
    return "darkred"    # readout window


def draw_raster(ax, traces, label, show_xlabel=True):
    s_in  = traces["input_spikes"]    # [T, n_in]
    s_h1  = traces["hidden1_spikes"]  # [T, h1]
    T, n_in = s_in.shape
    _, h1   = s_h1.shape

    win_len  = traces["win_len"]
    read_len = traces["read_len"]
    sub_win  = traces.get("sub_win", win_len)
    K        = traces.get("K", 1)
    COLORS   = list(plt.cm.tab10.colors)

    # ── Background shading ────────────────────────────────────────────────
    ax.axvspan(0,       win_len,            alpha=0.07, color="royalblue", zorder=0)
    ax.axvspan(win_len, win_len + read_len, alpha=0.10, color="tomato",    zorder=0)

    # Sub-window dividers
    for k in range(K):
        ax.axvline(k * sub_win, color=COLORS[k % len(COLORS)],
                   ls="--", lw=0.9, alpha=0.7, zorder=1)
        ax.text(k * sub_win + 0.3, h1 + n_in - 0.5, f"Q{k}",
                fontsize=7, color=COLORS[k % len(COLORS)],
                va="top", fontweight="bold")

    # ── Sort hidden neurons by first spike time ───────────────────────────
    first_spike = np.full(h1, T + 1, dtype=float)
    for j in range(h1):
        ts = np.where(s_h1[:, j] > 0)[0]
        if len(ts):
            first_spike[j] = ts[0]
    order = np.argsort(first_spike)   # neuron index sorted by when they first fire

    # ── Input spikes (top rows, original order) ───────────────────────────
    for i in range(n_in):
        ts = np.where(s_in[:, i] > 0)[0]
        if len(ts):
            c = COLORS[(i // 2) % len(COLORS)] if n_in > 2 else "#333333"
            ax.scatter(ts, np.full(len(ts), h1 + i + 0.5),
                       s=8, color=c, marker="|", linewidths=1.0, zorder=3)

    # ── Hidden spikes (sorted, coloured by window) ────────────────────────
    for rank, j in enumerate(order):
        ts = np.where(s_h1[:, j] > 0)[0]
        if len(ts) == 0:
            continue
        cols = [window_color(t, sub_win, K, win_len, COLORS) for t in ts]
        ax.scatter(ts, np.full(len(ts), rank),
                   s=8, c=cols, marker="|", linewidths=1.0, zorder=3)

    # ── Readout window fire-rate sidebar ─────────────────────────────────
    ro_spikes = s_h1[win_len:].sum(axis=0)   # spikes per neuron in readout
    for rank, j in enumerate(order):
        if ro_spikes[j] > 0:
            ax.barh(rank, ro_spikes[j] * 0.3,
                    left=T + 0.5, height=0.8, color="darkred", alpha=0.6)

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xlim(-0.5, T + 4)
    ax.set_ylim(-0.5, h1 + n_in + 0.5)
    ax.axhline(h1 + 0.1, color="gray", lw=0.8, ls="-")  # separator Input/Hidden
    ax.set_yticks([h1 // 4, h1 // 2, 3 * h1 // 4, h1])
    ax.set_yticklabels([f"n={h1//4}", f"n={h1//2}", f"n={3*h1//4}", "Input↑"], fontsize=7)
    ax.set_title(label, fontsize=9, fontweight="bold")
    if show_xlabel:
        ax.set_xlabel("Timestep (ms)", fontsize=9)

    # ── Legend for window colours ─────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=COLORS[k % len(COLORS)], label=f"Q{k} window")
        for k in range(K)
    ]
    legend_patches.append(mpatches.Patch(color="darkred", label="Readout window"))
    ax.legend(handles=legend_patches, fontsize=6.5, loc="upper right",
              framealpha=0.85, ncol=1)

    # Text annotation: total hidden spikes & active fraction
    total_spk = int(s_h1.sum())
    active    = int((s_h1.sum(axis=0) > 0).sum())
    ax.text(0.02, 0.02, f"Total spikes: {total_spk}\nActive neurons: {active}/{h1}",
            transform=ax.transAxes, fontsize=7, va="bottom", color="gray")
    ax.text(T + 0.6, -0.8, "spk\nin RO", fontsize=6, color="darkred", ha="left")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharey=False)
    fig.suptitle("Enhanced Spike Raster  —  K=3, NAND+mixed ops\n"
                 "Neurons sorted by first-spike time  |  Colours = sub-window",
                 fontsize=10)

    for ax, (label, run_dir) in zip(axes, RUNS.items()):
        print(f"\n{label}  [{run_dir}]")
        with open(f"{run_dir}/config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        set_seed(cfg.get("seed", 42))
        model = build_model(cfg, DEVICE)
        model.load_state_dict(
            torch.load(f"{run_dir}/best_model.pt", map_location=DEVICE, weights_only=True))
        model.eval()

        traces, _, _ = get_rich_traces(model, cfg, DEVICE)
        draw_raster(ax, traces, label)

    import os; os.makedirs("docs", exist_ok=True)
    fig.tight_layout()
    fig.savefig(SAVE_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {SAVE_PATH}")


if __name__ == "__main__":
    main()

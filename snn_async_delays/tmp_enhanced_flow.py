"""
Enhanced spike flow: explicitly shows 3 connection types.

  (A) BLUE/RED fan lines     : Input spike → delayed ARRIVAL at Hidden neuron
                               (floating ends = arrived but maybe didn't fire)

  (B) ORANGE horizontal span : recent arrivals → Hidden neuron FIRE
                               (shows which input signals charged this neuron)

  (C) GREEN/PINK vertical    : Hidden fire IN readout window → Readout accumulation
                               (shows which hidden fires actually count for the answer)

Usage (from snn_async_delays/):
    python tmp_enhanced_flow.py
"""
import sys, json, torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, '.')

from snn.model import SNNSimultaneousModel
from utils.viz import _extract_run_traces
from utils.seed import set_seed

# ── Config ──────────────────────────────────────────────────────────────────
RUN_DIR   = "runs/step2_planD/step2_seq_NAND_w_and_d_continuous_h50_K1_sw10_seed42"
SAVE_PATH = "docs/enhanced_spike_flow.png"
DEVICE    = "cpu"
W_THRESH  = 0.08    # min |weight| to draw a fan line
RO_THRESH = 0.05    # min |readout weight| to draw a readout line
MAX_SEEDS = 15       # search this many samples for the richest one


def build_model(cfg, device):
    return SNNSimultaneousModel(
        n_queries         = cfg["K"],
        n_hidden          = cfg.get("hidden_sizes", [cfg.get("n_hidden", 50)])[0],
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
        hidden_sizes      = cfg.get("hidden_sizes", [cfg.get("n_hidden", 50)]),
        use_output_spikes = cfg.get("use_output_spikes", False),
        n_output_neurons  = cfg.get("n_output_neurons", None),
    ).to(device)


def find_best_sample(model, cfg, device, max_seeds=MAX_SEEDS):
    """Try several seeds; return traces/weights/delays for the sample
    with the most hidden fires inside the readout window."""
    best_traces, best_w, best_d, best_count, best_seed = None, None, None, -1, 0
    op = "mixed" if cfg.get("n_ops", 0) > 0 else cfg.get("op_name", cfg.get("ops_list", ["NAND"])[0])
    for s in range(max_seeds):
        seed = 100 * (s + 1)
        tr, w, d = _extract_run_traces(model, cfg, cfg["K"], op, device, seed=seed)
        s_h1     = tr["hidden1_spikes"]   # [T, h1]
        win_len  = tr["win_len"]
        count    = int(s_h1[win_len:].sum())
        if count > best_count:
            best_traces, best_w, best_d = tr, w, d
            best_count = count
            best_seed  = seed
    print(f"Best sample: seed={best_seed}, hidden fires in readout window={best_count}")
    return best_traces, best_w, best_d


def plot_enhanced(traces, w_dict, d_dict, save_path):
    s_in  = traces["input_spikes"]    # [T, n_in]
    s_h1  = traces["hidden1_spikes"]  # [T, h1]
    T, n_in = s_in.shape
    _, h1   = s_h1.shape

    win_len  = traces["win_len"]
    read_len = traces["read_len"]
    tau_m    = 10.0    # default

    W_ih = w_dict["ih"]                # [n_in, h1]
    D_ih = d_dict["ih"]                # [n_in, h1]
    W_ro = w_dict.get("readout")       # [K, h1]  (K=1 → [1, h1])

    GAP     = max(3, n_in // 4 + 1)
    y0_in   = 0
    y0_h1   = n_in + GAP
    y_read  = y0_h1 + h1 + GAP        # readout bar y position

    def yi(n): return float(y0_in + n)
    def yh(n): return float(y0_h1 + n)

    fig_h = max(7, int(y_read * 0.30 + 3))
    fig, ax = plt.subplots(figsize=(14, fig_h))

    # ── Background shading ─────────────────────────────────────────────────
    ax.axvspan(0,        win_len,            alpha=0.06, color="royalblue", zorder=0)
    ax.axvspan(win_len,  win_len + read_len, alpha=0.10, color="tomato",    zorder=0)

    # Sub-window dividers (K>1)
    sub_win = traces.get("sub_win")
    K       = traces.get("K", 1)
    COLORS  = list(plt.cm.tab10.colors)
    if sub_win and K > 1:
        for k in range(K):
            ax.axvline(k * sub_win, color=COLORS[k % len(COLORS)],
                       ls=":", lw=1.0, alpha=0.5, zorder=1)

    # ── Layer bands & labels ───────────────────────────────────────────────
    ax.axhspan(y0_in - 0.6, y0_in + n_in - 0.4, alpha=0.07, color="royalblue", zorder=0)
    ax.axhspan(y0_h1 - 0.6, y0_h1 + h1  - 0.4, alpha=0.06, color="steelblue", zorder=0)

    for lbl, yl in [("Input", (y0_in + n_in / 2)),
                     ("Hidden", (y0_h1 + h1 / 2))]:
        ax.text(-1.5, yl, lbl, ha="right", va="center",
                fontsize=9, fontweight="bold", color="steelblue")

    # Readout bar
    ax.axhline(y_read, xmin=win_len/T, xmax=1.0,
               color="tomato", lw=2.5, alpha=0.7, zorder=3)
    ax.text(win_len + (T - win_len)/2, y_read + 0.3, "Readout accumulation",
            ha="center", va="bottom", fontsize=8, color="tomato")

    # ══════════════════════════════════════════════════════════════════════
    # (A) Fan lines: input spike → delayed arrival at hidden neuron
    # ══════════════════════════════════════════════════════════════════════
    fan_segs, fan_colors = [], []
    for i in range(n_in):
        ts_pre = np.where(s_in[:, i] > 0)[0]
        for j in range(h1):
            w = float(W_ih[i, j])
            if abs(w) <= W_THRESH:
                continue
            d = float(D_ih[i, j])
            c = (0.9, 0.1, 0.1, 0.07) if w > 0 else (0.1, 0.1, 0.9, 0.07)
            for t in ts_pre:
                arr = t + d
                if 0 <= arr < T:
                    fan_segs.append([(t, yi(i)), (arr, yh(j))])
                    fan_colors.append(c)

    if fan_segs:
        lc = LineCollection(fan_segs, colors=fan_colors, linewidths=0.6, zorder=2)
        ax.add_collection(lc)

    # Input spike markers
    for i in range(n_in):
        ts = np.where(s_in[:, i] > 0)[0]
        if len(ts):
            ax.scatter(ts, np.full(len(ts), yi(i)),
                       s=18, color="#222222", marker="|", linewidths=1.2, zorder=4)

    # ══════════════════════════════════════════════════════════════════════
    # (B) Arrival→Fire connectors  +  hidden spike markers
    #     For each hidden neuron j that fires at t_fire:
    #       - find all input arrivals within [t_fire - tau_m, t_fire]
    #       - draw a shaded span on that hidden neuron's y row
    #       - mark t_fire with a gold star
    # ══════════════════════════════════════════════════════════════════════
    for j in range(h1):
        t_fires = np.where(s_h1[:, j] > 0)[0]
        if len(t_fires) == 0:
            continue

        for t_fire in t_fires:
            # --- Collect arrivals that could have contributed ---
            arrivals = []
            for i in range(n_in):
                ts_pre = np.where(s_in[:, i] > 0)[0]
                for t in ts_pre:
                    arr = t + float(D_ih[i, j])
                    if (t_fire - tau_m) <= arr <= t_fire and abs(float(W_ih[i, j])) > W_THRESH:
                        arrivals.append(arr)

            if arrivals:
                t_earliest = min(arrivals)
                # Orange horizontal span: earliest relevant arrival → fire time
                ax.plot([t_earliest, t_fire], [yh(j), yh(j)],
                        color="darkorange", lw=2.0, alpha=0.75,
                        solid_capstyle="round", zorder=5)
                # Small dot at each arrival endpoint
                for arr in arrivals:
                    ax.scatter(arr, yh(j), s=20, color="darkorange",
                               marker="o", alpha=0.6, zorder=6, linewidths=0)

            # Star at actual fire time: green if in readout window, gold if in input window
            # zorder=12 → always on top of vertical lines and orange spans
            star_col  = "limegreen" if t_fire >= win_len else "gold"
            star_edge = "darkgreen"  if t_fire >= win_len else "darkorange"
            ax.scatter(t_fire, yh(j), s=100, color=star_col, marker="*",
                       edgecolors=star_edge, linewidths=0.6, zorder=12)

    # ══════════════════════════════════════════════════════════════════════
    # (C) Hidden fire → Readout vertical lines
    #     Only for fires INSIDE the readout window [win_len, T)
    #     Colored by readout weight sign, width by magnitude
    # ══════════════════════════════════════════════════════════════════════
    if W_ro is not None:
        # W_ro shape: [K, h1] for linear readout
        ro_weights = W_ro[0]  # [h1]  (K=1)
        max_w_ro   = np.max(np.abs(ro_weights)) + 1e-8
        n_ro_lines = 0

        for j in range(h1):
            w_ro = float(ro_weights[j])
            if abs(w_ro) <= RO_THRESH:
                continue
            t_fires_ro = np.where(s_h1[win_len:, j] > 0)[0] + win_len
            if len(t_fires_ro) == 0:
                continue

            col   = "crimson"   if w_ro > 0 else "mediumblue"
            mag   = min(abs(w_ro) / max_w_ro, 1.0)
            lw    = 4.0 + 5.0 * mag   # 4 ~ 9 px thick
            alpha = 0.90

            for t_fire in t_fires_ro:
                n_ro_lines += 1
                # Vertical line drawn BEHIND orange spans and stars (zorder=3)
                ax.plot([t_fire, t_fire], [0, y_read],
                        color=col, lw=lw, alpha=alpha,
                        solid_capstyle="butt", zorder=3,
                        solid_joinstyle="round")
                # Triangle at readout bar — in front of everything
                ax.scatter(t_fire, y_read, s=120, color=col, marker="^",
                           alpha=1.0, zorder=13, linewidths=0)

        print(f"  Readout vertical lines drawn: {n_ro_lines}")

    # ── Axes & legend ──────────────────────────────────────────────────────
    ax.set_xlim(-2, T + 0.5)
    ax.set_ylim(y0_in - 1.0, y_read + 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Time (ms)", fontsize=11)

    title_str = traces.get("_title", "")
    ax.set_title(f"Enhanced Spike Flow  |  K={K}  {title_str}", fontsize=10)

    legend_elems = [
        Line2D([0],[0], color=(0.9,0.1,0.1,0.5), lw=1.5,
               label="(A) Input→arrival (excitatory)"),
        Line2D([0],[0], color=(0.1,0.1,0.9,0.5), lw=1.5,
               label="(A) Input→arrival (inhibitory)"),
        mpatches.Patch(color="darkorange", alpha=0.75,
                       label="(B) Arrival window → Hidden fire (orange span)"),
        Line2D([0],[0], marker="*", color="gold",      markeredgecolor="darkorange",
               markersize=9, lw=0, label="(B) Hidden fire in INPUT window (gold ★, not counted)"),
        Line2D([0],[0], marker="*", color="limegreen", markeredgecolor="darkgreen",
               markersize=9, lw=0, label="(B) Hidden fire in READOUT window (green ★, counted!)"),
        Line2D([0],[0], color="crimson",    lw=3,
               label="(C) Hidden→Readout bar (positive weight → votes '1')"),
        Line2D([0],[0], color="mediumblue", lw=3,
               label="(C) Hidden→Readout bar (negative weight → votes '0')"),
    ]
    ax.legend(handles=legend_elems, fontsize=7.5, loc="upper left",
              framealpha=0.9, ncol=1)

    # Time annotations
    ax.axvline(win_len, color="gray", lw=1.0, ls="--", alpha=0.5, zorder=1)
    ax.text(win_len/2, y_read + 1.0, "Input window\n(0→10 ms)",
            ha="center", va="bottom", fontsize=8, color="royalblue")
    ax.text(win_len + read_len/2, y_read + 1.0, "Readout window\n(10→20 ms)",
            ha="center", va="bottom", fontsize=8, color="tomato")

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    with open(f"{RUN_DIR}/config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    set_seed(cfg.get("seed", 42))
    model = build_model(cfg, DEVICE)
    model.load_state_dict(
        torch.load(f"{RUN_DIR}/best_model.pt", map_location=DEVICE, weights_only=True))
    model.eval()

    traces, w_dict, d_dict = find_best_sample(model, cfg, DEVICE)
    plot_enhanced(traces, w_dict, d_dict, SAVE_PATH)


if __name__ == "__main__":
    main()

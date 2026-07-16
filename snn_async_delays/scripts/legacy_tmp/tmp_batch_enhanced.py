"""
Batch-generate all enhanced rasters and spike-flow plots.

Output layout:
  docs/enhanced/
    raster/
      4ops16k_K{1,2,3}_wad_vs_d0.png       main science
      h_sweep_K3_wad_h{10,20,30,50}.png    resource sweep (4-panel)
      depth_K3_L1_vs_L2.png                architecture (linear readout)
      planD_NAND_K3_wad_vs_d0.png          single-op NAND linear readout
    flow/
      planD_h50_K{1,2,3}_wad_linear.png    linear readout → shows vote lines
      planD_h50_K3_d0_linear.png           d=0 linear, contrast
      4ops16k_K{1,3}_wad_mlp.png           MLP readout
      h10_K3_wad_vs_h50_K3_wad.png         resource contrast in flow

Usage (from snn_async_delays/):
    python tmp_batch_enhanced.py
"""
import sys, os, json, torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

sys.path.insert(0, '.')
from snn.model import SNNSimultaneousModel
from utils.viz import _extract_run_traces
from utils.seed import set_seed

DEVICE     = "cpu"
OUT_BASE   = "docs/enhanced"
N_SEEDS    = 20      # candidate samples per run
W_THRESH   = 0.08
RO_THRESH  = 0.05
COLORS     = list(plt.cm.tab10.colors)

os.makedirs(f"{OUT_BASE}/raster", exist_ok=True)
os.makedirs(f"{OUT_BASE}/flow",   exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_model(run_dir, device=DEVICE):
    with open(f"{run_dir}/config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    hidden_sizes = cfg.get("hidden_sizes", [cfg.get("n_hidden", 50)])
    set_seed(cfg.get("seed", 42))
    model = SNNSimultaneousModel(
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
    model.load_state_dict(
        torch.load(f"{run_dir}/best_model.pt", map_location=device, weights_only=True))
    model.eval()
    return cfg, model


def get_traces(model, cfg, n_seeds=N_SEEDS):
    op = ("mixed" if cfg.get("n_ops", 0) > 0
          else cfg.get("op_name", cfg.get("ops_list", ["NAND"])[0]))
    best, best_n = None, -1
    for s in range(n_seeds):
        tr, w, d = _extract_run_traces(model, cfg, cfg["K"], op, DEVICE, seed=50*(s+1))
        n = int(tr["hidden1_spikes"].sum())
        if n > best_n:
            best, best_n = (tr, w, d), n
    return best


# ═══════════════════════════════════════════════════════════════════════════
# SHARED RASTER DRAWING
# ═══════════════════════════════════════════════════════════════════════════

def window_color(t, sub_win, K):
    for k in range(K):
        if k * sub_win <= t < (k + 1) * sub_win:
            return COLORS[k % len(COLORS)]
    return "darkred"


def draw_raster_panel(ax, traces, label, accstr=""):
    s_in  = traces["input_spikes"]
    s_h1  = traces["hidden1_spikes"]
    T, n_in = s_in.shape
    _, h1   = s_h1.shape
    win_len  = traces["win_len"]
    read_len = traces["read_len"]
    sub_win  = traces.get("sub_win", win_len)
    K        = traces.get("K", 1)

    ax.axvspan(0,       win_len,            alpha=0.07, color="royalblue", zorder=0)
    ax.axvspan(win_len, win_len + read_len, alpha=0.10, color="tomato",    zorder=0)
    for k in range(K):
        ax.axvline(k * sub_win, color=COLORS[k % len(COLORS)],
                   ls="--", lw=0.9, alpha=0.65, zorder=1)
        ax.text(k * sub_win + 0.3, h1 + n_in - 0.3,
                f"Q{k}", fontsize=6.5, color=COLORS[k % len(COLORS)],
                va="top", fontweight="bold")

    # Sort hidden by first spike
    first = np.full(h1, T + 1, dtype=float)
    for j in range(h1):
        ts = np.where(s_h1[:, j] > 0)[0]
        if len(ts): first[j] = ts[0]
    order = np.argsort(first)

    # Input rows
    for i in range(n_in):
        ts = np.where(s_in[:, i] > 0)[0]
        if len(ts):
            c = COLORS[(i // 2) % len(COLORS)] if n_in > 2 else "#333333"
            ax.scatter(ts, np.full(len(ts), h1 + i + 0.5),
                       s=7, color=c, marker="|", linewidths=1.0, zorder=3)

    # Hidden rows (sorted, window-coloured)
    for rank, j in enumerate(order):
        ts = np.where(s_h1[:, j] > 0)[0]
        if len(ts) == 0: continue
        cols = [window_color(t, sub_win, K) for t in ts]
        ax.scatter(ts, np.full(len(ts), rank),
                   s=7, c=cols, marker="|", linewidths=1.0, zorder=3)

    # Sidebar: readout-window spikes per neuron
    ro = s_h1[win_len:].sum(axis=0)
    max_ro = max(ro.max(), 1)
    for rank, j in enumerate(order):
        if ro[j] > 0:
            ax.barh(rank, ro[j] / max_ro * 3,
                    left=T + 0.3, height=0.8, color="darkred", alpha=0.55)

    ax.set_xlim(-0.5, T + 4.5)
    ax.set_ylim(-0.5, h1 + n_in + 0.5)
    ax.axhline(h1 + 0.1, color="gray", lw=0.7, ls="-")
    ax.set_yticks([0, h1 // 4, h1 // 2, 3 * h1 // 4, h1])
    ax.set_yticklabels([f"n=0", f"n={h1//4}", f"n={h1//2}",
                        f"n={3*h1//4}", "↑In"], fontsize=6.5)
    ax.set_xlabel("Time (ms)", fontsize=8)
    ttl = f"{label}"
    if accstr: ttl += f"\n{accstr}"
    ax.set_title(ttl, fontsize=8, fontweight="bold")

    total_spk = int(s_h1.sum())
    active    = int((s_h1.sum(axis=0) > 0).sum())
    ro_spk    = int(s_h1[win_len:].sum())
    ax.text(0.02, 0.02,
            f"spk={total_spk}  active={active}/{h1}  ro_spk={ro_spk}",
            transform=ax.transAxes, fontsize=6, va="bottom", color="gray")


def legend_patches(K):
    patches = [mpatches.Patch(color=COLORS[k % len(COLORS)], label=f"Q{k} window")
               for k in range(K)]
    patches.append(mpatches.Patch(color="darkred", label="Readout window"))
    return patches


# ═══════════════════════════════════════════════════════════════════════════
# SHARED SPIKE-FLOW DRAWING
# ═══════════════════════════════════════════════════════════════════════════

def draw_flow_panel(ax, traces, w_dict, d_dict, label):
    s_in  = traces["input_spikes"]
    s_h1  = traces["hidden1_spikes"]
    s_h2  = traces.get("hidden2_spikes")
    T, n_in = s_in.shape
    _, h1   = s_h1.shape
    win_len  = traces["win_len"]
    read_len = traces["read_len"]
    sub_win  = traces.get("sub_win")
    K        = traces.get("K", 1)

    GAP    = max(3, n_in // 4 + 1)
    y0_in  = 0
    y0_h1  = n_in + GAP
    y_read = y0_h1 + h1 + GAP
    def yi(n): return float(y0_in + n)
    def yh(n): return float(y0_h1 + n)

    ax.axvspan(0,       win_len,            alpha=0.06, color="royalblue", zorder=0)
    ax.axvspan(win_len, win_len + read_len, alpha=0.09, color="tomato",    zorder=0)
    ax.axhspan(y0_in - 0.6, y0_in + n_in - 0.4, alpha=0.07, color="royalblue", zorder=0)
    ax.axhspan(y0_h1 - 0.6, y0_h1 + h1  - 0.4, alpha=0.05, color="steelblue", zorder=0)

    if sub_win and K > 1:
        for k in range(K):
            ax.axvline(k * sub_win, color=COLORS[k % len(COLORS)],
                       ls=":", lw=0.9, alpha=0.5, zorder=1)

    # (A) Fan lines
    W_ih = w_dict["ih"]; D_ih = d_dict["ih"]
    segs, cols_ = [], []
    for i in range(n_in):
        ts_pre = np.where(s_in[:, i] > 0)[0]
        for j in range(h1):
            w = float(W_ih[i, j])
            if abs(w) <= W_THRESH: continue
            c = (0.9, 0.1, 0.1, 0.07) if w > 0 else (0.1, 0.1, 0.9, 0.07)
            d = float(D_ih[i, j])
            for t in ts_pre:
                arr = t + d
                if 0 <= arr < T:
                    segs.append([(t, yi(i)), (arr, yh(j))])
                    cols_.append(c)
    if segs:
        ax.add_collection(LineCollection(segs, colors=cols_, linewidths=0.5, zorder=2))

    # Input spikes
    for i in range(n_in):
        ts = np.where(s_in[:, i] > 0)[0]
        if len(ts):
            ax.scatter(ts, np.full(len(ts), yi(i)),
                       s=16, color="#222222", marker="|", linewidths=1.0, zorder=4)

    # (B) Arrival spans + stars
    for j in range(h1):
        t_fires = np.where(s_h1[:, j] > 0)[0]
        if len(t_fires) == 0: continue
        for t_fire in t_fires:
            arrivals = []
            for i in range(n_in):
                for t in np.where(s_in[:, i] > 0)[0]:
                    arr = t + float(D_ih[i, j])
                    if (t_fire - 10.0) <= arr <= t_fire and abs(float(W_ih[i, j])) > W_THRESH:
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

    # (C) Hidden → Readout (only for linear readout)
    W_ro = w_dict.get("readout")
    n_ro = 0
    if W_ro is not None:
        ro_w = W_ro[0]; max_w = np.max(np.abs(ro_w)) + 1e-8
        for j in range(h1):
            w_ro = float(ro_w[j])
            if abs(w_ro) <= RO_THRESH: continue
            t_fires_ro = np.where(s_h1[win_len:, j] > 0)[0] + win_len
            if len(t_fires_ro) == 0: continue
            col = "crimson" if w_ro > 0 else "mediumblue"
            mag = min(abs(w_ro) / max_w, 1.0)
            lw  = 3.0 + 4.0 * mag
            for t_fire in t_fires_ro:
                n_ro += 1
                ax.plot([t_fire, t_fire], [0, y_read],
                        color=col, lw=lw, alpha=0.85, zorder=3)
                ax.scatter(t_fire, y_read, s=100, color=col, marker="^",
                           alpha=1.0, zorder=13, linewidths=0)

    # Readout bar
    ax.axhline(y_read, xmin=win_len/T, xmax=1.0,
               color="tomato", lw=2.0, alpha=0.6, zorder=2)
    ax.text(win_len + (T - win_len)/2, y_read + 0.3,
            "Readout", ha="center", va="bottom", fontsize=7, color="tomato")

    # Labels
    for lbl, yl in [("Input", y0_in + n_in/2), ("Hidden", y0_h1 + h1/2)]:
        ax.text(-1.5, yl, lbl, ha="right", va="center",
                fontsize=8, fontweight="bold", color="steelblue")

    ax.set_xlim(-2, T + 0.5)
    ax.set_ylim(y0_in - 1.0, y_read + 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Time (ms)", fontsize=8)
    ro_note = f"  ({n_ro} vote lines)" if W_ro is not None else "  (MLP readout)"
    ax.set_title(label + ro_note, fontsize=8, fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════════
# BATCH JOBS
# ═══════════════════════════════════════════════════════════════════════════

def make_raster_2panel(run_wad, run_d0, save_name, suptitle):
    print(f"  Raster: {save_name}")
    cfg_w, m_w = load_model(run_wad)
    cfg_d, m_d = load_model(run_d0)
    tr_w, _, _ = get_traces(m_w, cfg_w)
    tr_d, _, _ = get_traces(m_d, cfg_d)
    K = cfg_w["K"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(suptitle, fontsize=10)
    draw_raster_panel(axes[0], tr_w, "w+d  (trainable delays)")
    draw_raster_panel(axes[1], tr_d, "d=0  (no delays)")
    for ax in axes:
        ax.legend(handles=legend_patches(K), fontsize=6.5, loc="upper right",
                  framealpha=0.85, ncol=1)
    fig.tight_layout()
    fig.savefig(f"{OUT_BASE}/raster/{save_name}", dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_raster_Npanel(run_list, labels, save_name, suptitle):
    """N-panel raster (single runs, e.g. resource sweep)."""
    print(f"  Raster: {save_name}")
    N = len(run_list)
    fig, axes = plt.subplots(1, N, figsize=(8 * N, 8))
    if N == 1: axes = [axes]
    fig.suptitle(suptitle, fontsize=10)
    K_ref = None
    for ax, run_dir, lbl in zip(axes, run_list, labels):
        cfg, model = load_model(run_dir)
        tr, _, _ = get_traces(model, cfg)
        K_ref = cfg["K"]
        draw_raster_panel(ax, tr, lbl)
    for ax in axes:
        ax.legend(handles=legend_patches(K_ref or 1), fontsize=6.5,
                  loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(f"{OUT_BASE}/raster/{save_name}", dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_flow_Npanel(run_list, labels, save_name, suptitle):
    """N-panel spike flow."""
    print(f"  Flow:   {save_name}")
    N = len(run_list)
    # Each flow panel needs its own height ratio based on n_in + h1
    fig, axes = plt.subplots(1, N, figsize=(13 * N, 22))
    if N == 1: axes = [axes]
    fig.suptitle(suptitle, fontsize=10)
    for ax, run_dir, lbl in zip(axes, run_list, labels):
        cfg, model = load_model(run_dir)
        tr, w_dict, d_dict = get_traces(model, cfg)
        draw_flow_panel(ax, tr, w_dict, d_dict, lbl)
    fig.tight_layout()
    fig.savefig(f"{OUT_BASE}/flow/{save_name}", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    BASE3 = "runs/step3_planD_4ops_16k"
    BASE3H = "runs/step3_planD_4ops_16k_h100"
    BASE2D = "runs/step2_planD"
    SWEEP = "runs/planD_h_sweep"
    DEPTH = "runs/depth_ablation"

    # ── RASTERS ──────────────────────────────────────────────────────────
    print("\n=== RASTERS ===")

    # (1) Main science: 4-op 16k, K=1,2,3  wad vs d0
    for K in [1, 2, 3]:
        make_raster_2panel(
            f"{BASE3}/w_and_d_K{K}_seed42",
            f"{BASE3}/d0_control_K{K}_seed42",
            f"4ops16k_K{K}_wad_vs_d0.png",
            f"4-op 16k  |  K={K}  |  h=50 MLP  —  wad vs d=0")

    # (2) Resource sweep: NAND h=10,20,30,50 at K=3, wad side-by-side
    make_raster_Npanel(
        [f"{SWEEP}/planD_sweep_NAND_w_and_d_h{h}_K3_sw10_seed42"
         for h in [10, 20, 30, 50]],
        [f"wad  h={h}" for h in [10, 20, 30, 50]],
        "h_sweep_K3_wad_h10_20_30_50.png",
        "Resource sweep  |  NAND K=3  |  wad  |  h=10→50  (neurons sorted by 1st spike)")

    # (3) Resource sweep: wad vs d0 at K=3, h=10 and h=50
    for h in [10, 50]:
        make_raster_2panel(
            f"{SWEEP}/planD_sweep_NAND_w_and_d_h{h}_K3_sw10_seed42",
            f"{SWEEP}/planD_sweep_NAND_d0_h{h}_K3_sw10_seed42",
            f"h_sweep_K3_h{h}_wad_vs_d0.png",
            f"NAND  K=3  h={h}  —  wad vs d=0")

    # (4) h=50 vs h=100 capacity comparison at K=3
    make_raster_Npanel(
        [f"{BASE3}/w_and_d_K3_seed42",
         f"{BASE3H}/w_and_d_K3_seed42"],
        ["wad  h=50  (K@90%=3)", "wad  h=100  (K@90%=3)"],
        "h50_vs_h100_K3_wad.png",
        "Capacity comparison  |  NAND K=3  wad  |  h=50 vs h=100")

    # (5) Depth ablation: L1-h50 vs L2-h25h25 at K=3, linear readout, wad
    make_raster_2panel(
        f"{DEPTH}/L1-h50-linear_K3_seed42",
        f"{DEPTH}/L2-h25h25-linear_K3_seed42",
        "depth_K3_L1_vs_L2_linear.png",
        "Architecture  |  NAND K=3  linear readout  —  L1-h50 vs L2-h25h25")

    # (6) NAND single-op step2 linear readout: K=1,2,3 wad vs d0 (h=20 has d0 pairs)
    for K in [1, 2, 3]:
        wad_run = f"{BASE2D}/step2_seq_NAND_w_and_d_continuous_h20_K{K}_sw10_seed42"
        d0_run  = f"{BASE2D}/step2_seq_NAND_weights_only_d0_h20_K{K}_sw10_seed42"
        if os.path.exists(f"{d0_run}/best_model.pt"):
            make_raster_2panel(
                wad_run, d0_run,
                f"planD_NAND_h20_K{K}_wad_vs_d0.png",
                f"NAND single-op  |  K={K}  h=20  linear  —  wad vs d=0")

    # ── SPIKE FLOWS ──────────────────────────────────────────────────────
    print("\n=== SPIKE FLOWS ===")

    # (7) step2 NAND linear readout h=50: K=1, K=2, K=3 wad (vote lines visible)
    for K in [1, 2, 3]:
        run = f"{BASE2D}/step2_seq_NAND_w_and_d_continuous_h50_K{K}_sw10_seed42"
        if os.path.exists(f"{run}/best_model.pt"):
            make_flow_Npanel(
                [run],
                [f"NAND h50 K={K} wad (linear readout)"],
                f"planD_h50_K{K}_wad_linear.png",
                f"Enhanced Spike Flow  |  NAND h=50 K={K}  wad  (linear readout + vote lines)")

    # (8) step2 NAND h=50 K=3: wad vs d0 side-by-side (linear readout)
    make_flow_Npanel(
        [f"{BASE2D}/step2_seq_NAND_w_and_d_continuous_h50_K3_sw10_seed42",
         f"{BASE2D}/step2_seq_NAND_weights_only_d0_h20_K3_sw20_seed42"],
        ["NAND h50 K=3 wad (linear)", "NAND h20 K=3 d=0 (linear)"],
        "planD_K3_wad_vs_d0_linear.png",
        "Enhanced Spike Flow  |  NAND K=3  linear readout  —  wad vs d=0")

    # (9) Resource sweep flow: h=10 vs h=50 at K=3 wad (MLP)
    make_flow_Npanel(
        [f"{SWEEP}/planD_sweep_NAND_w_and_d_h10_K3_sw10_seed42",
         f"{SWEEP}/planD_sweep_NAND_w_and_d_h50_K3_sw10_seed42"],
        ["wad  h=10  K=3 (sparse)", "wad  h=50  K=3 (full)"],
        "h10_vs_h50_K3_wad_flow.png",
        "Resource contrast  |  NAND K=3 wad  —  h=10 vs h=50  (MLP readout)")

    # (10) 4-ops 16k K=3 wad vs d0 flow (MLP readout, no vote lines)
    make_flow_Npanel(
        [f"{BASE3}/w_and_d_K3_seed42",
         f"{BASE3}/d0_control_K3_seed42"],
        ["4ops-16k K=3 wad (MLP)", "4ops-16k K=3 d=0 (MLP)"],
        "4ops16k_K3_wad_vs_d0_flow.png",
        "Enhanced Spike Flow  |  Mixed 4-op 16k K=3  —  wad vs d=0  (MLP readout)")

    print(f"\nAll done.  Output: {OUT_BASE}/")

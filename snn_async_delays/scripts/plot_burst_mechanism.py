"""
Burst-encoding delay-routing MECHANISM figure (L5 master plot).

Rebuilds a *focused* input->hidden->output spike raster from the burst
visualisation runs' diagnostic_data.npz, emphasising the ONE thing that
matters: the hidden->output delay d_ho shifts each hidden spike's ARRIVAL
time into the readout window [win_len, T). With trainable delays (wad) the
arrivals land inside the window -> the output LIF integrates them -> it
fires (correct). With d=0 the hidden layer is silent -> output silent ->
chance accuracy. Side-by-side wad | d0 makes the contrast a single glance.

Unlike layer_to_layer_spike_flow (dense weight-coloured fan lines that bury
the message), this plot draws only firing spikes and the delay arcs that
carry them, so the routing is legible.

Run (from snn_async_delays/):
    python -m scripts.plot_burst_mechanism
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(BASE, "runs", "vis_burst_spkout")
OUT = os.path.join(BASE, "runs", "vis_burst_spkout", "mechanism_figure")
os.makedirs(OUT, exist_ok=True)

# lane geometry (shared by both panels)
Y_IN = (0.0, 1.2)        # 2 input channels
Y_HID = (2.2, 8.2)       # 50 hidden neurons
Y_OUT = 9.4              # output


def _npz(cond, K):
    p = os.path.join(RUNS, f"vis_burst_spkout_{cond}_K{K}_seed42",
                     "plots", "diagnostic_data.npz")
    if not os.path.exists(p):
        # K2 runs use a doubled suffix
        p = os.path.join(RUNS, f"vis_burst_spkout_{cond}_K{K}_K{K}_seed42",
                         "plots", "diagnostic_data.npz")
    return np.load(p)


def _hid_y(j, n_hid):
    lo, hi = Y_HID
    return lo + (hi - lo) * (j / max(1, n_hid - 1))


def draw_panel(ax, d, title, acc, is_delay):
    win = int(d["traces__win_len"])
    T = int(d["traces__T"])
    inp = d["traces__input_spikes"]
    hid = d["traces__hidden1_spikes"]
    out = d["traces__output_spikes"]
    d_ho = d["delays__ho"][:, 0]
    w_ho = d["weights__ho"][:, 0]
    n_hid = hid.shape[1]

    sub = int(d["traces__sub_win"])
    K = int(d["traces__K"])

    # ── shading: input window vs readout window ─────────────────────────────
    ax.axvspan(0, win, color="#e8eef7", alpha=0.7, zorder=0)
    ax.axvspan(win, T, color="#e7f4ea", alpha=0.9, zorder=0)
    ax.axvline(win, color="#2e7d32", ls="--", lw=1.4, zorder=1)
    ax.text(win / 2, Y_OUT + 0.7, "input window", ha="center", fontsize=8,
            color="#4666a0")
    ax.text((win + T) / 2, Y_OUT + 0.7, "readout window", ha="center",
            fontsize=8, color="#2e7d32", fontweight="bold")

    # ── K temporal sub-window (slot) dividers within the input window ───────
    if K > 1:
        for k in range(1, K):
            ax.axvline(k * sub, color="#4666a0", ls=":", lw=1.0,
                       alpha=0.6, zorder=1)
        for k in range(K):
            ax.text((k + 0.5) * sub, -0.55, f"slot {k}\n(query {k})",
                    ha="center", va="top", fontsize=7.5, color="#4666a0")

    # ── input burst spikes ──────────────────────────────────────────────────
    ti, ci = np.where(inp > 0)
    for t, c in zip(ti, ci):
        y = Y_IN[0] + (Y_IN[1] - Y_IN[0]) * (c / max(1, inp.shape[1] - 1))
        ax.plot([t], [y], marker="|", ms=16, mew=2.4, color="black", zorder=5)

    # ── hidden spikes + delay-routing arcs to output arrival ────────────────
    th, jh = np.where(hid > 0)
    wmax = np.abs(w_ho).max() + 1e-9
    landed = 0
    for t, j in zip(th, jh):
        yh = _hid_y(j, n_hid)
        ax.plot([t], [yh], "o", ms=6, color="#1f4e9c", zorder=5)
        arrival = t + d_ho[j]
        color = "#c0392b" if w_ho[j] >= 0 else "#2c5fa8"
        alpha = 0.25 + 0.65 * (abs(w_ho[j]) / wmax)
        arc = FancyArrowPatch(
            (t, yh), (arrival, Y_OUT),
            connectionstyle="arc3,rad=-0.18",
            arrowstyle="-|>", mutation_scale=10,
            lw=1.0 + 1.8 * (abs(w_ho[j]) / wmax),
            color=color, alpha=alpha, zorder=4)
        ax.add_patch(arc)
        # arrival tick on the output lane
        in_win = win <= arrival < T
        ax.plot([arrival], [Y_OUT], marker="v", ms=7,
                mfc=("#c0392b" if in_win else "none"),
                mec=color, mew=1.3, zorder=6)
        landed += int(in_win)

    # ── output spikes ───────────────────────────────────────────────────────
    to, _ = np.where(out > 0)
    for t in to:
        ax.plot([t], [Y_OUT], marker="*", ms=20, color="#f1c40f",
                mec="#b8860b", mew=1.0, zorder=7)

    # ── lane labels ─────────────────────────────────────────────────────────
    ax.text(-2.4, np.mean(Y_IN), "Input\n(burst)", ha="right", va="center",
            fontsize=10, color="black")
    ax.text(-2.4, np.mean(Y_HID), "Hidden\n(50 LIF)", ha="right", va="center",
            fontsize=10, color="#1f4e9c")
    ax.text(-2.4, Y_OUT, "Output", ha="right", va="center",
            fontsize=10, color="#b8860b")

    if not is_delay:
        ax.text((win + T) / 2, np.mean(Y_HID),
                "d = 0\nhidden silent\n→ output silent",
                ha="center", va="center", fontsize=13, color="#c0392b",
                fontweight="bold", alpha=0.85)
    else:
        ax.text(T - 0.3, np.mean(Y_HID) + 1.5,
                f"{landed} spikes\nrouted into\nreadout window",
                ha="right", va="center", fontsize=9, color="#2e7d32")

    ax.set_xlim(-4.5, T + 0.5)
    ax.set_ylim(-1.7 if K > 1 else -0.8, Y_OUT + 1.4)
    ax.set_yticks([])
    ax.set_xlabel("time (ms)")
    tag = "trainable delays" if is_delay else "no delay (d=0 control)"
    ax.set_title(f"{title}\n{tag}  |  test acc = {acc}", fontsize=11)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def build(K, accs):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    draw_panel(axes[0], _npz("wad", K), f"WITH DELAYS (K={K})",
               accs["wad"], True)
    draw_panel(axes[1], _npz("d0", K), f"WITHOUT DELAYS (K={K})",
               accs["d0"], False)

    # shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="|", ls="", color="black", ms=12, mew=2,
               label="input burst spike"),
        Line2D([], [], marker="o", ls="", color="#1f4e9c", label="hidden spike"),
        Line2D([], [], marker="v", ls="", mfc="#c0392b", mec="#c0392b",
               label="arrival inside readout window"),
        Line2D([], [], marker="*", ls="", color="#f1c40f", mec="#b8860b",
               ms=15, label="output spike"),
        Line2D([], [], color="#c0392b", label="excitatory routing (d_ho)"),
        Line2D([], [], color="#2c5fa8", label="inhibitory routing"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Delay routing mechanism (burst encoding, NAND, Plan D): "
        "d_ho shifts hidden-spike ARRIVAL into the readout window",
        fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    p = os.path.join(OUT, f"burst_mechanism_K{K}.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


# ════════════════════════════════════════════════════════════════════════════
# Generalized PER-RUN mechanism figure (works for any Plan D run, spiking-output
# OR MLP-readout, rate OR burst). Reads the run's own diagnostic_data.npz.
#
#   - spiking-output run (has delays__ho + output spikes): d_ho routes each
#     HIDDEN spike's arrival into the readout window (hidden -> output).
#   - MLP-readout run (only delays__ih): d_ih routes each INPUT spike's arrival
#     into the readout window (input -> hidden); hidden spikes overlaid. This is
#     the correct analog when there is no output spike layer — the routing that
#     matters happens one layer earlier, and is visible even when the (often
#     sparse, esp. burst @ low h) hidden layer barely fires.
# ════════════════════════════════════════════════════════════════════════════
def plot_run_mechanism(run_dir, save_path=None):
    """Standalone per-run mechanism figure. Thin wrapper around the shared core
    utils.viz.draw_mechanism_on_ax (same logic as the diagnostic-panel bottom-right)."""
    from utils.viz import draw_mechanism_on_ax
    npz_path = os.path.join(run_dir, "plots", "diagnostic_data.npz")
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path, allow_pickle=True)
    traces  = {k[len("traces__"):]:  d[k] for k in d.files if k.startswith("traces__")}
    weights = {k[len("weights__"):]: d[k] for k in d.files if k.startswith("weights__")}
    delays  = {k[len("delays__"):]:  d[k] for k in d.files if k.startswith("delays__")}

    fig, ax = plt.subplots(figsize=(11, 6))
    landed = draw_mechanism_on_ax(ax, traces, weights, delays, title=False)

    acc = None
    ev = os.path.join(run_dir, "eval_results.json")
    if os.path.exists(ev):
        import json
        acc = json.load(open(ev, encoding="utf-8")).get("accuracy")
    name = os.path.basename(run_dir.rstrip("/\\"))
    sub = f"acc={acc*100:.1f}%  " if acc is not None else ""
    logits = traces.get("output_logits")
    if logits is not None:
        sub += f"logits={np.round(np.asarray(logits), 2).tolist()}  "
    sub += f"| {landed} arrivals in readout window"
    ax.set_title(f"Delay routing mechanism — {name}\n{sub}", fontsize=10)

    if save_path is None:
        save_path = os.path.join(run_dir, "plots", "mechanism_sample0.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None,
                    help="Render the per-run mechanism figure for one run and exit.")
    ap.add_argument("--runs_dir", default=None,
                    help="Render mechanism_sample0.png for every run under this folder.")
    args = ap.parse_args()

    if args.run_dir:
        p = plot_run_mechanism(args.run_dir)
        print(f"wrote {p}" if p else "  [skip] no diagnostic_data.npz in run")
        return
    if args.runs_dir:
        import glob
        n = 0
        for npz in glob.glob(os.path.join(args.runs_dir, "*", "plots", "diagnostic_data.npz")):
            rd = os.path.dirname(os.path.dirname(npz))
            if plot_run_mechanism(rd):
                n += 1
        print(f"wrote mechanism figures for {n} run(s) under {args.runs_dir}")
        return

    # default: the L5 master figures (spiking-output vis runs)
    build(1, {"wad": "100%", "d0": "26.4%"})
    build(2, {"wad": "100%", "d0": "23.8%"})
    print(f"\nDone. Outputs in: {OUT}")


if __name__ == "__main__":
    main()

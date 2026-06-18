"""
Max K@90% vs Total Hidden Neurons — temporal multiplexing scaling curve.

Data sources (CLAUDE.md / EXPERIMENT_LOG):
  Step 2  Plan D, single-op NAND, h=20 (linear readout)   → Section 11
  Step 2  Plan D, single-op NAND, h=50 (linear / MLP)     → Sections 12–13
  Depth ablation, single-op NAND, L2-h25h25, MLP           → Section 14
  Step 3  Direction A,  mixed 4-op, h=50,  L1, MLP         → Section 19
  Step 3  Direction C,  mixed 4-op, h=100, L1, MLP         → Section 20
  Step 3  Direction D,  mixed 4-op, h=100, L2 50+50, MLP   → Section 21

Run from snn_async_delays/:
    python -m scripts.plot_scaling_curve
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

# ── Experiment data ──────────────────────────────────────────────────────────
# Each entry: (total_hidden_neurons, max_k_at_90pct)

# trainable delays + L1 (single layer), single-op NAND
# h=20: linear readout, Max K@90%=2  (Section 11)
# h=50: MLP readout,    Max K@90%=3  (Section 13)
nand_l1 = [(20, 2), (50, 3)]

# trainable delays + L2 (two layers), single-op NAND, MLP
# total h = 25+25 = 50, Max K@90%=2   (Section 14)
nand_l2 = [(50, 2)]

# trainable delays + L1, mixed 4-op, n_train=16000, MLP
# h=50:  Max K@90%=2  (Section 19, Direction A)
# h=100: Max K@90%=3  (Section 20, Direction C)
mixed_l1 = [(50, 2), (100, 3)]

# trainable delays + L2 50+50, mixed 4-op, n_train=16000, MLP
# total h = 50+50 = 100, Max K@90%=2  (Section 21, Direction D)
mixed_l2 = [(100, 2)]

# d=0 baseline (weights only, fixed delay=0)
# both tasks, any readout → Max K@90%=0 at all h
d0 = [(20, 0), (50, 0), (100, 0)]

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150,
})

C_NAND  = "#3A7FC1"   # blue  — single-op NAND
C_MIXED = "#E07B39"   # orange — mixed 4-op
C_D0    = "#AAAAAA"   # grey  — d=0 baseline

def unzip(pts):
    xs, ys = zip(*pts)
    return list(xs), list(ys)

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.8, 4.6))

# ── d=0 baseline (flat) ───────────────────────────────────────────────────
x_d0, y_d0 = unzip(d0)
ax.plot(x_d0, y_d0, color=C_D0, linewidth=1.8, linestyle="--",
        zorder=1, label="_nolegend_")
ax.fill_between([15, 110], -0.15, 0.15,
                color=C_D0, alpha=0.08, zorder=0)
ax.text(108, 0.0, "d = 0\n(sync)", fontsize=9, color=C_D0,
        va="center", ha="left")

# ── Single-op NAND L1 ────────────────────────────────────────────────────
x, y = unzip(nand_l1)
ax.plot(x, y, color=C_NAND, linewidth=2.2, marker="o", markersize=8,
        zorder=4, label="_nolegend_")

# ── Single-op NAND L2 (depth, same total budget as h=50) ─────────────────
x2, y2 = unzip(nand_l2)
ax.plot(x2, y2, color=C_NAND, linewidth=0,
        marker="s", markersize=9, markerfacecolor="white",
        markeredgecolor=C_NAND, markeredgewidth=2.0,
        zorder=4, label="_nolegend_")
# dotted connector from (50,3) → (50,2) to show "same budget, worse"
ax.annotate("", xy=(50, 2.05), xytext=(50, 2.9),
            arrowprops=dict(arrowstyle="-", color=C_NAND,
                            linestyle="dotted", lw=1.4))

# ── Mixed 4-op L1 ────────────────────────────────────────────────────────
xm, ym = unzip(mixed_l1)
ax.plot(xm, ym, color=C_MIXED, linewidth=2.2, marker="o", markersize=8,
        zorder=4, label="_nolegend_")

# ── Mixed 4-op L2 (depth, same total budget as h=100) ────────────────────
xm2, ym2 = unzip(mixed_l2)
ax.plot(xm2, ym2, color=C_MIXED, linewidth=0,
        marker="s", markersize=9, markerfacecolor="white",
        markeredgecolor=C_MIXED, markeredgewidth=2.0,
        zorder=4, label="_nolegend_")
ax.annotate("", xy=(100, 2.05), xytext=(100, 2.9),
            arrowprops=dict(arrowstyle="-", color=C_MIXED,
                            linestyle="dotted", lw=1.4))

# ── Point annotations ─────────────────────────────────────────────────────
ann_kw = dict(fontsize=8.5, ha="center", va="bottom")

ax.annotate("K=2\n(linear readout)", xy=(20, 2), xytext=(20, 2.10),
            color=C_NAND, fontsize=8.5, ha="center", va="bottom")
# K=3 NAND — place to the right to avoid legend collision
ax.annotate("K=3 (MLP)", xy=(50, 3), xytext=(62, 3.10),
            color=C_NAND, fontsize=8.5, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", color=C_NAND, lw=0.8))
ax.annotate("K=2 (L2: 25+25)", xy=(48, 2), xytext=(33, 1.6),
            color=C_NAND, fontsize=8.5, ha="center", va="top",
            arrowprops=dict(arrowstyle="-", color=C_NAND, lw=0.8))

ax.annotate("K=2 (MLP)", xy=(50, 2), xytext=(38, 2.15),
            color=C_MIXED, fontsize=8.5, ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color=C_MIXED, lw=0.8))
ax.annotate("K=3 (MLP)", xy=(100, 3), xytext=(100, 3.10),
            color=C_MIXED, fontsize=8.5, ha="center", va="bottom")
ax.annotate("K=2 (L2: 50+50)", xy=(102, 2), xytext=(108, 1.65),
            color=C_MIXED, fontsize=8.5, ha="left", va="top",
            arrowprops=dict(arrowstyle="-", color=C_MIXED, lw=0.8))

# ── Shaded "asynchronous advantage" region ────────────────────────────────
# Between d=0 (y=0) and the async L1 lines
nand_interp_x = np.linspace(20, 50, 50)
nand_interp_y = np.interp(nand_interp_x, [20, 50], [2, 3])
mixed_interp_x = np.linspace(50, 100, 50)
mixed_interp_y = np.interp(mixed_interp_x, [50, 100], [2, 3])

ax.fill_between(nand_interp_x,  0, nand_interp_y,
                color=C_NAND, alpha=0.08, zorder=0)
ax.fill_between(mixed_interp_x, 0, mixed_interp_y,
                color=C_MIXED, alpha=0.08, zorder=0)

# ── Legend ───────────────────────────────────────────────────────────────────
legend_elements = [
    mlines.Line2D([0], [0], color=C_NAND, linewidth=2.2,
                  marker="o", markersize=7,
                  label="Single-op NAND  (L1, w+d)"),
    mlines.Line2D([0], [0], color=C_MIXED, linewidth=2.2,
                  marker="o", markersize=7,
                  label="Mixed 4-op  (L1, w+d, n=16k)"),
    mlines.Line2D([0], [0], color="#888888", linewidth=0,
                  marker="s", markersize=8,
                  markerfacecolor="white", markeredgecolor="#888888",
                  markeredgewidth=1.8,
                  label="2-layer split  (L2, same total budget)"),
    mlines.Line2D([0], [0], color=C_D0, linewidth=1.8, linestyle="--",
                  label="d = 0  (weights only, any task)"),
]
ax.legend(handles=legend_elements, fontsize=8.8,
          framealpha=0.0, loc="upper left",
          handlelength=2.2, handletextpad=0.6)

# ── Axes cosmetics ────────────────────────────────────────────────────────
ax.set_xlabel("Total hidden neurons  (neuron budget  h)", fontsize=12)
ax.set_ylabel("Max K  @ 90% accuracy", fontsize=12)
ax.set_title(
    "Temporal Multiplexing Capacity vs Neuron Budget\n"
    "Trainable synaptic delays scale K; weight-only baseline cannot multiplex",
    fontsize=11, pad=10,
)
ax.set_xticks([20, 50, 100])
ax.set_xticklabels(["20", "50", "100"])
ax.set_yticks([0, 1, 2, 3])
ax.set_xlim(12, 130)
ax.set_ylim(-0.4, 3.65)
ax.tick_params(axis="both", which="major", labelsize=10)

# horizontal reference lines
for k in [1, 2, 3]:
    ax.axhline(k, color="#E0E0E0", linewidth=0.6, zorder=0)

# ── Save ─────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "..", "docs", "scaling_curve.png")
out = os.path.normpath(out)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)

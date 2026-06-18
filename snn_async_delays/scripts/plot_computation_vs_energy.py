"""
# Computations vs Energy Budget — faithful to image3.

X : mean hidden spikes / trial  (energy budget, empirical)
Y : # computations  K  (number of queries executed per trial)

Filled markers : acc >= 90%  →  computation "succeeds"
Cross  markers : acc <  90%  →  computation "fails"

d=0 control spends energy but K always falls below 90% → K_eff = 0.
Trainable-delay network (w_and_d) achieves K=2 or K=3 at the same
energy level — the async advantage.

Data source: depth ablation, single-op NAND, Plan D
  (CLAUDE.md Sections 13-14, K/spk table + accuracy table)
  spikes = K / K_per_spk
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
import numpy as np

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

THRESHOLD = 0.90
C_ASYNC   = "#3A7FC1"   # blue  — trainable delays
C_D0      = "#AAAAAA"   # grey  — d=0 baseline

# ── Data (depth ablation, single-op NAND, h=50 budget) ───────────────────────
# (label, mode, K/spk, K, accuracy)
# spikes = K / K_per_spk
raw = [
    # ── L1-h50 + linear, w_and_d ──
    ("L1+linear", "async", 0.144, 2, 0.924),
    ("L1+linear", "async", 0.142, 3, 0.873),
    ("L1+linear", "async", 0.168, 4, 0.849),
    # ── L1-h50 + MLP, w_and_d ──
    ("L1+MLP",    "async", 0.100, 2, 0.959),
    ("L1+MLP",    "async", 0.110, 3, 0.927),
    ("L1+MLP",    "async", 0.100, 4, 0.899),
    # ── L2-h25h25 + MLP, w_and_d ──
    ("L2+MLP",    "async", 0.073, 2, 0.916),
    ("L2+MLP",    "async", 0.084, 3, 0.883),
    ("L2+MLP",    "async", 0.083, 4, 0.865),
    # ── L2-h25h25 + d=0 ──
    ("d=0",       "d0",    0.107, 2, 0.777),
    ("d=0",       "d0",    0.122, 3, 0.774),
    ("d=0",       "d0",    0.122, 4, 0.769),
]

records = [(lbl, mode, K / kps, K, acc) for lbl, mode, kps, K, acc in raw]

# Separate async vs d0, and success vs fail
async_ok   = [(s, k) for _, m, s, k, a in records if m == "async" and a >= THRESHOLD]
async_fail = [(s, k) for _, m, s, k, a in records if m == "async" and a <  THRESHOLD]
d0_pts     = [(s, k) for _, m, s, k, a in records if m == "d0"]

# ── Build "async frontier": best K achievable at each energy level ────────────
# for each K value, take the minimum-energy model that reaches that K
from collections import defaultdict
best_per_k = {}
for s, k in async_ok:
    if k not in best_per_k or s < best_per_k[k]:
        best_per_k[k] = s
frontier = sorted(best_per_k.items())          # [(K, spikes)] sorted by K
frontier_k   = [k for k, _ in frontier]
frontier_spk = [s for _, s in frontier]

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 4.8))

# ── d=0 region: energy is spent, K=0 computations produced ──────────────────
d0_x = [s for s, _ in d0_pts]
# Draw downward arrows: "energy spent → 0 useful computations"
for s, k in d0_pts:
    ax.annotate("", xy=(s, 0.06), xytext=(s, k - 0.10),
                arrowprops=dict(arrowstyle="-|>", color=C_D0,
                                lw=1.2, alpha=0.55))
    ax.scatter([s], [k],  color=C_D0, marker="x", s=110,
               linewidths=1.8, zorder=5, alpha=0.6)
ax.scatter(d0_x, [0]*len(d0_x), color=C_D0, s=72, zorder=6,
           facecolors="white", edgecolors=C_D0, linewidths=1.6)
# flat K=0 "capacity" line for d=0
ax.axhline(0, color=C_D0, linewidth=1.8, linestyle="--",
           xmin=0.04, xmax=0.92, zorder=1, alpha=0.8)
ax.text(46, -0.22, "d = 0  (weights only)", fontsize=9,
        color=C_D0, ha="right", va="top", style="italic")

# ── Async: scatter all points ─────────────────────────────────────────────────
# failed attempts (hollow ×)
ax.scatter([s for s, _ in async_fail], [k for _, k in async_fail],
           marker="x", color=C_ASYNC, s=90, linewidths=1.8,
           zorder=4, alpha=0.40)
# successful computations (filled circle)
ax.scatter([s for s, _ in async_ok], [k for _, k in async_ok],
           color=C_ASYNC, s=100, zorder=6, edgecolors="white", linewidths=0.8)

# ── Frontier: connect best-case success points ────────────────────────────────
ax.plot(frontier_spk, frontier_k,
        color=C_ASYNC, linewidth=2.4, zorder=5,
        solid_capstyle="round")
# shade below frontier
fill_x = [0] + frontier_spk + [frontier_spk[-1] + 10]
fill_y = [0] + frontier_k   + [frontier_k[-1]]
ax.fill_between(fill_x, 0, fill_y,
                step="post", color=C_ASYNC, alpha=0.08, zorder=0)

# ── Key point annotations ─────────────────────────────────────────────────────
# L1-linear K=2 (leftmost success)
ax.annotate("L1+linear\nK=2", xy=(13.9, 2), xytext=(10.5, 2.28),
            color=C_ASYNC, fontsize=8.2, ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color=C_ASYNC, lw=0.8))
# L1-MLP K=2
ax.annotate("L1+MLP\nK=2", xy=(20.0, 2), xytext=(20.5, 2.28),
            color=C_ASYNC, fontsize=8.2, ha="left", va="bottom")
# L1-MLP K=3  ← the headline result
ax.annotate("L1+MLP\nK = 3  ✓", xy=(27.3, 3), xytext=(30.0, 3.18),
            color=C_ASYNC, fontsize=9.0, ha="left", va="bottom", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C_ASYNC, lw=0.9))
# L2-MLP K=2 (same energy as L1-MLP K=3, but only K=2)
ax.annotate("L2+MLP (25+25)\nK=2  ← same energy,\nless capacity",
            xy=(27.4, 2), xytext=(30.5, 1.45),
            color=C_ASYNC, fontsize=8.0, ha="left", va="top", alpha=0.8,
            arrowprops=dict(arrowstyle="-", color=C_ASYNC, lw=0.7, alpha=0.7))

# "same budget" bracket at ~27 spikes
ax.annotate("", xy=(27.3, 3.0), xytext=(27.3, 0.0),
            arrowprops=dict(arrowstyle="<->", color="#CCCCCC",
                            lw=1.0, linestyle="dotted"))
ax.text(28.0, 1.5, "same\nbudget\n~27 spk", fontsize=7.5,
        color="#999999", ha="left", va="center")

# ── Legend ───────────────────────────────────────────────────────────────────
legend_elements = [
    mlines.Line2D([0], [0], color=C_ASYNC, marker="o", markersize=7,
                  linewidth=2.2, label="Trainable delays (w+d)  — ✓ success"),
    mlines.Line2D([0], [0], color=C_ASYNC, marker="x", markersize=8,
                  linewidth=0, markeredgewidth=1.8, alpha=0.5,
                  label="Trainable delays (w+d)  — ✗ below 90%"),
    mlines.Line2D([0], [0], color=C_D0, marker="x", markersize=8,
                  linewidth=1.8, linestyle="--", markeredgewidth=1.8, alpha=0.7,
                  label="d = 0  (weights only)  — always fails"),
]
ax.legend(handles=legend_elements, fontsize=8.5,
          framealpha=0.0, loc="upper left",
          handlelength=2.0, handletextpad=0.5)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel("Energy budget  (mean hidden spikes / trial)", fontsize=12)
ax.set_ylabel("# Computations  K  (queries / trial)", fontsize=12)
ax.set_title(
    "Async delays convert energy into more computations; d=0 (sync) cannot\n"
    "(Single-op NAND, Plan D, h = 50 neurons  ·  success = acc ≥ 90%)",
    fontsize=10.5, pad=9,
)
ax.set_xticks([10, 20, 30, 40, 50])
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_xlim(4, 50)
ax.set_ylim(-0.6, 4.2)
ax.tick_params(axis="both", labelsize=10)

# light horizontal guides
for k in [1, 2, 3, 4]:
    ax.axhline(k, color="#EFEFEF", linewidth=0.6, zorder=0)

plt.tight_layout()

out = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "docs", "computation_vs_energy.png")
)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)

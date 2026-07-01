"""
# Computations vs Energy Budget (v2) -- proper sync baseline.

Sync baseline construction:
  d=0 in Plan D (T=20ms for K=1) NEVER reaches 90% accuracy at any
  tested h (max ~84.7% at h=30). This is because the 10ms input window
  is too short for rate-coded spikes to accumulate without delay alignment.
  Therefore the sync reference uses Step-1 weights_only (h=50, T=50ms):
    - NAND accuracy: ~93%  (within 90% threshold)
    - Energy: 34.1 spikes / trial
  Cost of K sync computations = K * 34.1 spikes (K independent trials).

Async data: planD_h_sweep w_and_d h=50, K=1..5 (directly measured).
"""

import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import defaultdict

# ── Load planD_h_sweep ────────────────────────────────────────────────────────
RUNS = "D:/xukun/Documents/IC/SNN/SNN_project/snn_async_delays/runs"

def load_runs(pattern):
    out = []
    for path in glob.glob(os.path.join(RUNS, pattern, "eval_results.json")):
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return out

sweep = load_runs("NAND_neuron_sweep_(planD)/*")

def group_mean(records, key_fields, val_fields):
    buckets = defaultdict(list)
    for r in records:
        key = tuple(r.get(f) for f in key_fields)
        buckets[key].append(r)
    result = []
    for key, recs in sorted(buckets.items()):
        row = dict(zip(key_fields, key))
        for v in val_fields:
            vals = [r[v] for r in recs if r.get(v) is not None]
            row[v] = float(np.mean(vals)) if vals else None
        result.append(row)
    return result

THRESH = 0.90

# ── Async: w_and_d, h=50, K=1..5 ────────────────────────────────────────────
wad = [r for r in sweep
       if r.get("condition") == "w_and_d" and r.get("hidden_size") == 50]
wad_mean = group_mean(wad, ["K"], ["accuracy", "mean_hidden_spikes"])

async_pts = {}   # K -> (energy, acc)
for row in wad_mean:
    async_pts[row["K"]] = (row["mean_hidden_spikes"], row["accuracy"])

# ── Sync baseline: Step-1 weights_only h=50 (T=50ms, single query) ───────────
# NAND accuracy ~93% (above 90%), energy = 34.1 spk/trial  (from CLAUDE.md)
E_sync_unit = 34.1      # spikes per one sync computation

# ── Print summary ─────────────────────────────────────────────────────────────
print("Async (w_and_d, h=50, Plan D):")
for k in sorted(async_pts):
    e, a = async_pts[k]
    tag = "OK" if a >= THRESH else "xx"
    print(f"  K={k}  energy={e:.2f} spk  acc={a:.3f} [{tag}]")

print(f"\nSync ref (d=0, Step-1, h=50, T=50ms): {E_sync_unit} spk/query")
print("  d=0 in Plan D never reaches 90% (max ~84.7%) -> need longer window")
K_sync_range = [1, 2, 3, 4]
for k in K_sync_range:
    print(f"  K={k}: {k * E_sync_unit:.1f} spk  ({k} x {E_sync_unit} spk)")

# ── Build series ──────────────────────────────────────────────────────────────
async_K     = sorted(async_pts.keys())
async_E     = [async_pts[k][0] for k in async_K]
async_acc   = [async_pts[k][1] for k in async_K]
async_ok    = [(e, k) for k, e, a in zip(async_K, async_E, async_acc) if a >= THRESH]
async_fail  = [(e, k) for k, e, a in zip(async_K, async_E, async_acc) if a <  THRESH]

sync_K      = np.array([0] + K_sync_range, dtype=float)
sync_E      = sync_K * E_sync_unit

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "xtick.direction": "out", "ytick.direction": "out",
})
C_ASYNC = "#3A7FC1"
C_SYNC  = "#AAAAAA"

fig, ax = plt.subplots(figsize=(7.4, 5.2))

# ── Sync line (linear reference) ──────────────────────────────────────────────
ax.plot(sync_E, sync_K, color=C_SYNC, linewidth=2.0, linestyle="--",
        zorder=2)
ax.scatter([E_sync_unit * k for k in K_sync_range], K_sync_range,
           color=C_SYNC, s=75, zorder=4, edgecolors="white", linewidths=0.8)
ax.text(E_sync_unit * 4 + 2, 4.08,
        f"Sync (d=0)\nK x {E_sync_unit:.0f} spk/query",
        color=C_SYNC, fontsize=8.5, va="bottom", ha="left")

# ── Async: connect success points ─────────────────────────────────────────────
ok_E  = [e for e, k in async_ok]
ok_K  = [k for e, k in async_ok]
fai_E = [e for e, k in async_fail]
fai_K = [k for e, k in async_fail]

ax.plot(ok_E, ok_K, color=C_ASYNC, linewidth=2.4,
        solid_capstyle="round", zorder=3)
if ok_K and fai_K:   # dashed extension to first fail
    ax.plot([ok_E[-1], fai_E[0]], [ok_K[-1], fai_K[0]],
            color=C_ASYNC, linewidth=1.4, linestyle=":", zorder=3)

ax.scatter(ok_E,  ok_K,  color=C_ASYNC, s=100, zorder=5,
           edgecolors="white", linewidths=0.8)
ax.scatter(fai_E, fai_K, color=C_ASYNC, s=100, zorder=5,
           marker="x", linewidths=2.0, alpha=0.45)

# ── Shade async-advantage region ──────────────────────────────────────────────
overlap_K = [k for k in ok_K if 1 <= k <= 4]
if overlap_K:
    ae = [async_pts[k][0] for k in overlap_K]
    se = [E_sync_unit * k  for k in overlap_K]
    ax.fill_betweenx(overlap_K, ae, se,
                     color=C_ASYNC, alpha=0.09, zorder=0)

# ── Energy-saving annotations ─────────────────────────────────────────────────
for k in [2, 3]:
    if k in async_pts:
        e_a = async_pts[k][0]
        e_s = E_sync_unit * k
        pct = (e_s - e_a) / e_s * 100
        mid  = (e_a + e_s) / 2
        ax.annotate(f"-{pct:.0f}%",
                    xy=(mid, k), fontsize=9.5, fontweight="bold",
                    color=C_ASYNC, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=C_ASYNC, alpha=0.9, lw=0.8))

# ── Per-point labels (async) ──────────────────────────────────────────────────
label_offset = {1: (-2, 0.18), 2: (-2, 0.18), 3: (-2, 0.18),
                4: (2, 0.18),  5: (2, 0.18)}
for k in async_K:
    e, a = async_pts[k]
    tag  = "(OK)" if a >= THRESH else "(fail)"
    dx, dy = label_offset.get(k, (2, 0.18))
    ha = "right" if dx < 0 else "left"
    ax.annotate(f"K={k},  {a*100:.1f}% {tag}",
                xy=(e, k), xytext=(e + dx, k + dy),
                color=C_ASYNC, fontsize=8.0, ha=ha, va="bottom")

# ── Note: d=0 Plan-D failure ──────────────────────────────────────────────────
ax.text(0.98, 0.04,
        "Note: d=0 in Plan D never reaches 90% at K=1\n"
        "(max 84.7% at h=30) -- sync ref uses Step-1 (T=50ms)",
        transform=ax.transAxes, fontsize=7.5, color="#888888",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.6))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mlines.Line2D([0],[0], color=C_ASYNC, linewidth=2.2, marker="o",
                  markersize=7, label="Async (w+d, h=50, Plan D)  -- acc >= 90%"),
    mlines.Line2D([0],[0], color=C_ASYNC, linewidth=0, marker="x",
                  markersize=8, markeredgewidth=1.8, alpha=0.5,
                  label="Async -- acc < 90%  (capacity limit)"),
    mlines.Line2D([0],[0], color=C_SYNC, linewidth=2.0, linestyle="--",
                  marker="o", markersize=7,
                  label="Sync (d=0, Step-1 ref.) -- K independent trials"),
]
ax.legend(handles=legend_elements, fontsize=8.8, framealpha=0.0,
          loc="upper left", handlelength=2.2)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel("Energy  (total hidden spikes per trial)", fontsize=12)
ax.set_ylabel("# Computations  K  (queries per trial)", fontsize=12)
ax.set_title(
    "Async delays: same energy budget -> more computations\n"
    "(Single-op NAND, Plan D  |  h = 50 neurons)",
    fontsize=11, pad=9,
)
ax.set_yticks(range(0, 6))
ax.set_xlim(-2, 160)
ax.set_ylim(-0.5, 5.5)
ax.tick_params(axis="both", labelsize=10)
for k in range(1, 6):
    ax.axhline(k, color="#F0F0F0", linewidth=0.6, zorder=0)
ax.axvline(0, color="#CCCCCC", linewidth=0.6, zorder=0)

plt.tight_layout()
out = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "docs",
                 "computation_vs_energy_v2.png"))
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.close(fig)

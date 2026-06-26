"""
Schematic timing diagrams for the three multi-query trial designs compared in
Section "Multi-Query Design Comparison": SR (sequential-readout), DC
(dedicated-channel), SC (shared-channel). Pure illustration of trial structure
-- no model run, no data dependency.

Usage (from snn_async_delays/):
    python -m scripts.plot_design_schematic
"""
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

K = 3
SUBWIN = 1.0   # arbitrary time unit per sub-window
READ = 1.0     # readout window length (same unit)

INPUT_COLOR = "#4C72B0"
READ_COLOR = "#C44E52"
CHANNEL_COLOR = "#333333"


def draw_channel_lane(ax, y, x0, x1, active_spans, label):
    ax.plot([x0, x1], [y, y], color=CHANNEL_COLOR, lw=1.0, zorder=1)
    for (a, b, c) in active_spans:
        ax.add_patch(mpatches.Rectangle((a, y - 0.07), b - a, 0.14,
                                         color=c, zorder=2))
    ax.text(x0 - 0.12, y, label, ha="right", va="center", fontsize=7.5)


def draw_window_band(ax, x0, x1, y0, y1, color, label, alpha=0.12):
    ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     color=color, alpha=alpha, zorder=0))
    ax.text((x0 + x1) / 2, y1 + 0.06, label, ha="center", va="bottom",
            fontsize=7.5, color=color)


def panel_sr(ax):
    """Sequential-Readout: K blocks, each [input_k | readout_k], 2 shared channels."""
    n_ch = 2
    total = K * (SUBWIN + READ)
    top = 0.3 + n_ch * 0.4 + 0.15
    for k in range(K):
        x_in0 = k * (SUBWIN + READ)
        x_in1 = x_in0 + SUBWIN
        x_ro1 = x_in1 + READ
        draw_window_band(ax, x_in0, x_in1, 0.3, top, INPUT_COLOR, f"in$_{{{k}}}$")
        draw_window_band(ax, x_in1, x_ro1, 0.3, top, READ_COLOR, f"ro$_{{{k}}}$")
    for c in range(n_ch):
        y = 0.5 + c * 0.4
        spans = [(k * (SUBWIN + READ), k * (SUBWIN + READ) + SUBWIN, INPUT_COLOR)
                  for k in range(K)]
        draw_channel_lane(ax, y, 0, total, spans, f"ch{c}")
    ax.set_xlim(-0.7, total + 0.1)
    ax.set_ylim(0, top + 0.35)
    ax.set_title("SR -- Sequential-Readout\n2 shared ch., $K$ separate readout windows\n"
                 "trial length $\\propto K$", fontsize=9)
    ax.axis("off")


def panel_dc(ax):
    """Dedicated-Channel: 2K dedicated channels, simultaneous input, 1 shared readout."""
    n_ch = 2 * K
    total = SUBWIN + READ
    top = 0.3 + n_ch * 0.25 + 0.15
    draw_window_band(ax, 0, SUBWIN, 0.3, top, INPUT_COLOR, "in (all $K$ at once)")
    draw_window_band(ax, SUBWIN, total, 0.3, top, READ_COLOR, "ro (shared)")
    for c in range(n_ch):
        y = 0.5 + c * 0.25
        spans = [(0, SUBWIN, INPUT_COLOR)]
        label = f"Q{c // 2} ch{c % 2}"
        draw_channel_lane(ax, y, 0, total, spans, label)
    ax.set_xlim(-0.9, total + 0.1)
    ax.set_ylim(0, top + 0.35)
    ax.set_title("DC -- Dedicated-Channel\n$2K$ dedicated ch., 1 shared readout window\n"
                 "trial length fixed", fontsize=9)
    ax.axis("off")


def panel_sc(ax):
    """Shared-Channel: 2 shared channels, K sequential sub-windows, 1 shared readout."""
    n_ch = 2
    total = K * SUBWIN + READ
    top = 0.3 + n_ch * 0.4 + 0.15
    for k in range(K):
        x0, x1 = k * SUBWIN, (k + 1) * SUBWIN
        draw_window_band(ax, x0, x1, 0.3, top, INPUT_COLOR, f"in$_{{{k}}}$")
    draw_window_band(ax, K * SUBWIN, total, 0.3, top, READ_COLOR, "ro (shared)")
    for c in range(n_ch):
        y = 0.5 + c * 0.4
        spans = [(k * SUBWIN, (k + 1) * SUBWIN, INPUT_COLOR) for k in range(K)]
        draw_channel_lane(ax, y, 0, total, spans, f"ch{c}")
    ax.set_xlim(-0.7, total + 0.1)
    ax.set_ylim(0, top + 0.35)
    ax.set_title("SC -- Shared-Channel (central design)\n2 shared ch., $K$ sequential "
                 "sub-windows,\n1 shared readout window", fontsize=9, fontweight="bold")
    ax.axis("off")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    panel_sr(axes[0])
    panel_dc(axes[1])
    panel_sc(axes[2])
    fig.tight_layout()
    out_dir = "paper/figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig_design_schematics.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

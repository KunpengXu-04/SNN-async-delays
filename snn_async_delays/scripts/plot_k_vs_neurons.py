"""
Generate final K vs Neurons and Efficiency plots from h-sweep summary JSON.

Run this after run_plan_d_h_sweep.py has completed.

Usage (from snn_async_delays/):
    python -m scripts.plot_k_vs_neurons
    python -m scripts.plot_k_vs_neurons --summary runs/planD_h_sweep/planD_h_sweep_summary.json
    python -m scripts.plot_k_vs_neurons --tau 0.90 0.95
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.viz import (plot_K_vs_neurons, plot_neuron_efficiency,
                       plot_accuracy_heatmap, plot_accuracy_curves,
                       plot_capacity_clean)


def main():
    parser = argparse.ArgumentParser()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_summary = os.path.join(base, "runs", "planD_h_sweep",
                                   "planD_h_sweep_summary.json")
    parser.add_argument("--summary", default=default_summary)
    parser.add_argument("--tau",     type=float, nargs="+", default=[0.90, 0.95])
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        print(f"Summary not found: {args.summary}")
        print("Run scripts/run_plan_d_h_sweep.py first.")
        return

    with open(args.summary, encoding="utf-8") as f:
        summary = json.load(f)

    out_dir = args.out_dir or os.path.join(base, "runs", "planD_h_sweep", "plots")
    os.makedirs(out_dir, exist_ok=True)

    K_values = sorted({int(k) for cond_data in summary.values()
                       for tau_data in cond_data.get("min_h_by_K_tau", {}).values()
                       for k in tau_data})

    # ── New diagnostic plots (use raw per-run accuracy, not just min_h summary) ──
    print("\nGenerating new diagnostic plots...")
    plot_accuracy_heatmap(
        summary,
        save_path=os.path.join(out_dir, "plot_accuracy_heatmap.png"),
    )
    print("  -> plot_accuracy_heatmap.png")
    plot_accuracy_curves(
        summary,
        save_path=os.path.join(out_dir, "plot_accuracy_curves.png"),
    )
    print("  -> plot_accuracy_curves.png")

    # ── Per-tau plots (old A/B + new clean capacity) ──
    for tau in args.tau:
        tau_key = str(tau)
        min_h_wd = {}
        min_h_d0 = {}
        for K in K_values:
            wd_val = (summary.get("w_and_d", {})
                             .get("min_h_by_K_tau", {})
                             .get(tau_key, {})
                             .get(str(K)))
            d0_val = (summary.get("d0", {})
                             .get("min_h_by_K_tau", {})
                             .get(tau_key, {})
                             .get(str(K)))
            min_h_wd[K] = wd_val
            min_h_d0[K] = d0_val

        print(f"\nτ = {tau:.0%}")
        print(f"  with_delay : {min_h_wd}")
        print(f"  no_delay   : {min_h_d0}")

        tau_str = f"{int(tau*100)}"
        plot_K_vs_neurons(
            K_values=K_values,
            min_h_with_delay=min_h_wd,
            min_h_no_delay=min_h_d0,
            tau=tau,
            save_path=os.path.join(out_dir, f"plot_A_k_vs_neurons_tau{tau_str}.png"),
        )
        plot_neuron_efficiency(
            K_values=K_values,
            min_h_with_delay=min_h_wd,
            min_h_no_delay=min_h_d0,
            save_path=os.path.join(out_dir, f"plot_B_efficiency_tau{tau_str}.png"),
        )
        plot_capacity_clean(
            K_values=K_values,
            min_h_wd=min_h_wd,
            summary=summary,
            save_path=os.path.join(out_dir, f"plot_capacity_clean_tau{tau_str}.png"),
            tau=tau,
        )
        print(f"  -> all plots saved for τ={tau}")

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

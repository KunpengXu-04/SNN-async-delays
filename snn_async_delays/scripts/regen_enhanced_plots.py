"""
Regenerate enhanced_raster.png and enhanced_flow.png for every run.

Reads directly from the existing diagnostic_data.npz (saved during training)
— no model loading, no random seeds, no new forward passes.
The same recorded spike traces are used, so enhanced plots are directly
comparable to the original spike_raster_sample0.png and
layer_to_layer_spike_flow_sample0.png.

Walks both:
  runs/<run_dir>/plots/diagnostic_data.npz          (direct runs)
  runs/<group>/<run_dir>/plots/diagnostic_data.npz  (nested runs)

Usage (from snn_async_delays/):
    python -m scripts.regen_enhanced_plots                  # skip existing
    python -m scripts.regen_enhanced_plots --force          # overwrite all
    python -m scripts.regen_enhanced_plots --runs_dir runs
"""
import argparse, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.viz import (load_diagnostic_data,
                        plot_enhanced_raster_layers,
                        plot_enhanced_spike_flow)


def collect_runs(runs_dir):
    """Yield run_dir for every directory whose plots/ has diagnostic_data.npz."""
    for entry in sorted(os.listdir(runs_dir)):
        top = os.path.join(runs_dir, entry)
        if not os.path.isdir(top):
            continue
        # Direct run (depth=1)
        if os.path.exists(os.path.join(top, "plots", "diagnostic_data.npz")):
            yield top
        else:
            # Nested run (depth=2)
            for sub in sorted(os.listdir(top)):
                nested = os.path.join(top, sub)
                npz = os.path.join(nested, "plots", "diagnostic_data.npz")
                if os.path.isdir(nested) and os.path.exists(npz):
                    yield nested


def process_run(run_dir, force):
    plot_dir   = os.path.join(run_dir, "plots")
    raster_out = os.path.join(plot_dir, "enhanced_raster.png")
    flow_out   = os.path.join(plot_dir, "enhanced_flow.png")

    if not force and os.path.exists(raster_out) and os.path.exists(flow_out):
        return "skip"

    # Load recorded traces (same sample as existing diagnostic plots)
    traces, weights_dict, delays_dict = load_diagnostic_data(plot_dir)

    # Build title from run directory name
    rel    = os.path.basename(run_dir)
    parent = os.path.basename(os.path.dirname(run_dir))
    title  = f"{parent}/{rel}"

    plot_enhanced_raster_layers(traces, raster_out, title=title)
    plot_enhanced_spike_flow(traces, weights_dict, delays_dict, flow_out, title=title)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if plots already exist")
    args = ap.parse_args()

    runs = list(collect_runs(args.runs_dir))
    print(f"Found {len(runs)} runs with diagnostic_data.npz.")

    counts = {"ok": 0, "skip": 0, "error": 0}
    for run_dir in runs:
        rel = os.path.relpath(run_dir, args.runs_dir)
        try:
            status = process_run(run_dir, args.force)
            counts[status] += 1
            sym = {"ok": "✓", "skip": "–"}.get(status, status)
            print(f"  [{sym}] {rel}")
        except Exception:
            counts["error"] += 1
            print(f"  [!] {rel}")
            traceback.print_exc()

    print(f"\nDone.  ok={counts['ok']}  skipped={counts['skip']}  "
          f"errors={counts['error']}")


if __name__ == "__main__":
    main()

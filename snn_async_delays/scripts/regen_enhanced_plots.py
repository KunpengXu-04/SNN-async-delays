"""
Regenerate enhanced_raster.png, enhanced_flow.png, and diagnostic_panel.png
for every run that has diagnostic_data.npz.

Reads spike traces from diagnostic_data.npz (no model re-run needed).
Reads training curves from train_log.csv and eval results from eval_results.json.

Usage (from snn_async_delays/):
    python -m scripts.regen_enhanced_plots                  # skip existing
    python -m scripts.regen_enhanced_plots --force          # overwrite all
    python -m scripts.regen_enhanced_plots --runs_dir runs
"""
import argparse, csv, json, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.viz import (load_diagnostic_data,
                        plot_enhanced_raster_layers,
                        plot_enhanced_spike_flow,
                        plot_diagnostic_panel)


def _load_log_rows(run_dir: str) -> list:
    """Read train_log.csv and return list of dicts with proper Python types."""
    path = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(path):
        return []
    int_keys   = {"epoch"}
    float_keys = {"train_loss", "val_loss", "train_acc", "val_acc",
                  "mean_hidden_spikes", "time_s"}
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in int_keys:
                if k in row:
                    row[k] = int(row[k])
            for k in float_keys:
                if k in row:
                    row[k] = float(row[k])
            rows.append(row)
    return rows


def _load_eval_results(run_dir: str) -> dict:
    path = os.path.join(run_dir, "eval_results.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_cfg(run_dir: str) -> dict:
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_runs(runs_dir):
    """Yield run_dir for every directory whose plots/ has diagnostic_data.npz."""
    for entry in sorted(os.listdir(runs_dir)):
        top = os.path.join(runs_dir, entry)
        if not os.path.isdir(top):
            continue
        if os.path.exists(os.path.join(top, "plots", "diagnostic_data.npz")):
            yield top
        else:
            for sub in sorted(os.listdir(top)):
                nested = os.path.join(top, sub)
                npz = os.path.join(nested, "plots", "diagnostic_data.npz")
                if os.path.isdir(nested) and os.path.exists(npz):
                    yield nested


def process_run(run_dir, force):
    plot_dir    = os.path.join(run_dir, "plots")
    raster_out  = os.path.join(plot_dir, "enhanced_raster.png")
    flow_out    = os.path.join(plot_dir, "enhanced_flow.png")
    panel_out   = os.path.join(plot_dir, "diagnostic_panel.png")

    need_raster = force or not os.path.exists(raster_out)
    need_flow   = force or not os.path.exists(flow_out)
    need_panel  = force or not os.path.exists(panel_out)

    if not (need_raster or need_flow or need_panel):
        return "skip"

    traces, weights_dict, delays_dict = load_diagnostic_data(plot_dir)

    rel    = os.path.basename(run_dir)
    parent = os.path.basename(os.path.dirname(run_dir))
    title  = f"{parent}/{rel}"

    if need_raster:
        plot_enhanced_raster_layers(traces, raster_out, title=title)
    if need_flow:
        plot_enhanced_spike_flow(traces, weights_dict, delays_dict, flow_out, title=title)
    if need_panel:
        cfg          = _load_cfg(run_dir)
        log_rows     = _load_log_rows(run_dir)
        eval_results = _load_eval_results(run_dir)
        K = cfg.get("K", eval_results.get("K", traces.get("K", 1)))
        plot_diagnostic_panel(
            traces=traces,
            weights_dict=weights_dict,
            delays_dict=delays_dict,
            cfg=cfg,
            log_rows=log_rows,
            eval_results=eval_results,
            save_path=panel_out,
        )
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

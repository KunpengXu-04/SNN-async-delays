"""
Compare LIF firing thresholds from a tuning sweep produced by

    python -m scripts.run_plan_d_h_sweep ... --lif_threshold 0.3 0.4 0.5 0.6 0.8

Reads each run's eval_results.json + config.json and prints an
accuracy / spk-per-trial / rho / per_query table, then recommends the highest
threshold that still fires+trains at K=1 (fixes P1/P2 with the least sparsity
loss). rho = spikes / (n_hid * T) is the energy factor from EXPERIMENT_LOG §33.

Usage (from snn_async_delays/):
    python -m scripts.compare_thresholds --folder "runs/XOR_thr_tune"
    python -m scripts.compare_thresholds --folder "runs/XOR_thr_tune" --plot
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load(folder, condition):
    rows = []
    for f in glob.glob(os.path.join(folder, "*", "eval_results.json")):
        e = _json(f)
        c = _json(os.path.join(os.path.dirname(f), "config.json"))
        if not e or not c or e.get("condition") != condition:
            continue
        thr, h, K = c.get("lif_threshold"), e.get("hidden_size"), e.get("K")
        if thr is None or h is None or K is None:
            continue
        T = (c.get("win_len") or 0) + (c.get("read_len") or 0)
        spk = e.get("mean_hidden_spikes")
        rows.append(dict(
            thr=float(thr), h=int(h), K=int(K), seed=c.get("seed"),
            homeo=float(c.get("homeo_lambda", 0.0) or 0.0),
            htgt=float(c.get("homeo_target", 0.0) or 0.0),
            acc=e.get("accuracy"), spk=spk,
            rho=(spk / (h * T) if (spk is not None and h and T) else None),
            pq=e.get("per_query_acc"),
        ))
    return rows


def resolve(folder):
    if os.path.isabs(folder) or os.path.exists(folder):
        return folder
    return os.path.join(BASE, folder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="runs/XOR_thr_tune")
    ap.add_argument("--condition", default="w_and_d")
    ap.add_argument("--fire_target", type=float, default=0.85,
                    help="K=1 accuracy above which a threshold counts as fires+trains.")
    ap.add_argument("--plot", action="store_true",
                    help="Also save an accuracy-vs-rho + per-query figure.")
    args = ap.parse_args()

    folder = resolve(args.folder)
    rows = load(folder, args.condition)
    if not rows:
        print(f"No '{args.condition}' runs with configs found under {folder}")
        return

    # aggregate over seed per (thr, homeo_lambda, homeo_target, K, h)
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["thr"], r["homeo"], r["htgt"], r["K"], r["h"])
        for m in ("acc", "spk", "rho"):
            if r[m] is not None:
                agg[key][m].append(r[m])
        if r["pq"]:
            agg[key]["pq"].append(np.asarray(r["pq"], float))

    print(f"\nThreshold comparison  ({args.condition}, {folder})")
    print(f"{'thr':>5} {'h_lam':>6} {'h_tgt':>6} {'K':>3} {'h':>4} {'acc':>6} "
          f"{'spk/tr':>7} {'rho':>8}  per_query")
    print("-" * 80)
    for (thr, homeo, htgt, K, h) in sorted(agg):
        d = agg[(thr, homeo, htgt, K, h)]
        pq = np.round(np.mean(d["pq"], axis=0), 2).tolist() if d.get("pq") else "-"
        print(f"{thr:>5} {homeo:>6} {htgt:>6} {K:>3} {h:>4} {np.mean(d['acc']):>6.3f} "
              f"{np.mean(d['spk']):>7.2f} {np.mean(d['rho']):>8.4f}  {pq}")

    # recommendation: highest threshold whose mean K=1 acc >= fire_target
    k1 = defaultdict(list)
    for r in rows:
        if r["K"] == 1 and r["acc"] is not None:
            k1[r["thr"]].append(r["acc"])
    alive = [t for t in sorted(k1) if np.mean(k1[t]) >= args.fire_target]
    print("-" * 64)
    if alive:
        best = max(alive)
        print(f"Recommended threshold: {best}  "
              f"(highest with K=1 acc >= {args.fire_target}: fires+trains, least "
              f"sparsity loss). Check its high-h/K=3 rho + per_query above for the "
              f"energy/recency cost before committing.")
    else:
        print(f"No threshold reached K=1 acc >= {args.fire_target} -- all still dead or "
              f"undertrained. Try lower thresholds or more epochs.")

    if args.plot:
        _plot(agg, folder)


def _plot(agg, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thrs = sorted({thr for (thr, K, h) in agg})
    hs = sorted({h for (thr, K, h) in agg})
    Ks = sorted({K for (thr, K, h) in agg})
    h_fix = max(hs)
    K_fix = max(Ks)
    cmap = plt.cm.viridis

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # A: accuracy vs rho at (h_fix, K_fix), one point per threshold
    for thr in thrs:
        d = agg.get((thr, K_fix, h_fix))
        if d and d.get("acc") and d.get("rho"):
            col = cmap((thr - thrs[0]) / max(1e-9, thrs[-1] - thrs[0]))
            ax1.scatter(np.mean(d["rho"]), np.mean(d["acc"]), s=80, color=col, zorder=3)
            ax1.annotate(f"{thr}", (np.mean(d["rho"]), np.mean(d["acc"])),
                         fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax1.set_xlabel(r"$\rho$ = spikes/$(n_{hid}T)$  (energy; lower = sparser)")
    ax1.set_ylabel("accuracy")
    ax1.set_title(f"Accuracy vs sparsity tradeoff  (h={h_fix}, K={K_fix})")
    ax1.grid(True, alpha=0.3)

    # B: per_query vs position at (h_fix, K_fix), one line per threshold
    for thr in thrs:
        d = agg.get((thr, K_fix, h_fix))
        if d and d.get("pq"):
            pq = np.mean(d["pq"], axis=0)
            col = cmap((thr - thrs[0]) / max(1e-9, thrs[-1] - thrs[0]))
            ax2.plot(range(len(pq)), pq, "o-", color=col, label=f"thr={thr}")
    ax2.axhline(0.5, color="gray", ls="--", lw=1)
    ax2.set_xlabel("query position (0=oldest → K-1=freshest)")
    ax2.set_ylabel("per-query accuracy")
    ax2.set_title(f"Recency gradient by threshold  (h={h_fix}, K={K_fix})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("LIF threshold tuning: sparsity/recency cost of firing enough")
    fig.tight_layout()
    p = os.path.join(folder, "threshold_comparison.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()

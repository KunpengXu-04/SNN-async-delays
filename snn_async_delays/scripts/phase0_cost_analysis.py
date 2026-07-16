"""
LEGACY / DO NOT USE FOR NEW CLAIMS.

This script predates protocol v0.1.  It pools heterogeneous historical runs,
uses pooled accuracy, and contains a deliberately simplified cost proxy.  Its
existing figures are archival diagnostics only; see docs/METRICS_AND_COST.md
and scripts/build_experiment_registry.py before building a replacement
analysis.

Phase 0 -- Retrospective cost-law analysis (no model re-run).

Motivated by the supervisor's whiteboard framework:

    cost  ~  (n_in*n_hid + n_hid*n_out) * T   ~   n_hid * T   (n_in, n_out fixed)

    compress = n_hid' / (K * n_hid)
        n_hid   = hidden dim a *single* query needs to hit the accuracy target
        n_hid'  = shared hidden dim the *K-query* temporal-mux network needs
        ideal (no interference) -> n_hid' = n_hid -> compress = 1/K
        reality -> n_hid' > n_hid -> compress in (1/K, 1] (or worse)

This script walks every run under runs/, reconstructs the analytic cost
C = n_hid_total * T for each, and produces three figures + a master CSV that
test whether this single cost variable governs results across all steps /
topologies, using only already-computed eval_results.json + config.json.

Run (from snn_async_delays/):
    python -m scripts.phase0_cost_analysis
"""

import os
import re
import csv
import json
import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(BASE, "runs")
OUT = os.path.join(RUNS, "cost_law_(phase0)")
os.makedirs(OUT, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# 1. Aggregate every run into flat records
# ────────────────────────────────────────────────────────────────────────────
def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _total_hidden(ev, cfg):
    """Total hidden neurons n_hid'. Prefer explicit scalar, else sum layer list."""
    if ev.get("hidden_size") is not None:
        return int(ev["hidden_size"])
    hs = ev.get("hidden_sizes") or cfg.get("hidden_sizes")
    if isinstance(hs, list) and hs:
        return int(sum(hs))
    if cfg.get("n_hidden") is not None:
        return int(cfg["n_hidden"])
    return None


def _timesteps(ev, cfg):
    """Prefer recorded T; otherwise account for serial Plan-D subwindows."""
    for source in (ev, cfg):
        if source.get("trial_steps") is not None:
            return int(source["trial_steps"])
    K = ev.get("K", cfg.get("K", cfg.get("n_queries")))
    sub_win = ev.get("sub_win", cfg.get("sub_win"))
    win = ev.get("win_len", cfg.get("win_len"))
    read = cfg.get("read_len", ev.get("read_len"))
    if K is not None and sub_win is not None and read is not None:
        return int(K) * int(sub_win) + int(read)
    if win is None or read is None:
        return None
    return int(win) + int(read)


def _delay_active(ev, cfg):
    """True if delays are trainable/non-zero; False for d=0 controls."""
    name = str(ev.get("condition") or ev.get("model_name") or "").lower()
    tm = str(ev.get("train_mode") or cfg.get("train_mode") or "").lower()
    fdv = ev.get("fixed_delay_value", cfg.get("fixed_delay_value"))
    if "d0" in name or "d=0" in name:
        return False
    if fdv == 0 or fdv == 0.0:
        return False
    if tm == "weights_only":
        # weights-only can still use a non-zero *fixed* delay; do not label it
        # as d0 merely because the delay parameter is frozen.
        return fdv not in (None, 0, 0.0)
    # weights_and_delays / delays_only -> delays active
    return tm in ("weights_and_delays", "delays_only")


def experiment_of(run_dir):
    """Human-readable experiment bucket from the top-level runs/ folder name."""
    rel = os.path.relpath(run_dir, RUNS)
    return rel.split(os.sep)[0]


def collect():
    records = []
    for ev_path in glob.glob(os.path.join(RUNS, "**", "eval_results.json"),
                             recursive=True):
        run_dir = os.path.dirname(ev_path)
        if "_archive" in run_dir:            # skip pre-clean archives
            continue
        ev = _load_json(ev_path)
        if ev is None or ev.get("accuracy") is None:
            continue
        cfg = _load_json(os.path.join(run_dir, "config.json")) or {}

        n_hid = _total_hidden(ev, cfg)
        T = _timesteps(ev, cfg)
        K = ev.get("K", cfg.get("K"))
        if n_hid is None or T is None or K is None:
            continue

        rec = {
            "experiment": experiment_of(run_dir),
            "run": os.path.basename(run_dir),
            "delay": _delay_active(ev, cfg),
            "K": int(K),
            "n_hid": int(n_hid),
            "T": int(T),
            "cost": int(n_hid) * int(T),           # analytic C = n_hid * T
            "acc": float(ev["accuracy"]),
            "spikes": ev.get("mean_hidden_spikes"),
            "sub_win": ev.get("sub_win", cfg.get("sub_win")),
            "train_mode": ev.get("train_mode", cfg.get("train_mode")),
        }
        records.append(rec)
    return records


# ────────────────────────────────────────────────────────────────────────────
# 2. Master CSV
# ────────────────────────────────────────────────────────────────────────────
def write_csv(records):
    path = os.path.join(OUT, "master_table.csv")
    cols = ["experiment", "run", "delay", "K", "n_hid", "T", "cost",
            "acc", "spikes", "sub_win", "train_mode"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(records, key=lambda x: (x["experiment"], x["cost"])):
            w.writerow({c: r.get(c) for c in cols})
    print(f"  wrote {path}  ({len(records)} runs)")


# ────────────────────────────────────────────────────────────────────────────
# 3. Figure A -- retrospective collapse: accuracy vs cost = n_hid * T
# ────────────────────────────────────────────────────────────────────────────
def fig_collapse(records):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for delay, color, label in [(True, "#1f77b4", "delay active"),
                                (False, "#d62728", "no delay (d=0 / weights-only)")]:
        pts = [(r["cost"], r["acc"]) for r in records if r["delay"] is delay]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.scatter(x, y, s=26, c=color, alpha=0.55, edgecolors="none", label=label)
    ax.axhline(0.90, ls="--", c="gray", lw=1)
    ax.text(ax.get_xlim()[1], 0.902, " 90%", va="bottom", ha="right",
            color="gray", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"analytic cost  $C = n_{hid}\cdot T$   (log)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Phase 0 collapse: does cost = $n_{hid}\\cdot T$ govern accuracy?\n"
                 "(all experiments, all K, all topologies)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = os.path.join(OUT, "figA_collapse_acc_vs_cost.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  wrote {p}")


# ────────────────────────────────────────────────────────────────────────────
# 4. Op-aware compress / cost / energy curves (overlays any (op, encoding) sweep)
# ────────────────────────────────────────────────────────────────────────────
T_of_K = lambda K: 10 * K + 10          # Plan D: T = K*sub_win + read_len (sub_win=read_len=10)

# Candidate compress-sweep folders under runs/. Any that exist (with eval data) are
# auto-discovered; op + encoding are read from each folder's own eval_results.json.
KNOWN_SWEEPS = [
    "NAND_neuron_sweep_(planD)",        # rate NAND (legacy coarse grid)
    "NAND_compress_burst_(planD)",      # burst NAND (thr=1.0, no homeo)
    "NAND_compress_burst_homeo_(planD)",# burst NAND, thr=1.0 + homeostatic reg
    "NAND_compress_rate_(planD)",       # rate NAND (fine grid, if/when run)
    "XOR_compress_rate_(planD)",
    "XOR_compress_burst_(planD)",       # burst XOR (thr=1.0 + homeo — the clean sweep)
    "XNOR_compress_rate_(planD)",
    "XNOR_compress_burst_(planD)",
]
STYLE_ENC = {"rate": "#1f77b4", "burst": "#e07b00", "burst_jitter": "#8e44ad"}
STYLE_OP = {"NAND": ("o", "-"), "XOR": ("s", "--"), "XNOR": ("^", "-."),
            "AND": ("D", ":"), "OR": ("v", ":")}


def _sweep_meta(folder):
    """(op, encoding, homeo_on) from the first usable eval+config, else None."""
    for ev in glob.glob(os.path.join(folder, "*", "eval_results.json")):
        d = _load_json(ev)
        if d and d.get("accuracy") is not None:
            cfg = _load_json(os.path.join(os.path.dirname(ev), "config.json")) or {}
            homeo = float(cfg.get("homeo_lambda", 0.0) or 0.0) > 0
            return d.get("op", "?"), d.get("encoding_mode", "rate"), homeo
    return None


def discover_sweeps(explicit=None):
    """List of sweep descriptors {folder, op, enc, homeo, label, color, marker, ls}
    for every candidate folder that exists and has data. homeo runs get a '+h' label
    and a dotted line so they don't collide with the no-homeo run of the same op/enc.
    `explicit` overrides KNOWN_SWEEPS (names relative to runs/, or absolute paths)."""
    names = explicit if explicit else KNOWN_SWEEPS
    sweeps = []
    for n in names:
        folder = n if os.path.isabs(n) else os.path.join(RUNS, n)
        meta = _sweep_meta(folder)
        if meta is None:
            continue
        op, enc, homeo = meta
        marker, ls = STYLE_OP.get(op, ("o", "-"))
        if homeo:
            ls = ":"
        sweeps.append({
            "folder": folder, "op": op, "enc": enc, "homeo": homeo,
            "label": f"{op}/{enc}" + ("+h" if homeo else ""),
            "color": STYLE_ENC.get(enc, "gray"), "marker": marker, "ls": ls,
        })
    return sweeps


def load_sweep(folder, tau="0.9"):
    """Compact analysis bundle from a sweep folder's per-run eval_results.json
    (read directly, not the summary json which a partial re-run can truncate).
    Returns None if the folder has no usable Plan-D sweep runs."""
    tau_f = float(tau)
    # mean acc / spikes per (cond, K, h) averaged over seeds
    buckets = defaultdict(lambda: {"spk": [], "acc": []})
    hs = set()
    for ev_path in glob.glob(os.path.join(folder, "*", "eval_results.json")):
        ev = _load_json(ev_path)
        if ev is None or ev.get("accuracy") is None:
            continue
        cond = ev.get("condition")
        K = ev.get("K")
        h = ev.get("hidden_size")
        if cond not in ("w_and_d", "d0") or K is None or h is None:
            continue
        K, h = int(K), int(h)
        hs.add(h)
        buckets[(cond, K, h)]["spk"].append(ev.get("mean_hidden_spikes"))
        buckets[(cond, K, h)]["acc"].append(ev.get("accuracy"))
    if not buckets:
        return None

    spikes, accs = {}, {}
    for (cond, K, h), v in buckets.items():
        spk = [x for x in v["spk"] if x is not None]
        ac = [x for x in v["acc"] if x is not None]
        spikes[(cond, K, h)] = float(np.mean(spk)) if spk else None
        accs[(cond, K, h)] = float(np.mean(ac)) if ac else None

    Ks = sorted({K for (_, K, _) in buckets})
    hs_sorted = sorted(hs)
    # min h' per (cond, K): smallest tested h whose mean acc >= tau
    min_h = {"w_and_d": {}, "d0": {}}
    for cond in ("w_and_d", "d0"):
        for K in Ks:
            hit = next((h for h in hs_sorted
                        if accs.get((cond, K, h)) is not None
                        and accs[(cond, K, h)] >= tau_f), None)
            min_h[cond][K] = hit
    n_base = min_h["w_and_d"].get(1)
    return {"min_h": min_h, "spikes": spikes, "accs": accs,
            "Ks": Ks, "hs": hs_sorted, "n_base": n_base,
            "max_h": max(hs) if hs else 50}


def fig_compress(sweeps, tau="0.9"):
    """n'_hid(K) and compress(K) overlaid for every discovered (op, encoding) sweep."""
    bundles = [(s, load_sweep(s["folder"], tau)) for s in sweeps]
    bundles = [(s, b) for s, b in bundles if b is not None]
    if not bundles:
        print("  [skip] no sweeps found for compress figure")
        return
    allK = sorted({K for _, b in bundles for K in b["Ks"]})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    drawn_repl = {}
    for s, b in bundles:
        c, mk, ls = s["color"], s["marker"], s["ls"]
        gx, gy, cx, cy = [], [], [], []
        for K in b["Ks"]:
            v = b["min_h"]["w_and_d"].get(K)
            (gx.append(K) or gy.append(v)) if v is not None else \
                (cx.append(K) or cy.append(b["max_h"] * 1.15))
        if gx:
            ax1.plot(gx, gy, marker=mk, ls=ls, color=c, label=s["label"])
        if cx:
            ax1.plot(cx, cy, marker=mk, ls="", mfc="none", color=c)
        nb = b["n_base"]
        # one replication reference line per op (uses that op's own n_base)
        if nb and s["op"] not in drawn_repl:
            ax1.plot(allK, [K * nb for K in allK], ":", color="gray", lw=0.8, alpha=0.6)
            drawn_repl[s["op"]] = nb
        # -- compress(K) --
        if nb:
            px = [K for K in b["Ks"] if b["min_h"]["w_and_d"].get(K) is not None]
            py = [b["min_h"]["w_and_d"][K] / (K * nb) for K in px]
            if px:
                ax2.plot(px, py, marker=mk, ls=ls, color=c, label=s["label"])

    ax1.set_xlabel("K (queries per trial)")
    ax1.set_ylabel(r"min $n'_{hid}$ @ 90%   (open = > max tested h)")
    ax1.set_title(r"Neurons needed vs K  (gray dotted = per-op replication $K\,n_{hid}$)")
    ax1.set_xticks(allK); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)

    ax2.plot(allK, [1.0 / K for K in allK], "k:", lw=1.2, label=r"ideal  $1/K$")
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.text(allK[-1], 1.02, "compress = 1 (no gain vs replication)",
             ha="right", fontsize=7, color="gray")
    ax2.set_xlabel("K (queries per trial)")
    ax2.set_ylabel(r"compress $= n'_{hid} / (K\, n_{hid})$")
    ax2.set_title("Compression ratio (lower = better)")
    ax2.set_xticks(allK); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)

    fig.suptitle("Compress curve (Plan D, 90% threshold) — "
                 + ", ".join(s["label"] for s, _ in bundles))
    fig.tight_layout()
    p = os.path.join(OUT, "figB_compress_curve.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  wrote {p}  ({len(bundles)} sweep(s))")


def fig_cost_crossover(sweeps, tau="0.9"):
    """Cost/energy RATIO of temporal multiplexing to spatial replication, per sweep.
    Uses the master identity C_tmp/C_rep = compress(K)*(K+1)/2; <1 = temporal wins.
    Right panel: measured spike-energy ratio spikes(K)/(K*spikes_1)."""
    bundles = [(s, load_sweep(s["folder"], tau)) for s in sweeps]
    bundles = [(s, b) for s, b in bundles if b is not None and b["n_base"]]
    if not bundles:
        print("  [skip] no sweeps for cost crossover")
        return
    allK = sorted({K for _, b in bundles for K in b["Ks"]})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for s, b in bundles:
        c, mk, ls = s["color"], s["marker"], s["ls"]
        nb = b["n_base"]
        # cost ratio = compress * (K+1)/2
        kx, cy_, ex, ey = [], [], [], []
        spk1 = b["spikes"].get(("w_and_d", 1, nb))
        for K in b["Ks"]:
            h = b["min_h"]["w_and_d"].get(K)
            if h is not None:
                kx.append(K)
                cy_.append((h / (K * nb)) * (K + 1) / 2.0)
                spk = b["spikes"].get(("w_and_d", K, h))
                if spk is not None and spk1:
                    ex.append(K); ey.append(spk / (K * spk1))
        if kx:
            ax1.plot(kx, cy_, marker=mk, ls=ls, color=c, label=s["label"])
        if ex:
            ax2.plot(ex, ey, marker=mk, ls=ls, color=c, label=s["label"])

    for ax, ttl, ylab in [
        (ax1, r"Cost ratio $C_{tmp}/C_{rep}=\text{compress}\cdot(K{+}1)/2$",
         r"$C_{tmp}/C_{rep}$"),
        (ax2, "Energy ratio (measured spikes)", r"spikes$(K)\,/\,(K\cdot$spikes$_1)$")]:
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.text(allK[-1], 1.03, "temporal wins below 1", ha="right",
                fontsize=7, color="gray")
        ax.set_xlabel("K"); ax.set_ylabel(ylab); ax.set_title(ttl)
        ax.set_xticks(allK); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    fig.suptitle("Temporal-vs-replication cost & energy ratio (Plan D, 90% threshold)")
    fig.tight_layout()
    p = os.path.join(OUT, "figD_cost_crossover.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  wrote {p}  ({len(bundles)} sweep(s))")


# ────────────────────────────────────────────────────────────────────────────
# 5. Figure C -- the (n_hid, T) plane with iso-cost hyperbolas + feasibility
# ────────────────────────────────────────────────────────────────────────────
def fig_plane(records, sweeps):
    """The supervisor's (n_hid, T) plane, using the DENSEST discovered delay sweep.

    Points coloured by accuracy (colorbar), iso-cost hyperbolas n_hid*T=const, and
    the 90% feasibility boundary = min n'_hid needed at each T (=each K). Feasible
    region is to the RIGHT of the boundary.
    """
    import matplotlib.colors as mcolors

    # pick the discovered sweep with the most (delay) records
    cand = []
    for s in sweeps:
        exp = os.path.basename(s["folder"])
        n = sum(1 for r in records if r["experiment"] == exp and r["delay"])
        if n:
            cand.append((n, exp, s))
    if not cand:
        print("  [skip] no delay runs for plane plot")
        return
    _, experiment, s = max(cand, key=lambda t: t[0])
    label = s["label"]

    sweep = [r for r in records if r["experiment"] == experiment and r["delay"]]
    agg = defaultdict(list)
    for r in sweep:
        agg[(r["n_hid"], r["T"], r["K"])].append(r["acc"])
    pts = [(nh, T, K, float(np.mean(a))) for (nh, T, K), a in agg.items()]
    nh_max = max(p[0] for p in pts)
    T_max = max(p[1] for p in pts)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # iso-cost hyperbolas n_hid * T = const, labelled where they exit the frame
    nh_grid = np.linspace(3, nh_max * 1.05, 400)
    for C in [200, 500, 1000, 2000, 4000, 8000]:
        ax.plot(nh_grid, C / nh_grid, color="gray", lw=0.7, alpha=0.5, zorder=1)
        # label at right edge if the curve passes through the visible T range there
        y_right = C / (nh_max * 1.03)
        if 0 < y_right < T_max * 1.12:
            ax.text(nh_max * 1.03, y_right, f"C={C}", fontsize=7, color="gray",
                    va="center", ha="left")

    # scatter coloured by accuracy, diverging around the 0.90 threshold
    norm = mcolors.TwoSlopeNorm(vmin=0.5, vcenter=0.90, vmax=1.0)
    sc = ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                    c=[p[3] for p in pts], cmap="RdYlGn", norm=norm,
                    s=95, edgecolors="k", linewidths=0.4, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.10)
    cb.set_label("test accuracy")
    cb.ax.axhline(0.90, color="k", lw=1)

    # 90% feasibility boundary: min n_hid reaching 0.90 at each T
    byT = defaultdict(list)
    for nh, T, K, acc in pts:
        byT[T].append((nh, acc))
    bx, by = [], []
    for T in sorted(byT):
        feas = [nh for nh, acc in byT[T] if acc >= 0.90]
        if feas:
            bx.append(min(feas))
            by.append(T)
    if bx:
        ax.plot(bx, by, "-o", color="black", lw=1.8, ms=5, zorder=4,
                label=r"90% feasibility boundary (min $n'_{hid}$)")
        ax.text(bx[-1] + 2, by[-1], "feasible →", fontsize=8, color="black",
                va="center")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="gray", lw=0.7, label=r"iso-cost $n_{hid}\,T$")]
    if bx:
        handles.append(Line2D([], [], color="black", marker="o", lw=1.8,
                              label=r"90% boundary (min $n'_{hid}$)"))
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    ax.set_xlabel(r"$n_{hid}$ (hidden neurons)")
    ax.set_ylabel(r"$T$ (timesteps $= K\cdot$sub_win$+$read_len)")
    ax.set_title(f"(n_hid, T) plane — {label}, Plan D, delay active\n"
                 f"colour = accuracy; each row is a K (T=10K+10); "
                 f"boundary = min neurons to hit 90%")
    ax.set_xlim(0, nh_max * 1.15)
    ax.set_ylim(0, T_max + 10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = os.path.join(OUT, "figC_nhid_T_plane.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  wrote {p}  (source: {experiment})")


# ────────────────────────────────────────────────────────────────────────────
# 6. Figure E -- firing sparsity rho, the factor n_hid*T ignores
# ────────────────────────────────────────────────────────────────────────────
def fig_sparsity(records, sweeps, h_fix=50, K_fix=3):
    r"""rho = spikes / (n_hid * T) = mean firing probability per neuron per timestep.

    The supervisor's C = n_hid*T assumes dense activity (rho=1). SNNs are
    event-driven, so real spike energy = n_hid*T*rho. This figure plots rho
    (log scale) to expose where the burst advantage actually lives.
    """
    # rho per (K, n_hid) for each discovered sweep
    data = {}
    for s in sweeps:
        exp = os.path.basename(s["folder"])
        agg = defaultdict(list)
        for r in records:
            if (r["experiment"] == exp and r["delay"]
                    and r.get("spikes") is not None and r["n_hid"] and r["T"]):
                agg[(r["K"], r["n_hid"])].append(r["spikes"] / (r["n_hid"] * r["T"]))
        if agg:
            data[s["label"]] = ({k: float(np.mean(v)) for k, v in agg.items()}, s)
    if not data:
        print("  [skip] no sparsity data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for label, (d, s) in data.items():
        c, mk, ls = s["color"], s["marker"], s["ls"]
        Ks = sorted({K for (K, h) in d if h == h_fix})
        if Ks:
            ax1.plot(Ks, [d[(K, h_fix)] for K in Ks], marker=mk, ls=ls, color=c, label=label)
        hs = sorted({h for (K, h) in d if K == K_fix})
        if hs:
            ax2.plot(hs, [d[(K_fix, h)] for h in hs], marker=mk, ls=ls, color=c, label=label)

    ax1.set_xlabel("K"); ax1.set_ylabel(r"$\rho$ = spikes / $(n_{hid}\,T)$")
    ax1.set_yscale("log"); ax1.set_title(f"Firing sparsity vs K  (h={h_fix})")
    ax1.legend(fontsize=8); ax1.grid(True, which="both", alpha=0.3)
    ax2.set_xlabel(r"$n_{hid}$"); ax2.set_ylabel(r"$\rho$")
    ax2.set_yscale("log"); ax2.set_title(f"Firing sparsity vs $n_{{hid}}$  (K={K_fix})")
    ax2.legend(fontsize=8); ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(r"The missing factor $\rho$:  real spike energy $= n_{hid}\cdot T\cdot\rho$"
                 r"  (Plan D, delay active) — $n_{hid}T$ alone assumes $\rho=1$")
    fig.tight_layout()
    p = os.path.join(OUT, "figE_sparsity.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  wrote {p}")


# ────────────────────────────────────────────────────────────────────────────
# 7. Figure F -- per-query accuracy vs position (last-query-only diagnostic)
# ────────────────────────────────────────────────────────────────────────────
def _load_per_query(folder):
    """{(K, h): mean per_query_acc vector [K]} for the w_and_d condition."""
    agg = defaultdict(list)
    for ev_path in glob.glob(os.path.join(folder, "*", "eval_results.json")):
        ev = _load_json(ev_path)
        if not ev or ev.get("condition") != "w_and_d":
            continue
        pq = ev.get("per_query_acc")
        K, h = ev.get("K"), ev.get("hidden_size")
        if not pq or K is None or h is None or len(pq) != int(K):
            continue
        agg[(int(K), int(h))].append(np.asarray(pq, dtype=float))
    return {k: np.mean(np.stack(v), axis=0) for k, v in agg.items() if v}


def fig_per_query(sweeps):
    r"""Per-query accuracy vs query position, one panel per sweep.

    Position 0 = first/oldest query (most LIF-decayed by readout time),
    position K-1 = last/freshest query (nearest the readout window). A curve
    rising to the right = the network only solves the freshest query
    ("last-query-only", i.e. time-alignment not true multiplexing). A flat high
    curve = all queries solved (true multiplexing).
    """
    have = [(s, _load_per_query(s["folder"])) for s in sweeps]
    have = [(s, d) for s, d in have if d]
    if not have:
        print("  [skip] no per-query data")
        return

    fig, axes = plt.subplots(1, len(have), figsize=(6 * len(have), 5.2), squeeze=False)
    for ax, (s, d) in zip(axes[0], have):
        # representative h = the one with the most K covered
        hs = sorted({h for (K, h) in d})
        h_fix = max(hs, key=lambda h: sum(1 for (K, hh) in d if hh == h))
        Ks = sorted({K for (K, h) in d if h == h_fix})
        cmap = plt.cm.viridis
        lo, hi = (min(Ks), max(Ks)) if Ks else (1, 1)
        for K in Ks:
            pq = d[(K, h_fix)]
            ax.plot(range(K), pq, "o-", color=cmap((K - lo) / max(1, hi - lo)),
                    label=f"K={K}")
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.text(0, 0.515, "chance", fontsize=7, color="gray")
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("query position\n(0 = oldest → K−1 = freshest / nearest readout)")
        ax.set_ylabel("per-query accuracy")
        ax.set_title(f"{s['label']}  (h={h_fix})")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    fig.suptitle("Per-query accuracy vs position — rising-to-right = only freshest "
                 "query solved (last-query-only, not true multiplexing)")
    fig.tight_layout()
    p = os.path.join(OUT, "figF_per_query.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  wrote {p}  ({len(have)} sweep(s))")


# ────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+", default=None,
                    help="Explicit compress-sweep folders to overlay (names under runs/ "
                         "or absolute paths). Default: auto-discover known NAND/XOR/... "
                         "rate+burst folders that exist.")
    args = ap.parse_args()

    print("Collecting runs...")
    records = collect()
    print(f"  {len(records)} usable runs")
    sweeps = discover_sweeps(args.sweeps)
    print(f"  discovered {len(sweeps)} compress sweep(s): "
          f"{[s['label'] for s in sweeps]}")
    write_csv(records)
    fig_collapse(records)
    fig_compress(sweeps)
    fig_cost_crossover(sweeps)
    fig_plane(records, sweeps)
    fig_sparsity(records, sweeps)
    fig_per_query(sweeps)
    print(f"\nDone. Outputs in: {OUT}")


if __name__ == "__main__":
    main()

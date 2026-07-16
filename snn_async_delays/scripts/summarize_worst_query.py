"""Build labelled historical worst-query tables and figures from aggregate evals.

It does not claim checkpoint re-evaluation: old per-query averages are used to
triage historical runs, and every output is labelled accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tau", type=float, default=0.90)
    args = parser.parse_args()
    root = Path(args.runs_dir)
    if not root.is_absolute():
        root = BASE / root
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = BASE / out
    out.mkdir(parents=True, exist_ok=True)

    buckets: dict[tuple[str, int, int], list[tuple[float, float]]] = defaultdict(list)
    for path in root.rglob("eval_results.json"):
        result = load(path)
        per_query = result.get("per_query_acc")
        K, h, condition = result.get("K"), result.get("hidden_size"), result.get("condition")
        if not (isinstance(per_query, list) and per_query and K is not None and h is not None and condition):
            continue
        if not all(isinstance(value, (int, float)) for value in per_query):
            continue
        buckets[(str(condition), int(K), int(h))].append((float(result.get("accuracy", np.nan)), min(per_query)))

    rows = []
    for (condition, K, h), values in sorted(buckets.items()):
        pooled = [value[0] for value in values]
        worst = [value[1] for value in values]
        rows.append({
            "source": "historical per_query_acc aggregate; not checkpoint re-evaluation",
            "condition": condition,
            "K": K,
            "hidden_size": h,
            "n_seeds": len(values),
            "pooled_accuracy_mean": float(np.mean(pooled)),
            "worst_query_accuracy_mean": float(np.mean(worst)),
            "worst_query_accuracy_sd": float(np.std(worst, ddof=1)) if len(worst) > 1 else None,
            "passes_tau_on_seed_mean": float(np.mean(worst)) >= args.tau,
        })

    with (out / "worst_query_by_K_hidden.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["condition"])
        writer.writeheader(); writer.writerows(rows)

    # Select the first hidden size meeting tau on the *seed mean*, explicitly
    # retaining all raw rows in the CSV for later confidence-interval analysis.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for condition in sorted({row["condition"] for row in rows}):
        selected = []
        unresolved = []
        for K in sorted({row["K"] for row in rows if row["condition"] == condition}):
            all_candidates = [row for row in rows if row["condition"] == condition and row["K"] == K]
            candidates = [row for row in all_candidates if row["passes_tau_on_seed_mean"]]
            if candidates:
                selected.append((K, min(candidates, key=lambda row: row["hidden_size"])["hidden_size"]))
            elif all_candidates:
                # Open markers communicate a censored feasibility boundary:
                # no tested hidden size reached the threshold.
                unresolved.append((K, max(all_candidates, key=lambda row: row["hidden_size"])["hidden_size"]))
        if selected:
            x, y = zip(*selected)
            line, = ax.plot(x, y, "o-", label=f"{condition}: threshold met")
            color = line.get_color()
        else:
            color = None
        if unresolved:
            x, y = zip(*unresolved)
            ax.scatter(x, y, marker="o", facecolors="none", edgecolors=color or "black",
                       s=75, label=f"{condition}: no tested h met threshold")
    ax.set_xlabel("K (queries)")
    ax.set_ylabel(f"smallest tested hidden size with mean worst-query ≥ {args.tau:.0%}")
    ax.set_title("Historical Plan-D triage: position-sensitive threshold\nnot checkpoint re-evaluation; no confidence-interval claim")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "worst_query_min_hidden.png", dpi=180); plt.close(fig)

    metadata = {
        "source": "historical aggregate eval_results.json",
        "metric": "minimum(per_query_acc) computed within each seed/run, then averaged",
        "tau": args.tau,
        "n_cells": len(rows),
        "limitations": [
            "not a fresh checkpoint evaluation",
            "no exact-trial metric is recoverable from historical aggregates",
            "mean-threshold selection is descriptive, not a confirmatory Max-K claim",
        ],
    }
    (out / "README.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "n_cells": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

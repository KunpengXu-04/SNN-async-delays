"""Summarise fresh checkpoint re-evaluations and apply conservative evidence labels.

This is deliberately an evidence classifier, not a leaderboard.  A run cannot
become canonical merely by clearing an accuracy threshold: protocol gates
(adequate seeds, fair controls, declared test discipline) also matter.
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


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def config_for(run_path: str) -> dict[str, Any]:
    path = BASE / run_path / "config.json"
    return load(path) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Re-evaluation JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--canonical-min-seeds", type=int, default=5)
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = BASE / input_path
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = BASE / out
    out.mkdir(parents=True, exist_ok=True)

    records = load(input_path)
    run_rows = []
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        cfg = config_for(record["run_path"])
        result = record.get("result", {})
        condition = str(cfg.get("name", cfg.get("condition", "unknown")))
        K, h, seed = cfg.get("K"), cfg.get("n_hidden"), cfg.get("seed")
        if record.get("status") != "ok" or K is None or h is None:
            run_rows.append({
                "run_path": record["run_path"], "evidence_status": "archived",
                "reason": "checkpoint could not be re-evaluated or config is incomplete",
            })
            continue
        row = {
            "run_path": record["run_path"],
            "condition": condition,
            "K": int(K), "hidden_size": int(h), "seed": seed,
            "pooled_accuracy": result.get("accuracy"),
            "worst_query_accuracy": result.get("worst_query_accuracy"),
            "exact_trial_accuracy": result.get("exact_trial_accuracy"),
            # These runs have fresh metrics, but no predeclared 5-seed protocol,
            # fair readout control, or matched-resource baseline set.
            "evidence_status": "exploratory",
            "reason": (
                "fresh metric available; retained as exploratory because the historical "
                "series has insufficient seeds and unresolved interface/resource controls"
            ),
        }
        run_rows.append(row)
        grouped[(condition, int(K), int(h))].append(row)

    group_rows = []
    for (condition, K, h), values in sorted(grouped.items()):
        worst = [float(value["worst_query_accuracy"]) for value in values]
        exact = [float(value["exact_trial_accuracy"]) for value in values]
        all_seed_pass = all(value >= args.tau for value in worst)
        n_seeds = len(values)
        group_rows.append({
            "condition": condition, "K": K, "hidden_size": h, "n_seeds": n_seeds,
            "pooled_accuracy_mean": float(np.mean([value["pooled_accuracy"] for value in values])),
            "worst_query_accuracy_mean": float(np.mean(worst)),
            "worst_query_accuracy_min_seed": float(min(worst)),
            "exact_trial_accuracy_mean": float(np.mean(exact)),
            "all_evaluated_seeds_pass_worst_tau": all_seed_pass,
            "evidence_status": "exploratory",
            "promotion_gate": (
                "blocked: fewer than canonical-min-seeds"
                if n_seeds < args.canonical_min_seeds else
                "blocked: fair interface/resource controls not yet completed"
            ),
        })

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["evidence_status"])
            writer.writeheader(); writer.writerows(rows)

    write_csv(out / "run_classification.csv", run_rows)
    write_csv(out / "group_summary.csv", group_rows)

    fig, (ax_h, ax_exact) = plt.subplots(1, 2, figsize=(12, 4.8))
    for condition in sorted({row["condition"] for row in group_rows}):
        selected, unresolved = [], []
        Ks = sorted({row["K"] for row in group_rows if row["condition"] == condition})
        for K in Ks:
            cells = [row for row in group_rows if row["condition"] == condition and row["K"] == K]
            feasible = [row for row in cells if row["all_evaluated_seeds_pass_worst_tau"]]
            if feasible:
                selected.append(min(feasible, key=lambda row: row["hidden_size"]))
            else:
                unresolved.append(max(cells, key=lambda row: row["hidden_size"]))
        color = None
        if selected:
            line, = ax_h.plot([row["K"] for row in selected], [row["hidden_size"] for row in selected],
                              "o-", label=f"{condition}: all available seeds pass")
            color = line.get_color()
            ax_exact.plot([row["K"] for row in selected],
                          [row["exact_trial_accuracy_mean"] for row in selected],
                          "o-", color=color, label=condition)
        if unresolved:
            ax_h.scatter([row["K"] for row in unresolved], [row["hidden_size"] for row in unresolved],
                         facecolors="none", edgecolors=color or "black", s=75,
                         label=f"{condition}: no tested h passes")
            ax_exact.scatter([row["K"] for row in unresolved],
                             [row["exact_trial_accuracy_mean"] for row in unresolved],
                             facecolors="none", edgecolors=color or "black", s=75)

    ax_h.set_xlabel("K (queries)")
    ax_h.set_ylabel(f"smallest tested h with all available seeds worst-query ≥ {args.tau:.0%}")
    ax_h.set_title("Fresh checkpoint re-evaluation\nopen marker = no tested h passes")
    ax_h.grid(True, alpha=0.3); ax_h.legend(fontsize=8)
    ax_exact.set_xlabel("K (queries)")
    ax_exact.set_ylabel("mean exact-trial accuracy")
    ax_exact.set_ylim(0, 1.02)
    ax_exact.set_title("Exact-trial reliability at selected/censored h")
    ax_exact.grid(True, alpha=0.3); ax_exact.legend(fontsize=8)
    fig.suptitle("NAND burst Plan-D: fresh metrics, exploratory evidence only", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fresh_worst_and_exact_summary.png", dpi=180)
    plt.close(fig)

    status_counts = {label: sum(row["evidence_status"] == label for row in run_rows)
                     for label in ("canonical", "exploratory", "archived")}
    metadata = {
        "input": str(input_path), "tau": args.tau,
        "classification_policy": (
            "Canonical requires fresh metrics, sufficient predeclared seeds, and completed "
            "fair interface/resource controls; historical results that pass only an accuracy "
            "gate remain exploratory."
        ),
        "run_status_counts": status_counts,
        "group_status_counts": {"canonical": 0, "exploratory": len(group_rows), "archived": 0},
    }
    (out / "README.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

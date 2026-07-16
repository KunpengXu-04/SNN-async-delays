"""Audit historical evaluation files without changing any run artifact.

Historical evaluations usually retain per-query accuracy, which permits a
limited correction from pooled to worst-query accuracy.  They do *not* retain
per-trial predictions, so exact-trial and balanced accuracy cannot be
reconstructed faithfully; the report marks that limitation rather than
inventing values.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]
RUNS = BASE / "runs"
OUT = BASE / "docs" / "generated"


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def main() -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(RUNS.rglob("eval_results.json")):
        result = load(path)
        per_query = result.get("per_query_acc")
        usable_per_query = (
            isinstance(per_query, list)
            and per_query
            and all(isinstance(value, (int, float)) for value in per_query)
        )
        worst = min(per_query) if usable_per_query else None
        pooled = result.get("accuracy")
        pooled_passes_worst_fails = (
            isinstance(pooled, (int, float))
            and worst is not None
            and pooled >= 0.90
            and worst < 0.90
        )
        rows.append({
            "run_path": path.parent.relative_to(RUNS).as_posix(),
            "K": result.get("K"),
            "condition": result.get("condition"),
            "pooled_accuracy": pooled,
            "reported_worst_query_accuracy": worst,
            "has_exact_trial_accuracy": "exact_trial_accuracy" in result,
            "has_balanced_accuracy": "balanced_accuracy" in result,
            "pooled_90_but_worst_below_90": pooled_passes_worst_fails,
            "audit_status": (
                "position_metric_available" if usable_per_query else "position_metric_missing"
            ),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "historical_result_audit.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["run_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_eval_files": len(rows),
        "n_with_per_query": sum(row["audit_status"] == "position_metric_available" for row in rows),
        "n_pooled_90_but_worst_below_90": sum(
            bool(row["pooled_90_but_worst_below_90"]) for row in rows
        ),
        "limitation": (
            "Exact-trial and balanced accuracy cannot be reconstructed from "
            "historical aggregate files that lack per-sample predictions."
        ),
    }
    (OUT / "historical_result_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

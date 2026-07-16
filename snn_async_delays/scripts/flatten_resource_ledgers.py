"""Flatten versioned re-evaluation resource ledgers into analysis-ready CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    if not source.is_absolute():
        source = BASE / source
    if not destination.is_absolute():
        destination = BASE / destination

    records = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for record in records:
        if record.get("status") != "ok":
            continue
        result = record.get("result", {})
        ledger = result.get("resource_ledger")
        if not isinstance(ledger, dict):
            continue
        rows.append({
            "run_path": record.get("run_path"),
            "protocol_version": record.get("protocol_version"),
            "accuracy": result.get("accuracy"),
            "worst_query_accuracy": result.get("worst_query_accuracy"),
            "exact_trial_accuracy": result.get("exact_trial_accuracy"),
            **ledger,
        })

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run_path"])
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"input": str(source), "output": str(destination), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

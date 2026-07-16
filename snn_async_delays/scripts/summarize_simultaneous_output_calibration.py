"""Apply the preregistered d0-only opponent-output threshold rule."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "runs/exploratory/simultaneous_output_interface_calibration_v1"
OUT = BASE / "docs/generated/simultaneous_output_interface_calibration_v1"


def avg(xs): return sum(xs) / len(xs)


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("validation_results.json")):
        cfg = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        r = json.loads(path.read_text(encoding="utf-8"))
        interface = "temporal" if cfg.get("opponent_output_mode") == "shared_windowed" else "spatial"
        spikes = r.get("per_query_output_spikes") or []
        rows.append({"threshold": float(cfg["lif_output_threshold"]), "interface": interface,
                     "seed": int(cfg["seed"]), "worst_balanced": r["worst_query_balanced_accuracy"],
                     "silent_rate": r["output_silent_rate"], "tie_rate": r["output_tie_rate"],
                     "collision_rate": r["output_collision_rate"],
                     "mean_output_spikes_per_query": avg(spikes)})
    if len(rows) != 18: raise SystemExit(f"Incomplete calibration: {len(rows)}/18")
    grouped = defaultdict(list)
    for row in rows: grouped[row["threshold"]].append(row)
    decisions = []
    for threshold, items in sorted(grouped.items()):
        decision = {
            "threshold": threshold,
            "mean_worst_balanced": avg([x["worst_balanced"] for x in items]),
            "mean_silent_rate": avg([x["silent_rate"] for x in items]),
            "mean_tie_rate": avg([x["tie_rate"] for x in items]),
            "mean_collision_rate": avg([x["collision_rate"] for x in items]),
            "mean_output_spikes_per_query": avg([x["mean_output_spikes_per_query"] for x in items]),
        }
        decision["viable"] = (.50 <= decision["mean_worst_balanced"] <= .95
                              and decision["mean_silent_rate"] < .50
                              and decision["mean_tie_rate"] < .50
                              and decision["mean_collision_rate"] < .25)
        decision["distance_to_one_spike"] = abs(decision["mean_output_spikes_per_query"] - 1)
        decisions.append(decision)
    viable = [x for x in decisions if x["viable"]]
    selected = min(viable, key=lambda x: x["distance_to_one_spike"])["threshold"] if viable else None
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "cells.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    output = {"complete": True, "n_cells": len(rows), "thresholds": decisions,
              "selected_output_threshold": selected}
    (OUT / "decision.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()

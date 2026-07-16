"""Apply the preregistered Stage-B WAD optimization decision rule."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "runs/exploratory/wad_optimization_audit_v1/stage_b"
OUT = BASE / "docs/generated/wad_optimization_audit_v1"
ORDER = ["baseline", "narrow_d10", "lr_low", "lr_high", "scalar_noise", "warmup20", "alternating"]


def avg(xs): return sum(xs) / len(xs)


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("validation_results.json")):
        cfg = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        result = json.loads(path.read_text(encoding="utf-8"))
        with (path.parent / "train_log.csv").open(encoding="utf-8", newline="") as f:
            log = list(csv.DictReader(f))
        final = log[-1]
        rows.append({
            "variant": cfg["audit_variant"], "condition": cfg["name"], "seed": int(cfg["seed"]),
            "worst_query_accuracy": float(result["worst_query_accuracy"]),
            "exact_trial_accuracy": float(result["exact_trial_accuracy"]),
            "mean_hidden_spikes": float(result["mean_hidden_spikes"]),
            "final_delay_movement": float(final["mean_abs_delay_movement"]),
            "final_delay_saturation": float(final["delay_saturation_fraction"]),
            "mean_delay_gradient": avg([float(x["delay_grad_norm"]) for x in log]),
            "run_dir": str(path.parent.relative_to(BASE)),
        })
    if len(rows) != 42: raise SystemExit(f"Incomplete Stage B: {len(rows)}/42")
    by = {(x["variant"], x["condition"], x["seed"]): x for x in rows}
    baseline = {s: by[("baseline", "w_and_d", s)]["worst_query_accuracy"] for s in (0, 1, 42)}
    decisions = []
    for variant in ORDER:
        wad = [by[(variant, "w_and_d", s)] for s in (0, 1, 42)]
        scalar = [by[(variant, "scalar", s)] for s in (0, 1, 42)]
        gains = [wad[i]["worst_query_accuracy"] - baseline[s] for i, s in enumerate((0, 1, 42))]
        gaps = [wad[i]["worst_query_accuracy"] - scalar[i]["worst_query_accuracy"] for i in range(3)]
        retained = (
            avg(gains) >= .03 and sum(g >= .03 for g in gains) >= 2
            and avg(gaps) >= -.01
            and all(x["final_delay_movement"] >= .05 for x in wad)
            and all(x["final_delay_saturation"] < .10 for x in wad)
        )
        decisions.append({
            "variant": variant,
            "mean_wad_worst_query": avg([x["worst_query_accuracy"] for x in wad]),
            "mean_scalar_worst_query": avg([x["worst_query_accuracy"] for x in scalar]),
            "paired_wad_minus_baseline": gains, "mean_wad_minus_baseline": avg(gains),
            "n_seed_gains_ge_0_03": sum(g >= .03 for g in gains),
            "paired_wad_minus_scalar": gaps, "mean_wad_minus_scalar": avg(gaps),
            "min_wad_delay_movement": min(x["final_delay_movement"] for x in wad),
            "max_wad_saturation": max(x["final_delay_saturation"] for x in wad),
            "mean_wad_exact_trial": avg([x["exact_trial_accuracy"] for x in wad]),
            "mean_scalar_exact_trial": avg([x["exact_trial_accuracy"] for x in scalar]),
            "retained": retained,
        })
    eligible = [x for x in decisions if x["retained"]]
    selected = None
    if eligible:
        best = max(x["mean_wad_worst_query"] for x in eligible)
        tied = [x for x in eligible if best - x["mean_wad_worst_query"] <= .01]
        selected = min(tied, key=lambda x: ORDER.index(x["variant"]))["variant"]
    else:
        selected = "baseline"
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "stage_b_cells.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (OUT / "stage_b_decisions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(decisions[0])); w.writeheader(); w.writerows(decisions)
    output = {"complete": True, "n_cells": len(rows), "variants": decisions,
              "eligible_variants": [x["variant"] for x in eligible], "selected_variant": selected,
              "optimization_rescue_passed": bool(eligible)}
    (OUT / "stage_b_decision.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()

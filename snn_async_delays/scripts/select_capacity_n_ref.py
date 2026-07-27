"""Freeze N_ref from the registered K=6, B=48 fixed-oracle calibration."""

from __future__ import annotations

import json

from scripts.run_spatial_temporal_capacity_slayer import BASE, load_protocol


def select_n_ref() -> dict:
    protocol = load_protocol()
    root = (
        BASE / protocol["execution"]["exploratory_root"]
        / "spatial_calibration" / "shared_temporal_fixed"
    )
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in root.rglob("validation_results.json")
    ]
    seeds = set(map(int, protocol["gates"]["s1"]["calibration_seeds"]))
    candidates = []
    for hidden in protocol["spatial_scaling"]["hidden_neurons_total"]:
        members = [
            row for row in rows
            if int(row["N_hidden_total"]) == int(hidden)
            and int(row["seed"]) in seeds
        ]
        candidates.append({
            "N_hidden_total": int(hidden), "seeds_observed": len(members),
            "seeds_passing": sum(bool(row["pass_reliability"]) for row in members),
            "passes_5_of_5": len(members) == 5 and all(
                bool(row["pass_reliability"]) for row in members
            ),
        })
    passing = [row["N_hidden_total"] for row in candidates if row["passes_5_of_5"]]
    decision = {
        "protocol_id": protocol["protocol_id"],
        "rule": "smallest_fixed_oracle_N_passing_K6_B48_in_all_calibration_seeds",
        "K": 6, "output_budget_B": 48, "candidates": candidates,
        "passed": bool(passing), "N_ref": min(passing) if passing else None,
        "test_split_opened": False,
    }
    output = BASE / protocol["execution"]["generated_root"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "N_ref_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    return decision


if __name__ == "__main__":
    print(json.dumps(select_n_ref(), indent=2))

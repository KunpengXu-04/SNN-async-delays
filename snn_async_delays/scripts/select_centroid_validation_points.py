"""Select fixed-oracle spatial boundary points for centroid equivalence checks."""

from __future__ import annotations

import json
from collections import defaultdict

from scripts.run_spatial_temporal_capacity_slayer import BASE, load_protocol


def select_points() -> dict:
    protocol = load_protocol()
    root = (
        BASE / protocol["execution"]["exploratory_root"] / "spatial"
        / "shared_temporal_fixed"
    )
    groups = defaultdict(list)
    for path in root.rglob("validation_results.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        groups[(int(row["K"]), int(row["N_hidden_total"]))].append(row)
    points = []
    for K in protocol["workload"]["K_values"]:
        passing = [
            hidden for (query_count, hidden), rows in groups.items()
            if query_count == int(K) and len(rows) == 5
            and all(bool(row["pass_reliability"]) for row in rows)
        ]
        if passing:
            points.append({"K": int(K), "N_hidden_total": min(passing)})
    decision = {
        "protocol_id": protocol["protocol_id"],
        "selection_rule": "minimum_5_of_5_fixed_oracle_width_per_K",
        "selected_points": points,
        "all_K_represented": len(points) == len(protocol["workload"]["K_values"]),
        "test_split_opened": False,
    }
    output = BASE / protocol["execution"]["generated_root"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "centroid_validation_points.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    return decision


if __name__ == "__main__":
    print(json.dumps(select_points(), indent=2))

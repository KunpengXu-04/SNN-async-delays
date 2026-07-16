"""Run the six locked cells of temporal viability preflight v1."""

from __future__ import annotations

import argparse
import copy

import yaml

from scripts.run_simultaneous_pilot import BASE, run_cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prereg = yaml.safe_load(
        (BASE / "configs/simultaneous_temporal_viability_preflight_v1.yaml")
        .read_text(encoding="utf-8")
    )
    if prereg.get("status") != "preregistered":
        raise SystemExit("Temporal viability preflight v1 is not preregistered")
    base = yaml.safe_load(
        (BASE / "configs/simultaneous_temporal_routing_pilot_v1.yaml")
        .read_text(encoding="utf-8")
    )
    root = BASE / "runs/exploratory" / prereg["protocol_id"]
    print(f"{prereg['protocol_id']}: {prereg['cells']} locked cells")

    for condition in prereg["conditions"]:
        protocol = copy.deepcopy(base)
        protocol["protocol_id"] = prereg["protocol_id"]
        protocol["frozen_output_threshold"] = prereg["frozen_output_threshold"]
        protocol["output_membrane_warmup_epochs"] = prereg["output_membrane_warmup_epochs"]
        protocol["output_membrane_aux_weight"] = prereg["output_membrane_aux_weight"]
        protocol["checkpoint_selection"] = prereg["checkpoint_selection"]
        protocol["training"] = copy.deepcopy(prereg["training"])
        protocol["fixed_matched_delay_range"] = prereg["fixed_narrow_range"]
        if condition == "temporal_scaffold":
            protocol["input_hidden_delay_scaffold"] = prereg["temporal_scaffold"]
        else:
            protocol.pop("input_hidden_delay_scaffold", None)
        run_cell(protocol, prereg["endpoint"], condition, int(prereg["seed"]),
                 root, args.device, args.dry_run)


if __name__ == "__main__":
    main()

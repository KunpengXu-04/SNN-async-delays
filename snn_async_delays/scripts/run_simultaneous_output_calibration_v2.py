"""Run the 9-cell temporal-scaffold arm of output calibration v2."""

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
    prereg = yaml.safe_load((BASE / "configs/simultaneous_output_interface_calibration_v2.yaml").read_text(encoding="utf-8"))
    if prereg.get("status") != "preregistered":
        raise SystemExit("Calibration v2 is not preregistered")
    base = yaml.safe_load((BASE / "configs/simultaneous_temporal_routing_pilot_v1.yaml").read_text(encoding="utf-8"))
    arm = prereg["temporal_arm"]
    total = len(arm["thresholds"]) * len(arm["seeds"])
    print(f"{prereg['protocol_id']}: {total} temporal-scaffold cells")
    for threshold in arm["thresholds"]:
        protocol = copy.deepcopy(base)
        protocol["protocol_id"] = prereg["protocol_id"]
        protocol["frozen_output_threshold"] = threshold
        protocol["input_hidden_delay_scaffold"] = arm["input_hidden_delay_scaffold"]
        protocol["output_membrane_warmup_epochs"] = arm["output_membrane_warmup_epochs"]
        protocol["output_membrane_aux_weight"] = arm["output_membrane_aux_weight"]
        protocol["checkpoint_selection"] = arm["checkpoint_selection"]
        root = BASE / "runs/exploratory" / prereg["protocol_id"] / f"threshold_{threshold}"
        for seed in arm["seeds"]:
            run_cell(protocol, arm["endpoint"], arm["condition"], seed,
                     root, args.device, args.dry_run)


if __name__ == "__main__":
    main()

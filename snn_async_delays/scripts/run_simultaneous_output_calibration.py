"""Run the preregistered d0-only opponent-output threshold calibration."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from scripts.run_simultaneous_pilot import BASE, run_cell


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    calibration = yaml.safe_load((BASE / "configs/simultaneous_output_interface_calibration_v1.yaml").read_text(encoding="utf-8"))
    if calibration["status"] != "preregistered": raise SystemExit("Calibration is not preregistered")
    total = len(calibration["thresholds"]) * len(calibration["seeds"]) * len(calibration["interfaces"])
    print(f"{calibration['protocol_id']}: {total} d0-only cells")
    for interface in calibration["interfaces"]:
        filename = ("simultaneous_spatial_control_pilot_v1.yaml" if interface == "spatial"
                    else "simultaneous_temporal_routing_pilot_v1.yaml")
        base = yaml.safe_load((BASE / "configs" / filename).read_text(encoding="utf-8"))
        endpoint = "opponent_parallel" if interface == "spatial" else "opponent_shared_windowed"
        for threshold in calibration["thresholds"]:
            protocol = copy.deepcopy(base)
            protocol["protocol_id"] = calibration["protocol_id"]
            protocol["frozen_output_threshold"] = threshold
            root = BASE / "runs/exploratory" / calibration["protocol_id"] / interface / f"threshold_{threshold}"
            for seed in calibration["seeds"]:
                run_cell(protocol, endpoint, "d0", seed, root, args.device, args.dry_run)


if __name__ == "__main__": main()

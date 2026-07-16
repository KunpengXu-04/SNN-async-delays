"""Run the locked 45-cell counterbalanced temporal performance matrix."""

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
        (BASE / "configs/simultaneous_temporal_counterbalanced_performance_v1.yaml")
        .read_text(encoding="utf-8")
    )
    if prereg.get("status") != "preregistered":
        raise SystemExit("Counterbalanced temporal performance v1 is not preregistered")
    base = yaml.safe_load(
        (BASE / "configs/simultaneous_temporal_routing_pilot_v1.yaml")
        .read_text(encoding="utf-8")
    )
    root = BASE / "runs/exploratory" / prereg["protocol_id"]
    print(f"{prereg['protocol_id']}: {prereg['cells']} locked cells")
    for order_id, query_ops in prereg["operation_orders"].items():
        for seed in prereg["held_out_seeds"]:
            for condition in prereg["conditions"]:
                protocol = copy.deepcopy(base)
                protocol.update({
                    "protocol_id": prereg["protocol_id"],
                    "query_ops": list(query_ops),
                    "K": prereg["K"],
                    "input_win": prereg["input_win"],
                    "output_window_len": prereg["output_window_len"],
                    "read_len": prereg["read_len"],
                    "hidden_size": prereg["hidden_size"],
                    "frozen_wad_config": copy.deepcopy(prereg["frozen_wad_config"]),
                    "frozen_output_threshold": prereg["frozen_output_threshold"],
                    "fixed_matched_delay_range": prereg["fixed_matched_delay_range"],
                    "encoding": copy.deepcopy(prereg["encoding"]),
                    "training": copy.deepcopy(prereg["training"]),
                    "loss_reduction": prereg["loss_reduction"],
                    "output_membrane_warmup_epochs": prereg["output_membrane_warmup_epochs"],
                    "output_membrane_aux_weight": prereg["output_membrane_aux_weight"],
                    "checkpoint_selection": prereg["checkpoint_selection"],
                })
                protocol.pop("input_hidden_delay_scaffold", None)
                run_cell(
                    protocol, prereg["endpoint"], condition, int(seed),
                    root / order_id, args.device, args.dry_run,
                )


if __name__ == "__main__":
    main()


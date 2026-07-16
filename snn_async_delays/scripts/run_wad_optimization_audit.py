"""Run Stage A or gated Stage B of WAD optimization audit v1."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from scripts.run_plan_d_h_sweep import run_single


BASE = Path(__file__).resolve().parents[1]
PROTOCOL = "wad_optimization_audit_v1"


def condition(name: str) -> dict:
    if name == "scalar":
        return {"name": "scalar", "train_mode": "weights_and_delays",
                "fixed_delay_value": None, "shared_delay": True}
    if name == "w_and_d":
        return {"name": "w_and_d", "train_mode": "weights_and_delays",
                "fixed_delay_value": None, "shared_delay": False}
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wad_optimization_audit_v1.yaml")
    parser.add_argument("--stage", choices=["a", "b"], default="a")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = Path(args.config)
    if not path.is_absolute(): path = BASE / path
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL or protocol.get("status") != "preregistered":
        raise SystemExit("Refusing modified or non-preregistered audit")
    task, fixed = protocol["fixed_task"], protocol["fixed_training"]
    if args.stage == "a":
        stage = protocol["stage_a_threshold_screen"]
        variants = [
            {"name": f"threshold_{threshold}", "lif_threshold": threshold,
             "d_max": stage["d_max"], "lr_d": stage["lr_d"],
             "optimization_schedule": stage["optimization_schedule"],
             "delay_init_mode": stage["delay_init_mode"]}
            for threshold in stage["thresholds"]
        ]
    else:
        stage = protocol["stage_b_gated_variants"]
        if stage.get("selected_threshold") is None:
            raise SystemExit("Stage B locked: selected_threshold is null; complete and log Stage A first")
        variants = [dict(item, lif_threshold=stage["selected_threshold"])
                    for item in stage["variants"]]

    total = len(variants) * len(stage["conditions"]) * len(stage["seeds"])
    root = BASE / "runs" / "exploratory" / PROTOCOL / f"stage_{args.stage}"
    print(f"{PROTOCOL} stage {args.stage}: {total} validation-only cells")
    idx = 0
    for variant in variants:
        variant_root = root / variant["name"]
        for seed in stage["seeds"]:
            for condition_name in stage["conditions"]:
                idx += 1
                cfg = {
                    **fixed, **task, **{k:v for k,v in variant.items() if k != "name"},
                    "seed": seed, "n_hidden": task["hidden_size"], "ops_list": ["XOR"],
                    "same_op": True, "experiment": PROTOCOL, "protocol_id": PROTOCOL,
                    "audit_stage": args.stage, "audit_variant": variant["name"],
                    "protocol_config": str(path.relative_to(BASE)), "test_split_opened": False,
                    "no_diag": False,
                }
                print(f"[{idx}/{total}] {variant['name']} seed={seed} cond={condition_name}")
                run_single(cfg, task["K"], task["hidden_size"], condition(condition_name),
                           seed, args.device, str(variant_root), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

"""Run preregistered XOR delay-control matrix v1 (validation only)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from scripts.run_plan_d_h_sweep import run_single


BASE = Path(__file__).resolve().parents[1]
PROTOCOL = "xor_delay_control_matrix_v1"


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("protocol_id") != PROTOCOL or cfg.get("status") != "preregistered":
        raise SystemExit("Refusing modified, launched, or non-preregistered matrix")
    if cfg["fixed_training"].get("evaluation_split") != "val":
        raise SystemExit("Matrix v1 must keep the test split sealed")
    return cfg


def condition_for(name: str, protocol: dict, seed: int) -> dict:
    spec = dict(protocol["delay_controls"][name])
    spec.pop("initialization", None)
    spec.pop("range", None)
    spec.pop("seed_rule", None)
    condition = {"name": name, "fixed_delay_value": None,
                 "fixed_delay_distribution": None, "shared_delay": False, **spec}
    if name == "fixed_heterogeneous":
        condition["fixed_delay_seed"] = seed
    return condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xor_delay_control_matrix_v1.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = Path(args.config)
    if not path.is_absolute():
        path = BASE / path
    protocol = load_protocol(path)
    root = Path(args.runs_dir) if args.runs_dir else BASE / "runs" / "exploratory" / PROTOCOL
    if not root.is_absolute():
        root = BASE / root
    root.mkdir(parents=True, exist_ok=True)

    total = sum(
        len(setting["readout_types"]) * len(protocol["factors"]["seeds"]) *
        len(protocol["factors"]["delay_conditions"])
        for setting in protocol["settings"]
    )
    print(f"{PROTOCOL}: {total} validation-only training cells; output={root}")
    index = 0
    for setting in protocol["settings"]:
        for readout_type in setting["readout_types"]:
            for seed in protocol["factors"]["seeds"]:
                for condition_name in protocol["factors"]["delay_conditions"]:
                    index += 1
                    condition = condition_for(condition_name, protocol, seed)
                    cfg = {
                        **protocol["fixed_training"], **protocol["task"],
                        "K": setting["K"], "n_hidden": setting["hidden_size"],
                        "seed": seed, "ops_list": ["XOR"], "same_op": True,
                        "observation_mode": protocol["factors"]["observation_mode"],
                        "readout_type": readout_type,
                        "experiment": PROTOCOL, "protocol_id": PROTOCOL,
                        "protocol_config": str(path.relative_to(BASE)),
                        "matrix_setting": setting["name"],
                        "calibration_only": False, "test_split_opened": False,
                        "no_diag": False,
                    }
                    print(
                        f"[{index}/{total}] {setting['name']} K={setting['K']} "
                        f"h={setting['hidden_size']} ro={readout_type} seed={seed} "
                        f"cond={condition_name}"
                    )
                    run_single(
                        cfg, setting["K"], setting["hidden_size"], condition,
                        seed, args.device, str(root), dry_run=args.dry_run,
                    )

    if not args.dry_run:
        print("Training matrix complete. Run post-training interventions with:")
        print("python -m scripts.run_xor_delay_control_interventions --device", args.device)


if __name__ == "__main__":
    main()

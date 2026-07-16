"""Run preregistered validation-only XOR difficulty calibration v1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.run_plan_d_h_sweep import CONDITIONS, run_single


BASE = Path(__file__).resolve().parents[1]
EXPECTED_PROTOCOL = "xor_difficulty_calibration_v1"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def summarize(run_root: Path, output: Path) -> None:
    rows = []
    for result_path in sorted(run_root.rglob("validation_results.json")):
        config_path = result_path.with_name("config.json")
        if not config_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        ledger = result.get("resource_ledger", {})
        rows.append({
            "run_path": result_path.parent.relative_to(BASE).as_posix(),
            "K": cfg.get("K"),
            "hidden_size": cfg.get("n_hidden"),
            "T": cfg.get("win_len", 0) + cfg.get("read_len", 0),
            "sub_win": cfg.get("sub_win"),
            "seed": cfg.get("seed"),
            "condition": cfg.get("name"),
            "evaluation_split": result.get("evaluation_split"),
            "accuracy": result.get("accuracy"),
            "worst_query_accuracy": result.get("worst_query_accuracy"),
            "exact_trial_accuracy": result.get("exact_trial_accuracy"),
            "balanced_accuracy": result.get("balanced_accuracy"),
            "mean_hidden_spikes": result.get("mean_hidden_spikes"),
            "trainable_parameters": ledger.get("trainable_parameters"),
            "model_scalar_storage_elements": ledger.get("model_scalar_storage_elements"),
            "delay_buffer_elements_per_sample": ledger.get("delay_buffer_elements_per_sample"),
            "neuron_updates_per_trial": ledger.get("neuron_updates_per_trial"),
            "dense_synapse_macs_per_trial": ledger.get("dense_synapse_macs_per_trial"),
            "decoder_weight_macs_per_trial": ledger.get("decoder_weight_macs_per_trial"),
            "mean_synaptic_events_total": ledger.get("mean_synaptic_events_total"),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run_path"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xor_difficulty_calibration_v1.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE / config_path
    protocol = load_yaml(config_path)
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL or protocol.get("status") != "preregistered":
        raise SystemExit("Refusing modified or non-preregistered calibration protocol")
    if protocol["fixed_training"].get("evaluation_split") != "val":
        raise SystemExit("Calibration must evaluate validation only")

    task = protocol["task"]
    factors = protocol["factors"]
    fixed = {**protocol["fixed_training"], **protocol["fixed_interface"]}
    run_root = Path(args.runs_dir) if args.runs_dir else BASE / "runs" / "exploratory" / EXPECTED_PROTOCOL
    if not run_root.is_absolute():
        run_root = BASE / run_root
    run_root.mkdir(parents=True, exist_ok=True)

    condition_map = {condition["name"]: condition for condition in CONDITIONS}
    total = len(task["K"]) * len(factors["hidden_size"]) * len(factors["seeds"]) * len(factors["delay_conditions"])
    print(f"{EXPECTED_PROTOCOL}: {total} validation-only cells; output={run_root}")

    index = 0
    for K in task["K"]:
        for hidden_size in factors["hidden_size"]:
            for seed in factors["seeds"]:
                for condition_name in factors["delay_conditions"]:
                    index += 1
                    cfg = {
                        **fixed, **task,
                        "K": K,
                        "n_hidden": hidden_size,
                        "seed": seed,
                        "ops_list": ["XOR"],
                        "experiment": EXPECTED_PROTOCOL,
                        "protocol_id": EXPECTED_PROTOCOL,
                        "protocol_config": str(config_path.relative_to(BASE)),
                        "calibration_only": True,
                        "test_split_opened": False,
                        "no_diag": False,
                    }
                    print(f"[{index}/{total}] K={K} h={hidden_size} seed={seed} cond={condition_name}")
                    run_single(
                        cfg, K, hidden_size, condition_map[condition_name], seed,
                        args.device, str(run_root), dry_run=args.dry_run,
                    )

    if not args.dry_run:
        output = BASE / "docs" / "generated" / f"{EXPECTED_PROTOCOL}_run_level.csv"
        summarize(run_root, output)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()

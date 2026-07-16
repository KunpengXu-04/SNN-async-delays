"""Run the pre-registered XOR readout-interface pilot v1.

The YAML protocol is copied into each saved config through ``run_single``.
The runner refuses non-preregistered protocol identifiers and never reuses a
late-window result for a different observation mode or decoder.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.run_plan_d_h_sweep import CONDITIONS, run_single


BASE = Path(__file__).resolve().parents[1]
EXPECTED_PROTOCOL = "xor_readout_interface_pilot_v1"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def summarize(run_root: Path, output: Path) -> None:
    rows = []
    for result_path in sorted(run_root.rglob("eval_results.json")):
        config_path = result_path.with_name("config.json")
        if not config_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        ledger = result.get("resource_ledger", {})
        rows.append({
            "run_path": result_path.parent.relative_to(BASE).as_posix(),
            "seed": cfg.get("seed"),
            "condition": cfg.get("name"),
            "observation_mode": cfg.get("observation_mode"),
            "readout_type": cfg.get("readout_type"),
            "accuracy": result.get("accuracy"),
            "worst_query_accuracy": result.get("worst_query_accuracy"),
            "exact_trial_accuracy": result.get("exact_trial_accuracy"),
            "balanced_accuracy": result.get("balanced_accuracy"),
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
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_xor_readout_v1.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE / config_path
    protocol = load_yaml(config_path)
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL or protocol.get("status") != "preregistered":
        raise SystemExit("Refusing to run a modified or non-preregistered pilot protocol")

    task = protocol["task"]
    factors = protocol["factors"]
    fixed = protocol["fixed_training"]
    run_root = Path(args.runs_dir) if args.runs_dir else BASE / "runs" / "canonical" / EXPECTED_PROTOCOL
    if not run_root.is_absolute():
        run_root = BASE / run_root
    run_root.mkdir(parents=True, exist_ok=True)

    condition_map = {condition["name"]: condition for condition in CONDITIONS}
    total = (
        len(factors["hidden_size"]) * len(factors["seeds"]) *
        len(factors["delay_conditions"]) * len(factors["observation_modes"]) *
        len(factors["readout_types"]) * len(task["K"])
    )
    print(f"{EXPECTED_PROTOCOL}: {total} preregistered cells; output={run_root}")

    index = 0
    for K in task["K"]:
        for hidden_size in factors["hidden_size"]:
            for seed in factors["seeds"]:
                for condition_name in factors["delay_conditions"]:
                    for observation_mode in factors["observation_modes"]:
                        for readout_type in factors["readout_types"]:
                            index += 1
                            cfg = {
                                **fixed, **task,
                                "K": K,
                                "n_hidden": hidden_size,
                                "seed": seed,
                                "observation_mode": observation_mode,
                                "readout_type": readout_type,
                                "ops_list": ["XOR"],
                                "experiment": EXPECTED_PROTOCOL,
                                "protocol_id": EXPECTED_PROTOCOL,
                                "protocol_config": str(config_path.relative_to(BASE)),
                                "no_diag": True,
                            }
                            print(
                                f"[{index}/{total}] K={K} h={hidden_size} seed={seed} "
                                f"cond={condition_name} obs={observation_mode} ro={readout_type}"
                            )
                            run_single(
                                cfg, K, hidden_size, condition_map[condition_name], seed,
                                args.device, str(run_root), dry_run=args.dry_run,
                            )

    summary = BASE / "docs" / "generated" / f"{EXPECTED_PROTOCOL}_run_level.csv"
    if not args.dry_run:
        summarize(run_root, summary)
        print(f"wrote {summary}")


if __name__ == "__main__":
    main()

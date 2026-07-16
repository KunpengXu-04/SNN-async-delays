"""Run preregistered spatial-vs-temporal Pareto Phase-0 Stage A."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data.boolean_dataset import (
    ExhaustiveFixedOperationQueryDataset,
    FixedOperationQueryDataset,
)
from data.encoding import encode_simultaneous_trial
from snn.model import SNNSimultaneousModel
from train.eval import evaluate_simultaneous, save_eval_results
from train.trainer import SimultaneousTrainer
from utils.seed import set_seed
from utils.viz import save_run_diagnostic_plots


BASE = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE / "configs/spatial_vs_temporal_pareto_phase0.yaml"


def build_stage_a_config(protocol: dict, hidden: int, seed: int, smoke: bool) -> dict:
    stage = protocol["stage_a"]
    training = dict(stage["training"])
    if smoke:
        training.update({"epochs": 100, "n_train": 256, "n_val": 128})
    encoding = protocol["encoding_decision"]
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_stage": "stage_a",
        "experiment": protocol["protocol_id"],
        "name": "spatial_independent_d0",
        "seed": seed,
        "K": 1,
        "query_ops": [stage["operation"]],
        "ops_list": [stage["operation"]],
        "input_schedule": "simultaneous",
        "n_input": 4,
        "n_hidden": hidden,
        "win_len": int(stage["input_win"]),
        "read_len": int(stage["read_len"]),
        "output_window_len": int(stage["output_window_len"]),
        "d_max": int(stage["d_max"]),
        "train_mode": "weights_only",
        "fixed_delay_value": 0.0,
        "fixed_delay_distribution": None,
        "shared_delay": False,
        "output_delay_mode": stage["output_delay_mode"],
        # Stage A is the all-d0 spatial baseline. The prior metadata string
        # "input_to_hidden_only" described the future Stage-B intervention
        # locus, but was misleading here: fixed_delay_value=0 freezes d_ih,
        # while output_delay_mode=d0 freezes d_ho.
        "delay_placement": "all_synapses_fixed_d0",
        "delay_param_type": "sigmoid",
        "delay_step": 1.0,
        "delay_init_mode": "constant",
        "delay_init_raw": -2.0,
        "delay_init_std": 0.25,
        "lif_tau_m": 10.0,
        "lif_threshold": float(stage["hidden_threshold"]),
        "lif_output_threshold": float(stage["output_threshold"]),
        "lif_reset": 0.0,
        "lif_refractory": 2,
        "surrogate_beta": 4.0,
        "dt": 1.0,
        "readout_type": "linear",
        "readout_endpoint": "opponent_parallel_target_spike",
        "observation_mode": "all_time",
        "use_output_spikes": True,
        "opponent_output_mode": "parallel_pairs",
        "n_output_neurons": 2,
        "encoding_mode": "binary_one_hot",
        "one_hot_phase": float(encoding["one_hot_phase"]),
        "one_hot_n_spikes": int(encoding["spikes_per_selected_value_channel"]),
        "burst_n_spikes_on": 1,
        "burst_n_spikes_off": 1,
        "burst_phase_on": 1.0,
        "burst_phase_off": 1.0,
        "burst_jitter_ms": int(encoding["jitter_ms"]),
        "opponent_target_timing_mode": encoding["output_target_timing"],
        "output_target_offset_steps": int(encoding["output_target_offset_steps"]),
        "target_filter_tau_steps": float(encoding["target_filter_tau_steps"]),
        "target_spike_loss_weight": float(encoding["target_spike_loss_weight"]),
        "target_timing_tolerance_steps": int(encoding["target_timing_tolerance_steps"]),
        "output_membrane_warmup_epochs": int(stage["output_membrane_warmup_epochs"]),
        "output_membrane_aux_weight": float(stage["output_membrane_aux_weight"]),
        "checkpoint_selection": stage["checkpoint_selection"],
        "loss_reduction": "window_class_balanced",
        "lr_w": float(training["lr_w"]),
        "lr_d": float(training["lr_d"]),
        "lr_readout": float(training["lr_readout"]),
        "batch_size": int(training["batch_size"]),
        "epochs": int(training["epochs"]),
        "n_train": int(training["n_train"]),
        "n_val": int(training["n_val"]),
        "r_on": 400.0,
        "r_off": 10.0,
        "spike_penalty": 0.0,
        "delay_penalty": 0.0,
        "homeo_lambda": 0.0,
        "homeo_target": 0.0,
        "grad_clip": 1.0,
        "evaluation_split": "val",
        "test_split_opened": False,
        "smoke": smoke,
    }


def build_model(cfg: dict) -> SNNSimultaneousModel:
    return SNNSimultaneousModel(
        n_queries=1,
        n_hidden=cfg["n_hidden"],
        win_len=cfg["win_len"],
        read_len=cfg["read_len"],
        d_max=cfg["d_max"],
        train_mode=cfg["train_mode"],
        fixed_delay_value=cfg["fixed_delay_value"],
        delay_init_mode=cfg["delay_init_mode"],
        delay_init_raw=cfg["delay_init_raw"],
        delay_init_std=cfg["delay_init_std"],
        lif_tau_m=cfg["lif_tau_m"],
        lif_threshold=cfg["lif_threshold"],
        lif_reset=cfg["lif_reset"],
        lif_refractory=cfg["lif_refractory"],
        surrogate_beta=cfg["surrogate_beta"],
        n_input_channels=4,
        readout_type=cfg["readout_type"],
        use_output_spikes=True,
        n_output_neurons=2,
        lif_output_threshold=cfg["lif_output_threshold"],
        observation_mode=cfg["observation_mode"],
        opponent_output_mode=cfg["opponent_output_mode"],
        output_window_len=cfg["output_window_len"],
        output_delay_mode=cfg["output_delay_mode"],
    )


def run_stage_a_cell(protocol: dict, hidden: int, seed: int, device: str,
                     smoke: bool, dry_run: bool) -> None:
    root_kind = "smoke" if smoke else "exploratory"
    root = BASE / "runs" / root_kind / protocol["protocol_id"] / "stage_a"
    run_dir = root / f"h{hidden}_seed{seed}"
    if dry_run:
        print("[DRY]", run_dir.relative_to(BASE))
        return
    if (run_dir / "validation_results.json").exists():
        print("[SKIP]", run_dir.name)
        return

    cfg = build_stage_a_config(protocol, hidden, seed, smoke)
    set_seed(seed)
    model = build_model(cfg)
    encode_fn = partial(
        encode_simultaneous_trial,
        encoding_mode=cfg["encoding_mode"],
        burst_n_spikes_on=cfg["burst_n_spikes_on"],
        burst_n_spikes_off=cfg["burst_n_spikes_off"],
        burst_phase_on=cfg["burst_phase_on"],
        burst_phase_off=cfg["burst_phase_off"],
        burst_jitter_ms=cfg["burst_jitter_ms"],
        one_hot_phase=cfg["one_hot_phase"],
        one_hot_n_spikes=cfg["one_hot_n_spikes"],
    )

    def loader(split: str) -> DataLoader:
        dataset = FixedOperationQueryDataset(
            n_samples=cfg[f"n_{split}"],
            query_ops=cfg["query_ops"],
            seed=seed + (0 if split == "train" else 1),
        )
        return DataLoader(
            dataset, batch_size=cfg["batch_size"], shuffle=(split == "train")
        )

    trainer = SimultaneousTrainer(model, cfg, str(run_dir), device, encode_fn=encode_fn)
    trainer.save_config(cfg)
    logs = trainer.fit(loader("train"), loader("val"), cfg["epochs"])
    model.load_state_dict(torch.load(
        run_dir / "last_model.pt", map_location=device, weights_only=True
    ))
    results = evaluate_simultaneous(
        model, loader("val"), cfg, device, encode_fn=encode_fn
    )
    results.update({
        "condition": cfg["name"],
        "hidden_size": hidden,
        "query_ops": cfg["query_ops"],
        "evaluation_split": "val",
        "output_encoding": "class_opponent_one_target_spike",
    })
    save_eval_results(results, str(run_dir / "validation_results.json"))

    truth_loader = DataLoader(
        ExhaustiveFixedOperationQueryDataset(cfg["query_ops"]),
        batch_size=cfg["batch_size"], shuffle=False,
    )
    truth = evaluate_simultaneous(
        model, truth_loader, cfg, device,
        encode_fn=encode_fn, return_trial_records=True,
    )
    truth.update({
        "condition": cfg["name"],
        "hidden_size": hidden,
        "query_ops": cfg["query_ops"],
        "evaluation_split": "exhaustive_truth_table",
    })
    save_eval_results(truth, str(run_dir / "exhaustive_truth_table_results.json"))
    save_run_diagnostic_plots(
        model, cfg, logs, results, str(run_dir), 1, "XOR", device, seed=seed + 10000
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a"], default="a")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hidden", type=int, default=None,
                        help="Restrict smoke execution to one preregistered hidden size")
    parser.add_argument("--seed", type=int, default=None,
                        help="Restrict smoke execution to one preregistered seed")
    args = parser.parse_args()

    protocol = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    launchable_statuses = {"preregistered_ready", "formal_running"}
    if (not args.dry_run and not args.smoke
            and protocol["stage_a"]["status"] not in launchable_statuses):
        raise SystemExit("Stage A is not launch-ready")
    stage = protocol["stage_a"]
    hidden_sizes = [args.hidden] if args.hidden is not None else stage["hidden_sizes"]
    seeds = [args.seed] if args.seed is not None else stage["seeds"]
    if any(value not in stage["hidden_sizes"] for value in hidden_sizes):
        raise SystemExit("--hidden must be preregistered")
    if any(value not in stage["seeds"] for value in seeds):
        raise SystemExit("--seed must be preregistered")
    if (args.hidden is not None or args.seed is not None) and not args.smoke and not args.dry_run:
        raise SystemExit("formal Stage A cannot be selectively executed")

    print(json.dumps({
        "protocol": protocol["protocol_id"], "stage": "a",
        "cells": len(hidden_sizes) * len(seeds), "smoke": args.smoke,
        "test_split_opened": False,
    }))
    for hidden in hidden_sizes:
        for seed in seeds:
            run_stage_a_cell(protocol, int(hidden), int(seed), args.device,
                             args.smoke, args.dry_run)


if __name__ == "__main__":
    main()

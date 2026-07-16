"""Run the preregistered exploratory MLP Pareto scaffold.

This is deliberately separate from the failed direct-spiking-output Phase-0
Stage A. It diagnoses hidden dynamics and resource scaling; it cannot unlock
formal Stage B.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from data.boolean_dataset import ExhaustiveFixedOperationQueryDataset
from data.encoding import encode_simultaneous_trial
from snn.model import SNNSpatialParallelModel, SNNSimultaneousModel
from train.eval import evaluate_simultaneous, save_eval_results
from train.trainer import SimultaneousTrainer
from utils.seed import set_seed
from utils.viz import save_run_diagnostic_plots


BASE = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE / "configs/spatial_vs_temporal_pareto_mlp_scaffold_v2.yaml"
PROTOCOL = "spatial_vs_temporal_pareto_mlp_scaffold_v2"


def _deep_merge(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_protocol() -> dict:
    amendment = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    parent_path = BASE / "configs" / amendment["parent_config"]
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    metadata = {
        key: value for key, value in amendment.items()
        if key not in {"parent_config", "overrides"}
    }
    protocol = _deep_merge(parent, metadata)
    protocol = _deep_merge(protocol, amendment.get("overrides", {}))
    protocol["parent_config"] = amendment["parent_config"]
    return protocol


class RepeatedDataset(Dataset):
    """Repeat an exact truth table without changing its within-batch balance."""

    def __init__(self, base: Dataset, repeats: int):
        if repeats < 1:
            raise ValueError("repeats must be positive")
        self.base = base
        self.repeats = int(repeats)

    def __len__(self) -> int:
        return len(self.base) * self.repeats

    def __getitem__(self, index: int):
        return self.base[index % len(self.base)]


def _grid(protocol: dict, smoke: bool) -> tuple[list[str], list[int], list[int], list[int]]:
    surface = protocol["surface"]
    if smoke:
        point = protocol["smoke_gate"]["point"]
        return (
            list(protocol["conditions"]),
            [int(point["T"])],
            [int(point["hidden_width"])],
            [int(point["seed"])],
        )
    return (
        list(protocol["conditions"]),
        [int(value) for value in surface["total_latency_steps"]],
        [int(value) for value in surface["hidden_widths"]],
        [int(value) for value in surface["seeds"]],
    )


def build_config(
    protocol: dict,
    condition: str,
    total_latency: int,
    hidden_width: int,
    seed: int,
    smoke: bool,
) -> dict:
    K = int(protocol["scientific_scope"]["K"])
    win_len = int(protocol["surface"]["input_window_steps"])
    read_len = int(total_latency) - win_len
    if read_len <= 0 or read_len % K:
        raise ValueError("T-input_window_steps must be positive and divisible by K")
    output_window_len = read_len // K
    temporal = condition.startswith("shared_temporal_")
    spatial_independent = condition == "spatial_independent_d0"
    shared_spatial = condition == "shared_spatial_d0"
    if not (temporal or spatial_independent or shared_spatial):
        raise ValueError(condition)

    neuron = protocol["neuron"]
    delay = protocol["delay"]
    training = protocol["training"]
    encoding = protocol["encoding"]
    is_wad = condition == "shared_temporal_wad"
    is_oracle = condition == "shared_temporal_oracle"
    d_max = output_window_len if (is_wad or is_oracle) else 0
    train_mode = "weights_and_delays" if is_wad else "weights_only"
    observation_mode = "windowed_shared" if temporal else "all_time"
    epochs = int(protocol["smoke_gate"]["epochs"] if smoke else training["epochs"])
    repeats = 2 if smoke else int(training["exact_truth_table_repeats_per_epoch"])
    total_hidden = K * hidden_width if spatial_independent else hidden_width
    return {
        "protocol_id": PROTOCOL,
        "protocol_stage": "exploratory_mlp_scaffold",
        "study_class": protocol["study_class"],
        "experiment": PROTOCOL,
        "name": condition,
        "condition": condition,
        "seed": seed,
        "smoke": smoke,
        "K": K,
        "query_ops": ["XOR"] * K,
        "ops_list": ["XOR"],
        "input_schedule": "simultaneous",
        "n_input": 4 * K,
        "surface_hidden_width": hidden_width,
        "hidden_width_semantics": (
            "per_query_hidden_width_total_Kh" if spatial_independent
            else "shared_hidden_total_hprime"
        ),
        "n_hidden": total_hidden,
        "n_hidden_total": total_hidden,
        "topology_type": (
            "spatial_independent_shared_decoder" if spatial_independent
            else "shared_dense"
        ),
        "win_len": win_len,
        "read_len": read_len,
        "T": total_latency,
        "output_window_len": output_window_len,
        "d_max": d_max,
        "train_mode": train_mode,
        "fixed_delay_value": None if is_wad else 0.0,
        "fixed_delay_distribution": None,
        "shared_delay": False,
        "oracle_delay_schedule": (
            [0.0, float(output_window_len)] if is_oracle else None
        ),
        "delay_placement": "input_to_hidden_only",
        "delay_param_type": delay["parameterization"],
        "delay_step": float(delay["step"]),
        "delay_init_mode": delay["init_mode"],
        "delay_init_raw": float(delay["init_raw"]),
        "delay_init_std": float(delay["init_std"]),
        "optimization_schedule": "joint",
        "lif_tau_m": float(neuron["tau_m"]),
        "lif_threshold": float(neuron["threshold"]),
        "lif_reset": float(neuron["reset"]),
        "lif_refractory": int(neuron["refractory_steps"]),
        "surrogate_beta": float(neuron["surrogate_beta"]),
        "dt": 1.0,
        "readout_type": "mlp",
        "readout_endpoint": (
            "shared_mlp_per_independent_module" if spatial_independent
            else ("shared_mlp_per_time_window" if temporal else "all_time_K_output_mlp")
        ),
        "observation_mode": observation_mode,
        "use_output_spikes": False,
        "output_encoding": "nonspiking_mlp_logits",
        "lr_w": float(training["lr_w"]),
        "lr_d": float(training["lr_d"]),
        "lr_readout": float(training["lr_readout"]),
        "batch_size": int(training["batch_size"]),
        "epochs": epochs,
        "exact_truth_table_repeats_per_epoch": repeats,
        "n_train": 16 * repeats,
        "n_val": 16,
        "loss_reduction": "pooled_bce_balanced_exhaustive_batches",
        "checkpoint_selection": training["checkpoint_selection"],
        "r_on": 400.0,
        "r_off": 10.0,
        "spike_penalty": 0.0,
        "delay_penalty": 0.0,
        "homeo_lambda": 0.0,
        "homeo_target": 0.0,
        "grad_clip": float(training["grad_clip"]),
        "encoding_mode": encoding["mode"],
        "one_hot_phase": float(encoding["one_hot_phase"]),
        "one_hot_n_spikes": int(encoding["spikes_per_selected_value_channel"]),
        "burst_n_spikes_on": 1,
        "burst_n_spikes_off": 1,
        "burst_phase_on": 1.0,
        "burst_phase_off": 1.0,
        "burst_jitter_ms": int(encoding["jitter_ms"]),
        "evaluation_split": "exhaustive_validation",
        "test_split_opened": False,
        "formal_phase0_stage_b_unlocked": False,
    }


def build_model(cfg: dict):
    if cfg["name"] == "spatial_independent_d0":
        model = SNNSpatialParallelModel(
            n_queries=cfg["K"],
            hidden_per_query=cfg["surface_hidden_width"],
            win_len=cfg["win_len"],
            read_len=cfg["read_len"],
            d_max=cfg["d_max"],
            train_mode=cfg["train_mode"],
            fixed_delay_value=cfg["fixed_delay_value"],
            lif_tau_m=cfg["lif_tau_m"],
            lif_threshold=cfg["lif_threshold"],
            lif_reset=cfg["lif_reset"],
            lif_refractory=cfg["lif_refractory"],
            dt=cfg["dt"],
            surrogate_beta=cfg["surrogate_beta"],
            readout_type=cfg["readout_type"],
        )
    else:
        model = SNNSimultaneousModel(
            n_queries=cfg["K"],
            n_hidden=cfg["surface_hidden_width"],
            win_len=cfg["win_len"],
            read_len=cfg["read_len"],
            d_max=cfg["d_max"],
            train_mode=cfg["train_mode"],
            delay_param_type=cfg["delay_param_type"],
            delay_step=cfg["delay_step"],
            fixed_delay_value=cfg["fixed_delay_value"],
            delay_init_mode=cfg["delay_init_mode"],
            delay_init_raw=cfg["delay_init_raw"],
            delay_init_std=cfg["delay_init_std"],
            lif_tau_m=cfg["lif_tau_m"],
            lif_threshold=cfg["lif_threshold"],
            lif_reset=cfg["lif_reset"],
            lif_refractory=cfg["lif_refractory"],
            dt=cfg["dt"],
            surrogate_beta=cfg["surrogate_beta"],
            n_input_channels=cfg["n_input"],
            readout_type=cfg["readout_type"],
            use_output_spikes=False,
            observation_mode=cfg["observation_mode"],
            output_window_len=cfg["output_window_len"],
        )
        if cfg["name"] == "shared_temporal_oracle":
            schedule = torch.tensor(
                cfg["oracle_delay_schedule"], dtype=model.syn_ih.weight.dtype
            )
            row_delays = torch.repeat_interleave(schedule, repeats=4)
            with torch.no_grad():
                model.syn_ih.fixed_delay_tensor = row_delays[:, None].expand(
                    model.n_input, model.n_hidden
                ).clone()
    return model


def _encode_fn(cfg: dict):
    return partial(
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


def _loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    truth = ExhaustiveFixedOperationQueryDataset(cfg["query_ops"])
    repeated = RepeatedDataset(truth, cfg["exact_truth_table_repeats_per_epoch"])
    train = DataLoader(
        repeated, batch_size=cfg["batch_size"], shuffle=False, drop_last=False
    )
    validation = DataLoader(
        truth, batch_size=cfg["batch_size"], shuffle=False, drop_last=False
    )
    return train, validation


def run_cell(
    protocol: dict,
    condition: str,
    total_latency: int,
    hidden_width: int,
    seed: int,
    device: str,
    smoke: bool,
    dry_run: bool,
) -> None:
    root_kind = "smoke" if smoke else "exploratory"
    run_dir = (
        BASE / "runs" / root_kind / PROTOCOL / condition
        / f"T{total_latency}_h{hidden_width}_seed{seed}"
    )
    if dry_run:
        print("[DRY]", run_dir.relative_to(BASE))
        return
    required = [
        "validation_results.json", "exhaustive_truth_table_results.json",
        "resource_ledger.json", "plots/diagnostic_data.npz",
        "plots/diagnostic_panel.png",
    ]
    if all((run_dir / path).exists() for path in required):
        print("[SKIP]", run_dir.relative_to(BASE))
        return

    cfg = build_config(protocol, condition, total_latency, hidden_width, seed, smoke)
    set_seed(seed)
    model = build_model(cfg)
    train_loader, validation_loader = _loaders(cfg)
    encode_fn = _encode_fn(cfg)
    trainer = SimultaneousTrainer(
        model, cfg, str(run_dir), device=device, encode_fn=encode_fn
    )
    trainer.save_config(cfg)
    logs = trainer.fit(train_loader, validation_loader, cfg["epochs"])
    selected_checkpoint = (
        "best_model.pt" if cfg["checkpoint_selection"] == "best_pooled_accuracy"
        else "last_model.pt"
    )
    model.load_state_dict(torch.load(
        run_dir / selected_checkpoint, map_location=device, weights_only=True
    ))
    results = evaluate_simultaneous(
        model, validation_loader, cfg, device, encode_fn=encode_fn,
        return_trial_records=True,
    )
    results.update({
        "condition": condition,
        "total_latency_steps": total_latency,
        "surface_hidden_width": hidden_width,
        "hidden_width_semantics": cfg["hidden_width_semantics"],
        "evaluation_split": "exhaustive_validation",
        "query_ops": cfg["query_ops"],
        "output_encoding": cfg["output_encoding"],
        "selected_checkpoint": selected_checkpoint,
        "test_split_opened": False,
        "formal_phase0_stage_b_unlocked": False,
    })
    save_eval_results(results, str(run_dir / "validation_results.json"))
    truth = dict(results)
    truth["evaluation_split"] = "exhaustive_truth_table"
    save_eval_results(truth, str(run_dir / "exhaustive_truth_table_results.json"))
    (run_dir / "resource_ledger.json").write_text(
        json.dumps(results["resource_ledger"], indent=2) + "\n", encoding="utf-8"
    )
    save_run_diagnostic_plots(
        model, cfg, logs, results, str(run_dir), cfg["K"], "XOR", device,
        seed=seed + 10000,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--condition", choices=[
        "spatial_independent_d0", "shared_spatial_d0", "shared_temporal_d0",
        "shared_temporal_oracle", "shared_temporal_wad",
    ])
    parser.add_argument("--T", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    protocol = load_protocol()
    conditions, latencies, widths, seeds = _grid(protocol, args.smoke)
    if args.condition is not None:
        conditions = [args.condition]
    if args.T is not None:
        if args.T not in protocol["surface"]["total_latency_steps"]:
            raise SystemExit("--T must be preregistered")
        latencies = [args.T]
    if args.hidden is not None:
        if args.hidden not in protocol["surface"]["hidden_widths"]:
            raise SystemExit("--hidden must be preregistered")
        widths = [args.hidden]
    if args.seed is not None:
        if args.seed not in protocol["surface"]["seeds"]:
            raise SystemExit("--seed must be preregistered")
        seeds = [args.seed]
    selective = any(value is not None for value in (
        args.condition, args.T, args.hidden, args.seed
    ))
    if selective and not (args.smoke or args.dry_run):
        raise SystemExit("formal exploratory grid cannot be selectively executed")
    launchable_statuses = {"preregistered_ready", "formal_exploratory_running"}
    if (not args.smoke and not args.dry_run
            and protocol["status"] not in launchable_statuses):
        raise SystemExit(
            f"protocol status is {protocol['status']!r}; pass the smoke gate first"
        )

    cells = len(conditions) * len(latencies) * len(widths) * len(seeds)
    print(json.dumps({
        "protocol": PROTOCOL, "cells": cells, "smoke": args.smoke,
        "direct_spiking_output": False, "formal_stage_b_unlocked": False,
    }))
    for condition in conditions:
        for total_latency in latencies:
            for hidden_width in widths:
                for seed in seeds:
                    run_cell(
                        protocol, condition, total_latency, hidden_width, seed,
                        args.device, args.smoke, args.dry_run,
                    )


if __name__ == "__main__":
    main()

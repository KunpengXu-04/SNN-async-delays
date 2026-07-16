"""Run locked spatial-control or true temporal-routing simultaneous pilot."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data.boolean_dataset import FixedOperationQueryDataset, ExhaustiveFixedOperationQueryDataset
from data.encoding import encode_simultaneous_trial
from snn.model import SNNSimultaneousModel
from train.eval import evaluate_simultaneous, save_eval_results
from train.trainer import SimultaneousTrainer
from utils.seed import set_seed
from utils.viz import save_run_diagnostic_plots


BASE = Path(__file__).resolve().parents[1]


def delay_condition(name: str, seed: int, matched_range: list[float]) -> dict:
    common = {"fixed_delay_value": None, "fixed_delay_distribution": None,
              "shared_delay": False}
    if name == "d0": return {**common, "name": name, "train_mode": "weights_only", "fixed_delay_value": 0.0}
    if name == "temporal_scaffold":
        return {**common, "name": name, "train_mode": "weights_only", "fixed_delay_value": 0.0}
    if name == "scalar": return {**common, "name": name, "train_mode": "weights_and_delays", "shared_delay": True}
    if name == "w_and_d": return {**common, "name": name, "train_mode": "weights_and_delays"}
    if name == "fixed_matched":
        return {**common, "name": name, "train_mode": "weights_only",
                "fixed_delay_distribution": "uniform", "fixed_delay_seed": seed,
                "fixed_delay_low": matched_range[0], "fixed_delay_high": matched_range[1]}
    if name == "fixed_full_support":
        return {**common, "name": name, "train_mode": "weights_only",
                "fixed_delay_distribution": "uniform", "fixed_delay_seed": seed,
                "fixed_delay_low": 0.0, "fixed_delay_high": None}
    raise ValueError(name)


def endpoint_spec(endpoint: str, K: int, output_window_len: int | None) -> dict:
    if endpoint == "linear": return {"readout_type": "linear", "observation_mode": "all_time"}
    if endpoint == "mlp": return {"readout_type": "mlp", "observation_mode": "all_time"}
    if endpoint == "shared_linear": return {"readout_type": "linear", "observation_mode": "windowed_shared", "output_window_len": output_window_len}
    if endpoint == "shared_mlp": return {"readout_type": "mlp", "observation_mode": "windowed_shared", "output_window_len": output_window_len}
    if endpoint == "opponent_parallel":
        return {"readout_type": "linear", "observation_mode": "all_time",
                "use_output_spikes": True, "opponent_output_mode": "parallel_pairs",
                "n_output_neurons": 2 * K}
    if endpoint == "opponent_shared_windowed":
        return {"readout_type": "linear", "observation_mode": "windowed_shared",
                "use_output_spikes": True, "opponent_output_mode": "shared_windowed",
                "n_output_neurons": 2, "output_window_len": output_window_len}
    raise ValueError(endpoint)


def run_cell(protocol: dict, endpoint: str, condition_name: str, seed: int,
             root: Path, device: str, dry_run: bool) -> None:
    frozen = protocol["frozen_wad_config"]
    K, hidden = int(protocol["K"]), int(protocol["hidden_size"])
    cond = delay_condition(condition_name, seed, protocol["fixed_matched_delay_range"])
    ep = endpoint_spec(endpoint, K, protocol.get("output_window_len"))
    run_name = f"{condition_name}_{endpoint}_seed{seed}"
    run_dir = root / run_name
    result_path = run_dir / "validation_results.json"
    if dry_run:
        try:
            display_path = run_dir.relative_to(BASE)
        except ValueError:
            display_path = run_dir
        print("[DRY]", display_path); return
    if result_path.exists():
        print("[SKIP]", run_name); return

    set_seed(seed)
    win_len, read_len = int(protocol["input_win"]), int(protocol["read_len"])
    training, encoding = protocol["training"], protocol["encoding"]
    cfg = {
        "protocol_id": protocol["protocol_id"], "experiment": protocol["protocol_id"],
        "seed": seed,
        "input_schedule": "simultaneous", "query_ops": protocol["query_ops"],
        "K": K, "n_hidden": hidden, "n_input": 2 * K,
        "win_len": win_len, "read_len": read_len, "sub_win": None,
        "d_max": int(frozen["d_max"]), "lif_threshold": float(frozen["threshold"]),
        "lr_d": float(frozen["lr_d"]), "delay_init_mode": frozen["delay_init_mode"],
        "optimization_schedule": frozen["optimization_schedule"],
        "lif_tau_m": 10.0, "lif_reset": 0.0, "lif_refractory": 2,
        "surrogate_beta": 4.0, "delay_param_type": "sigmoid", "delay_step": 1.0,
        "delay_init_raw": -2.0, "delay_init_std": 0.25,
        "dt": 1.0, "lr_w": training["lr_w"], "lr_readout": training["lr_readout"],
        "batch_size": training["batch_size"], "epochs": training["epochs"],
        "n_train": training["n_train"], "n_val": training["n_val"],
        "r_on": 400.0, "r_off": 10.0, "spike_penalty": 0.0,
        "delay_penalty": 0.0, "homeo_lambda": 0.0, "homeo_target": 0.0,
        "grad_clip": 1.0, "encoding_mode": encoding["mode"],
        "burst_n_spikes_on": encoding["n_spikes_on"],
        "burst_n_spikes_off": encoding["n_spikes_off"],
        "burst_phase_on": encoding["phase_on"], "burst_phase_off": encoding["phase_off"],
        "burst_jitter_ms": encoding["jitter_ms"], "readout_endpoint": endpoint,
        "evaluation_split": "val", "test_split_opened": False,
        "lif_output_threshold": protocol.get("frozen_output_threshold"),
        "output_membrane_warmup_epochs": int(protocol.get("output_membrane_warmup_epochs", 0)),
        "output_membrane_aux_weight": float(protocol.get("output_membrane_aux_weight", 0.2)),
        "loss_reduction": protocol.get("loss_reduction", "pooled_bce"),
        "checkpoint_selection": protocol.get("checkpoint_selection", "best_pooled_accuracy"),
        "input_hidden_delay_scaffold": protocol.get("input_hidden_delay_scaffold"),
        **ep, **{k:v for k,v in cond.items() if k != "name"}, "name": condition_name,
    }
    model = SNNSimultaneousModel(
        n_queries=K, n_hidden=hidden, win_len=win_len, read_len=read_len,
        d_max=cfg["d_max"], train_mode=cfg["train_mode"],
        fixed_delay_value=cfg.get("fixed_delay_value"),
        fixed_delay_distribution=cfg.get("fixed_delay_distribution"),
        fixed_delay_seed=cfg.get("fixed_delay_seed", seed),
        fixed_delay_low=cfg.get("fixed_delay_low", 0.0),
        fixed_delay_high=cfg.get("fixed_delay_high"), shared_delay=cfg.get("shared_delay", False),
        delay_init_mode=cfg["delay_init_mode"], delay_init_raw=cfg["delay_init_raw"],
        delay_init_std=cfg["delay_init_std"], lif_tau_m=cfg["lif_tau_m"],
        lif_threshold=cfg["lif_threshold"], lif_reset=cfg["lif_reset"],
        lif_refractory=cfg["lif_refractory"], surrogate_beta=cfg["surrogate_beta"],
        n_input_channels=2*K, readout_type=cfg["readout_type"],
        use_output_spikes=cfg.get("use_output_spikes", False),
        n_output_neurons=cfg.get("n_output_neurons"),
        lif_output_threshold=cfg["lif_output_threshold"],
        observation_mode=cfg["observation_mode"],
        opponent_output_mode=cfg.get("opponent_output_mode"),
        output_window_len=cfg.get("output_window_len"),
    )
    scaffold = cfg.get("input_hidden_delay_scaffold")
    if scaffold is not None:
        row_delays = torch.tensor(scaffold["per_input_channel_delays"],
                                  dtype=model.syn_ih.weight.dtype,
                                  device=model.syn_ih.weight.device)
        if row_delays.numel() != model.n_input:
            raise ValueError("input_hidden_delay_scaffold must specify one delay per input channel")
        if float(row_delays.min()) < 0 or float(row_delays.max()) > model.d_max:
            raise ValueError("input_hidden_delay_scaffold lies outside [0,d_max]")
        with torch.no_grad():
            model.syn_ih.fixed_delay_tensor = row_delays[:, None].expand(
                model.n_input, model.n_hidden
            ).clone()
    encode_fn = partial(
        encode_simultaneous_trial, encoding_mode=cfg["encoding_mode"],
        burst_n_spikes_on=cfg["burst_n_spikes_on"],
        burst_n_spikes_off=cfg["burst_n_spikes_off"],
        burst_phase_on=cfg["burst_phase_on"], burst_phase_off=cfg["burst_phase_off"],
        burst_jitter_ms=cfg["burst_jitter_ms"],
    )
    def loader(split: str):
        ds = FixedOperationQueryDataset(
            n_samples=cfg[f"n_{split}"], query_ops=cfg["query_ops"],
            seed=seed + (0 if split == "train" else 1),
        )
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=(split == "train"))
    trainer = SimultaneousTrainer(model, cfg, str(run_dir), device, encode_fn=encode_fn)
    trainer.save_config(cfg)
    logs = trainer.fit(loader("train"), loader("val"), cfg["epochs"])
    checkpoint_name = ("last_model.pt" if cfg["checkpoint_selection"] == "final"
                       else "best_model.pt")
    model.load_state_dict(torch.load(run_dir / checkpoint_name, map_location=device, weights_only=True))
    results = evaluate_simultaneous(model, loader("val"), cfg, device, encode_fn=encode_fn)
    semantic_prefix = "operation" if protocol["protocol_id"].startswith("simultaneous_spatial") else "window"
    results.update({"condition": condition_name, "endpoint": endpoint,
                    "query_ops": cfg["query_ops"], "evaluation_split": "val",
                    f"per_{semantic_prefix}_accuracy": results["per_query_acc"],
                    f"worst_{semantic_prefix}_accuracy": results["worst_query_accuracy"],
                    f"per_{semantic_prefix}_balanced_accuracy": results["per_query_balanced_accuracy"],
                    f"worst_{semantic_prefix}_balanced_accuracy": results["worst_query_balanced_accuracy"]})
    save_eval_results(results, str(result_path))
    truth_loader = DataLoader(
        ExhaustiveFixedOperationQueryDataset(cfg["query_ops"]),
        batch_size=cfg["batch_size"], shuffle=False,
    )
    truth = evaluate_simultaneous(
        model, truth_loader, cfg, device, encode_fn=encode_fn, return_trial_records=True,
    )
    truth.update({"condition": condition_name, "endpoint": endpoint,
                  "query_ops": cfg["query_ops"], "evaluation_split": "exhaustive_truth_table"})
    save_eval_results(truth, str(run_dir / "exhaustive_truth_table_results.json"))
    save_run_diagnostic_plots(model, cfg, logs, results, str(run_dir), K,
                              "mixed", device, seed=999)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", choices=["spatial", "temporal"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    filename = ("simultaneous_spatial_control_pilot_v1.yaml" if args.pilot == "spatial"
                else "simultaneous_temporal_routing_pilot_v1.yaml")
    protocol = yaml.safe_load((BASE / "configs" / filename).read_text(encoding="utf-8"))
    ready_statuses = {"preregistered_ready", "preregistered_ready_after_output_calibration"}
    if protocol.get("status") not in ready_statuses and not args.dry_run:
        raise SystemExit(
            f"Pilot status is {protocol.get('status')!r}, not launch-ready; "
            "complete the declared gate before changing the status"
        )
    missing = [k for k,v in protocol["frozen_wad_config"].items() if v is None]
    if protocol.get("frozen_output_threshold") is None:
        missing.append("frozen_output_threshold")
    if missing and not args.dry_run:
        raise SystemExit(f"Pilot locked pending audit fields: {missing}")
    root = BASE / "runs" / "exploratory" / protocol["protocol_id"]
    total = len(protocol["conditions"]) * len(protocol["endpoints"]) * len(protocol["seeds"])
    print(f"{protocol['protocol_id']}: {total} cells; locked fields={missing}")
    for endpoint in protocol["endpoints"]:
        for seed in protocol["seeds"]:
            for condition in protocol["conditions"]:
                run_cell(protocol, endpoint, condition, seed, root, args.device, args.dry_run)


if __name__ == "__main__":
    main()

"""Versioned checkpoint re-evaluation for sequential Plan-D runs.

This tool never overwrites `runs/**/eval_results.json`.  It reconstructs a
saved checkpoint and writes a separate aggregate report under `docs/generated`.
Version 1 intentionally supports only `SNNSimultaneousModel` sequential
Plan-D runs; unsupported schemas are reported rather than guessed.

Example:
    python -m scripts.reevaluate_checkpoints \
      --runs-dir 'runs/NAND_compress_burst_(planD)' --device cpu
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.boolean_dataset import MultiQueryDataset
from data.encoding import encode_sequential_trial
from snn.model import SNNSimultaneousModel
from train.eval import evaluate_simultaneous
from utils.seed import set_seed


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "docs" / "generated"
PROTOCOL_VERSION = "checkpoint_reevaluation_v1"


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def is_sequential_plan_d(cfg: dict[str, Any], run_dir: Path) -> bool:
    experiment = str(cfg.get("experiment", "")).lower()
    return (
        "pland" in experiment
        or "plan_d" in experiment
        or "pland" in str(run_dir.parent).lower()
        or cfg.get("n_input") == 2 and "sub_win" in cfg
    )


def model_from_config(cfg: dict[str, Any]) -> SNNSimultaneousModel:
    K = int(cfg["K"])
    return SNNSimultaneousModel(
        n_queries=K,
        n_hidden=int(cfg["n_hidden"]),
        win_len=int(cfg["win_len"]),
        read_len=int(cfg["read_len"]),
        d_max=int(cfg["d_max"]),
        train_mode=cfg.get("train_mode", "weights_and_delays"),
        delay_param_type=cfg.get("delay_param_type", "sigmoid"),
        delay_step=float(cfg.get("delay_step", 1.0)),
        fixed_delay_value=cfg.get("fixed_delay_value"),
        lif_tau_m=float(cfg.get("lif_tau_m", 10.0)),
        lif_threshold=float(cfg.get("lif_threshold", 1.0)),
        lif_reset=float(cfg.get("lif_reset", 0.0)),
        lif_refractory=int(cfg.get("lif_refractory", 2)),
        dt=float(cfg.get("dt", 1.0)),
        surrogate_beta=float(cfg.get("surrogate_beta", 4.0)),
        n_input_channels=cfg.get("n_input", cfg.get("n_input_channels", 2)),
        readout_type=cfg.get("readout_type", "linear"),
        num_hidden_layers=int(cfg.get("num_hidden_layers", 1)),
        hidden_sizes=cfg.get("hidden_sizes"),
        use_output_spikes=bool(cfg.get("use_output_spikes", False)),
        n_output_neurons=cfg.get("n_output_neurons"),
        lif_output_threshold=cfg.get("lif_output_threshold"),
        observation_mode=cfg.get("observation_mode", "late_window"),
    )


def encoder(cfg: dict[str, Any]):
    return partial(
        encode_sequential_trial,
        encoding_mode=cfg.get("encoding_mode", "rate"),
        burst_n_spikes_on=int(cfg.get("burst_n_spikes_on", 2)),
        burst_n_spikes_off=int(cfg.get("burst_n_spikes_off", 1)),
        burst_phase_on=float(cfg.get("burst_phase_on", 0.2)),
        burst_phase_off=float(cfg.get("burst_phase_off", 0.8)),
        burst_jitter_ms=int(cfg.get("burst_jitter_ms", 1)),
    )


def reevaluate(run_dir: Path, device: str, evaluation_seed: int) -> dict[str, Any]:
    cfg = load_json(run_dir / "config.json")
    relative = run_dir.relative_to(BASE).as_posix()
    if not cfg or not (run_dir / "best_model.pt").exists():
        return {"run_path": relative, "status": "skip", "reason": "missing config or checkpoint"}
    if not is_sequential_plan_d(cfg, run_dir):
        return {"run_path": relative, "status": "skip", "reason": "unsupported non-Plan-D schema"}

    try:
        K = int(cfg["K"])
        set_seed(evaluation_seed)
        model = model_from_config(cfg).to(device)
        state = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        dataset = MultiQueryDataset(
            K=K,
            n_samples=int(cfg.get("n_test", 1000)),
            same_op=bool(cfg.get("same_op", True)),
            op_name=cfg.get("op_name", "NAND"),
            ops_list=cfg.get("ops_list"),
            seed=int(cfg.get("seed", 0)) + 2,
        )
        loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 128)), shuffle=False)
        result = evaluate_simultaneous(model, loader, cfg, device, encode_fn=encoder(cfg), K_query=K)
        return {
            "run_path": relative,
            "status": "ok",
            "protocol_version": PROTOCOL_VERSION,
            "evaluation_seed": evaluation_seed,
            "task_interface": "sequential_plan_d_late_window",
            "result": result,
        }
    except Exception as exc:  # report unsupported historical variance explicitly
        return {"run_path": relative, "status": "error", "reason": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True, help="Path relative to snn_async_delays or absolute")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluation-seed", type=int, default=20260711)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--run-glob", action="append", default=[],
        help="Optional run-directory glob; may be supplied more than once for a pilot set",
    )
    parser.add_argument("--output", default=None, help="Output JSON path under docs/generated by default")
    args = parser.parse_args()

    root = Path(args.runs_dir)
    if not root.is_absolute():
        root = BASE / root
    run_dirs = sorted({path.parent for path in root.rglob("best_model.pt")})
    if args.run_glob:
        run_dirs = [
            path for path in run_dirs
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in args.run_glob)
        ]
    if args.max_runs is not None:
        run_dirs = run_dirs[:args.max_runs]

    records = [reevaluate(path, args.device, args.evaluation_seed) for path in run_dirs]
    output = Path(args.output) if args.output else OUT / f"{root.name}_{PROTOCOL_VERSION}.json"
    if not output.is_absolute():
        output = BASE / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
    counts = {status: sum(row["status"] == status for row in records) for status in ("ok", "skip", "error")}
    print(json.dumps({"output": str(output), "n_runs": len(records), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()

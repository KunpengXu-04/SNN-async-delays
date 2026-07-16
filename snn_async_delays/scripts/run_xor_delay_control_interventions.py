"""Run preregistered post-training interventions for XOR control matrix v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.boolean_dataset import MultiQueryDataset
from data.encoding import encode_sequential_trial
from scripts.generate_diagnostic_plots import _load_log_rows, _load_model_from_run
from scripts.run_plan_d_h_sweep import build_encode_fn
from train.eval import evaluate_simultaneous, save_eval_results
from utils.delay_controls import shuffle_delay_parameters_
from utils.seed import set_seed
from utils.viz import save_run_diagnostic_plots


BASE = Path(__file__).resolve().parents[1]
PROTOCOL = "xor_delay_control_matrix_v1"


def validation_loader(cfg: dict) -> DataLoader:
    seed = int(cfg["seed"])
    dataset = MultiQueryDataset(
        K=int(cfg["K"]), n_samples=int(cfg["n_val"]), same_op=True,
        op_name="XOR", ops_list=["XOR"], seed=seed + 1,
    )
    return DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False)


def evaluate_intervention(run_dir: Path, name: str, device: str, force: bool = False) -> str:
    output_dir = run_dir / "interventions" / name
    result_path = output_dir / "validation_results.json"
    if result_path.exists() and not force:
        return "skip"

    model, cfg = _load_model_from_run(str(run_dir), device)
    if cfg.get("protocol_id") != PROTOCOL:
        return "skip"
    intervention = {"name": name, "source_checkpoint": "best_model.pt"}
    if name == "shuffle_learned_delays":
        if cfg.get("name") != "w_and_d":
            return "skip"
        intervention.update(shuffle_delay_parameters_(model, 10000 + int(cfg["seed"])))
    elif name == "late_window_probe":
        if cfg.get("observation_mode") != "all_time":
            raise ValueError("late-window probe requires an all-time-trained checkpoint")
        model.observation_mode = "late_window"
        cfg = {**cfg, "observation_mode": "late_window"}
        intervention["decoder_training_observation"] = "all_time"
        intervention["evaluation_observation"] = "late_window"
    else:
        raise ValueError(name)

    set_seed(int(cfg["seed"]))
    encode_fn = build_encode_fn(encode_sequential_trial, cfg)
    results = evaluate_simultaneous(
        model, validation_loader(cfg), cfg, device, encode_fn=encode_fn,
    )
    results.update({
        "protocol_id": PROTOCOL,
        "evaluation_split": "val",
        "post_training_intervention": intervention,
        "test_split_opened": False,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    save_eval_results(results, str(result_path))
    (output_dir / "source_run.json").write_text(json.dumps({
        "source_run": run_dir.relative_to(BASE).as_posix(),
        "intervention": intervention,
    }, indent=2), encoding="utf-8")
    save_run_diagnostic_plots(
        model=model, cfg=cfg, log_rows=_load_log_rows(str(run_dir)),
        eval_results=results, run_dir=str(output_dir), K=int(cfg["K"]),
        op="XOR", device=device, seed=999,
    )
    return "done"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=f"runs/exploratory/{PROTOCOL}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate and refresh existing intervention results/diagnostics")
    args = parser.parse_args()
    root = Path(args.runs_dir)
    if not root.is_absolute():
        root = BASE / root
    run_dirs = sorted({p.parent for p in root.rglob("validation_results.json")
                       if "interventions" not in p.parts})
    expected = 60
    if len(run_dirs) != expected and not args.dry_run:
        raise SystemExit(f"Expected {expected} completed training cells, found {len(run_dirs)}")
    print(f"source training cells={len(run_dirs)}")
    counts = {"done": 0, "skip": 0}
    for run_dir in run_dirs:
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        names = ["late_window_probe"]
        if cfg.get("name") == "w_and_d":
            names.append("shuffle_learned_delays")
        for name in names:
            if args.dry_run:
                print(run_dir.name, name)
                continue
            status = evaluate_intervention(run_dir, name, args.device, force=args.force)
            counts[status] += 1
    if not args.dry_run:
        print(counts)


if __name__ == "__main__":
    main()

"""Generate a fixed, non-cherry-picked diagnostic subset for XOR pilot v1.

Selection rule is deliberately independent of accuracy/spike richness:
lowest preregistered seed (0), MLP decoder, every delay condition and every
observation mode.  These six panels illustrate the interface treatment; the
36-cell aggregate remains the evidence source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.generate_diagnostic_plots import _load_log_rows, _load_model_from_run
from utils.viz import save_run_diagnostic_plots


BASE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "xor_readout_interface_pilot_v1"
SELECTION_RULE = {
    "seed": 0,
    "readout_type": "mlp",
    "include_every_delay_condition": True,
    "include_every_observation_mode": True,
    "reason": "lowest preregistered seed; no performance or spike-richness selection",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=f"runs/canonical/{PROTOCOL_ID}")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_root = Path(args.runs_dir)
    if not run_root.is_absolute():
        run_root = BASE / run_root
    selected = []
    for eval_path in sorted(run_root.rglob("eval_results.json")):
        run_dir = eval_path.parent
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if (
            cfg.get("protocol_id") == PROTOCOL_ID
            and cfg.get("seed") == SELECTION_RULE["seed"]
            and cfg.get("readout_type") == SELECTION_RULE["readout_type"]
        ):
            selected.append(run_dir)
    expected = 6
    if len(selected) != expected:
        raise SystemExit(f"Expected {expected} selected pilot cells, found {len(selected)}")

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "selection_rule": SELECTION_RULE,
        "selected_runs": [path.relative_to(BASE).as_posix() for path in selected],
        "evidence_boundary": (
            "Illustrative diagnostic subset only; use the full 36-cell group summary for inference."
        ),
    }
    output = BASE / "docs" / "generated" / f"{PROTOCOL_ID}_diagnostic_selection.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for index, run_dir in enumerate(selected, 1):
        model, cfg = _load_model_from_run(str(run_dir), args.device)
        logs = _load_log_rows(str(run_dir))
        results = json.loads((run_dir / "eval_results.json").read_text(encoding="utf-8"))
        print(f"[{index}/{len(selected)}] {run_dir.name}")
        save_run_diagnostic_plots(
            model=model, cfg=cfg, log_rows=logs, eval_results=results,
            run_dir=str(run_dir), K=cfg["K"], op=cfg["op_name"],
            device=args.device, seed=999,
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

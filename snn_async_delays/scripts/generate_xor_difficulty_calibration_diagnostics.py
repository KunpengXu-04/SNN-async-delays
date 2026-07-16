"""Generate the preregistered diagnostic subset for XOR calibration v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.generate_diagnostic_plots import _load_log_rows, _load_model_from_run
from utils.viz import save_run_diagnostic_plots


BASE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "xor_difficulty_calibration_v1"
SELECTION_RULE = {
    "seed": 0,
    "hidden_size": 35,
    "include_K": [2, 3, 4],
    "include_every_delay_condition": True,
    "trace_seed": 999,
    "reason": "fixed middle hidden size and lowest seed; outcome-independent",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=f"runs/exploratory/{PROTOCOL_ID}")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_root = Path(args.runs_dir)
    if not run_root.is_absolute():
        run_root = BASE / run_root

    selected = []
    for result_path in sorted(run_root.rglob("validation_results.json")):
        run_dir = result_path.parent
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if (
            cfg.get("protocol_id") == PROTOCOL_ID
            and cfg.get("seed") == SELECTION_RULE["seed"]
            and cfg.get("n_hidden") == SELECTION_RULE["hidden_size"]
            and cfg.get("K") in SELECTION_RULE["include_K"]
        ):
            selected.append(run_dir)
    if len(selected) != 6:
        raise SystemExit(f"Expected 6 preregistered diagnostic cells, found {len(selected)}")

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "selection_rule": SELECTION_RULE,
        "selected_runs": [path.relative_to(BASE).as_posix() for path in selected],
        "evidence_boundary": "Illustrative only; use the complete 54-cell validation table for calibration.",
    }
    output = BASE / "docs" / "generated" / f"{PROTOCOL_ID}_diagnostic_selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for index, run_dir in enumerate(selected, 1):
        model, cfg = _load_model_from_run(str(run_dir), args.device)
        logs = _load_log_rows(str(run_dir))
        results = json.loads((run_dir / "validation_results.json").read_text(encoding="utf-8"))
        print(f"[{index}/{len(selected)}] {run_dir.name}")
        save_run_diagnostic_plots(
            model=model, cfg=cfg, log_rows=logs, eval_results=results,
            run_dir=str(run_dir), K=cfg["K"], op=cfg["op_name"],
            device=args.device, seed=SELECTION_RULE["trace_seed"],
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

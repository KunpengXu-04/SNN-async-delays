"""Build a read-only registry of every experiment artifact under ``runs/``.

The registry is deliberately conservative: historical runs are marked
``exploratory`` unless a future protocol explicitly promotes them.  No run is
deleted, moved, or rewritten.  This is an evidence index, not a result
aggregator.

Run from ``snn_async_delays``:
    python -m scripts.build_experiment_registry
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]
RUNS = BASE / "runs"
OUT = BASE / "docs" / "generated"


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def diagnostic_artifacts_complete(run_dir: Path, metrics: dict[str, Any]) -> bool:
    """Recognize versioned mechanism cells that intentionally have no eval file."""
    return bool(
        metrics.get("complete") is True
        and (run_dir / "config.json").exists()
        and any(run_dir.glob("*.pt"))
        and (run_dir / "plots" / "diagnostic_data.npz").exists()
        and (run_dir / "plots" / "diagnostic_panel.png").exists()
    )


def infer_status(
    relative_path: str,
    *,
    has_eval: bool,
    has_metrics: bool,
    diagnostic_complete: bool,
) -> tuple[str, str]:
    """Return a conservative status and an auditable reason."""
    name = relative_path.lower()
    if "_archive" in name or "preclean" in name:
        return "archived", "legacy/pre-clean archive"
    if any(token in name for token in ("smoke", "tmp", "test_", "debug")):
        return "invalid", "smoke, test, temporary, or debug artifact"
    if not has_eval and diagnostic_complete:
        return "exploratory", "complete diagnostic unit cell; no task evaluation"
    if not has_eval and has_metrics:
        return "incomplete", "diagnostic metrics exist but required unit-cell artifacts are incomplete"
    if not has_eval:
        return "incomplete", "missing eval_results.json and complete diagnostic metrics"
    if any(token in name for token in ("5epoch", "5ep", "short")):
        return "exploratory", "short-run artifact; not eligible for claim"
    return "exploratory", "historical run; not promoted by protocol v0.1"


def infer_trial_steps(cfg: dict[str, Any], ev: dict[str, Any]) -> int | None:
    """Use recorded steps first; otherwise retain the best explicit inference."""
    for source in (ev, cfg):
        value = source.get("trial_steps")
        if isinstance(value, (int, float)):
            return int(value)

    K = ev.get("K", cfg.get("K", cfg.get("n_queries")))
    sub_win = ev.get("sub_win", cfg.get("sub_win"))
    read_len = ev.get("read_len", cfg.get("read_len", 0))
    if all(isinstance(x, (int, float)) for x in (K, sub_win, read_len)):
        return int(K * sub_win + read_len)

    win_len = ev.get("win_len", cfg.get("win_len"))
    if all(isinstance(x, (int, float)) for x in (win_len, read_len)):
        return int(win_len + read_len)
    return None


def record(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.json"
    ev_path = run_dir / "eval_results.json"
    metrics_path = run_dir / "metrics.json"
    cfg = load_json(cfg_path)
    ev = load_json(ev_path)
    metrics = load_json(metrics_path)
    rel = run_dir.relative_to(RUNS).as_posix()
    diagnostic_complete = diagnostic_artifacts_complete(run_dir, metrics)
    status, reason = infer_status(
        rel,
        has_eval=ev_path.exists(),
        has_metrics=metrics_path.exists(),
        diagnostic_complete=diagnostic_complete,
    )

    K = ev.get("K", cfg.get("K", cfg.get("n_queries")))
    hidden_sizes = ev.get("hidden_sizes", cfg.get("hidden_sizes"))
    n_hidden = ev.get("hidden_size", cfg.get("n_hidden"))
    if isinstance(hidden_sizes, list):
        n_hidden = sum(x for x in hidden_sizes if isinstance(x, (int, float)))

    return {
        "run_path": rel,
        "experiment": rel.split("/", 1)[0],
        "status": status,
        "status_reason": reason,
        "has_config": cfg_path.exists(),
        "has_eval": ev_path.exists(),
        "has_metrics": metrics_path.exists(),
        "diagnostic_artifacts_complete": diagnostic_complete,
        "has_checkpoint": any(run_dir.glob("*.pt")),
        "encoding_mode": cfg.get("encoding_mode", ev.get("encoding_mode")),
        "condition": ev.get("condition", cfg.get("condition")),
        "train_mode": ev.get("train_mode", cfg.get("train_mode")),
        "readout_type": cfg.get("readout_type"),
        "use_output_spikes": cfg.get("use_output_spikes"),
        "K": K,
        "n_hidden_total": n_hidden,
        "n_input": cfg.get("n_input_channels", cfg.get("n_input")),
        "n_output": cfg.get("n_queries", K),
        "trial_steps": infer_trial_steps(cfg, ev),
        "accuracy": ev.get("accuracy"),
        "worst_query_accuracy": ev.get("worst_query_accuracy"),
        "exact_trial_accuracy": ev.get("exact_trial_accuracy"),
        "seed": cfg.get("seed", ev.get("seed")),
        "config_path": cfg_path.relative_to(BASE).as_posix() if cfg_path.exists() else "",
        "eval_path": ev_path.relative_to(BASE).as_posix() if ev_path.exists() else "",
        "metrics_path": metrics_path.relative_to(BASE).as_posix() if metrics_path.exists() else "",
    }


def main() -> None:
    if not RUNS.is_dir():
        raise SystemExit(f"Runs directory missing: {RUNS}")
    run_dirs = {p.parent for p in RUNS.rglob("config.json")}
    run_dirs.update(p.parent for p in RUNS.rglob("eval_results.json"))
    rows = sorted((record(path) for path in run_dirs), key=lambda row: row["run_path"])

    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["run_path", "status"]
    csv_path = OUT / "experiment_registry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {
        "schema_version": "0.2",
        "registry_policy": "Historical task runs and complete diagnostic unit cells are exploratory unless explicitly promoted.",
        "n_runs": len(rows),
        "status_counts": counts,
        "registry": csv_path.relative_to(BASE).as_posix(),
    }
    (OUT / "experiment_registry_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

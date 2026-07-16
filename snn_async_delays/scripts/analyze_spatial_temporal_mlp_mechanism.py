"""Aggregate delay/window mechanism diagnostics for MLP Pareto scaffold v2."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[1]
PROTOCOL = "spatial_vs_temporal_pareto_mlp_scaffold_v2"
ROOT = BASE / "runs" / "exploratory" / PROTOCOL
OUT = BASE / "docs" / "generated" / PROTOCOL / "exploratory"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = []
    for result_path in sorted(ROOT.rglob("validation_results.json")):
        run = result_path.parent
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if not cfg["name"].startswith("shared_temporal_"):
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        npz = np.load(run / "plots" / "diagnostic_data.npz", allow_pickle=True)
        delays = np.asarray(npz["delays__ih"], dtype=float)
        if delays.shape[0] != 8:
            raise ValueError(f"expected eight one-hot input rows in {run}")
        width = int(cfg["output_window_len"])
        q0, q1 = delays[:4], delays[4:]
        window_spikes = result.get("per_query_hidden_window_spikes")
        window_active = result.get("per_query_hidden_window_activity_fraction")
        rows.append({
            "condition": cfg["name"],
            "T": int(cfg["T"]),
            "surface_hidden_width": int(cfg["surface_hidden_width"]),
            "seed": int(cfg["seed"]),
            "output_window_len": width,
            "delay_mean": float(delays.mean()),
            "delay_std": float(delays.std()),
            "delay_min": float(delays.min()),
            "delay_max": float(delays.max()),
            "q0_delay_mean": float(q0.mean()),
            "q1_delay_mean": float(q1.mean()),
            "q1_minus_q0_delay_mean": float(q1.mean() - q0.mean()),
            "q0_delay_over_window": float(q0.mean() / width),
            "q1_delay_over_window": float(q1.mean() / width),
            "q0_fraction_before_second_window": float((q0 < width).mean()),
            "q1_fraction_reaching_second_window": float((q1 >= width - 1).mean()),
            "window0_hidden_spikes": (
                float(window_spikes[0]) if window_spikes is not None else None
            ),
            "window1_hidden_spikes": (
                float(window_spikes[1]) if window_spikes is not None else None
            ),
            "window0_activity_fraction": (
                float(window_active[0]) if window_active is not None else None
            ),
            "window1_activity_fraction": (
                float(window_active[1]) if window_active is not None else None
            ),
            "worst_query_balanced_accuracy": float(
                result["worst_query_balanced_accuracy"]
            ),
            "exact_trial_accuracy": float(result["exact_trial_accuracy"]),
            "routing_selectivity_gap": result.get("routing_selectivity_gap"),
        })

    if len(rows) != 96:
        raise SystemExit(f"expected 96 temporal cells, found {len(rows)}")
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "temporal_mechanism_cells.csv", rows)

    summary = []
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        item = {"condition": condition, "n": len(selected)}
        for field in (
            "delay_mean", "delay_std", "delay_max", "q0_delay_mean",
            "q1_delay_mean", "q1_minus_q0_delay_mean",
            "q0_delay_over_window", "q1_delay_over_window",
            "q1_fraction_reaching_second_window", "window0_hidden_spikes",
            "window1_hidden_spikes", "window0_activity_fraction",
            "window1_activity_fraction", "worst_query_balanced_accuracy",
            "exact_trial_accuracy", "routing_selectivity_gap",
        ):
            values = [row[field] for row in selected if row[field] is not None]
            item[f"{field}_mean"] = float(np.mean(values)) if values else None
            item[f"{field}_min"] = float(np.min(values)) if values else None
            item[f"{field}_max"] = float(np.max(values)) if values else None
        summary.append(item)
    write_csv(OUT / "temporal_mechanism_summary.csv", summary)
    conditions = [
        "shared_temporal_d0", "shared_temporal_oracle", "shared_temporal_wad"
    ]
    labels = ["temporal d0", "fixed oracle", "WAD"]
    x = np.arange(len(conditions))
    by_condition = {item["condition"]: item for item in summary}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    width_bar = .34
    for index, (field, label) in enumerate((
        ("q0_delay_over_window_mean", "query 0"),
        ("q1_delay_over_window_mean", "query 1"),
    )):
        axes[0].bar(
            x + (index - .5) * width_bar,
            [by_condition[condition][field] for condition in conditions],
            width_bar, label=label,
        )
    axes[0].axhline(1.0, color="black", ls="--", lw=.8,
                    label="second-window boundary")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("mean delay / output-window length")
    axes[0].set_title("Selected-checkpoint input→hidden delays")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=.25)

    for index, (field, label) in enumerate((
        ("window0_hidden_spikes_mean", "window 0"),
        ("window1_hidden_spikes_mean", "window 1"),
    )):
        axes[1].bar(
            x + (index - .5) * width_bar,
            [by_condition[condition][field] for condition in conditions],
            width_bar, label=label,
        )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("mean hidden spikes / trial")
    axes[1].set_title("Observed hidden activity by output window")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle(
        "Why WAD fails: no query-conditioned delay separation and no second-window activity"
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_temporal_mechanism_summary.png", dpi=190,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

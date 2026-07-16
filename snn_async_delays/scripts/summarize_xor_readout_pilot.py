"""Summarize completed cells of the preregistered XOR readout pilot.

This script is read-only with respect to run artifacts.  It reports completion
explicitly and refuses to label a partial pilot as a scientific conclusion.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


BASE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "xor_readout_interface_pilot_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=f"runs/canonical/{PROTOCOL_ID}")
    parser.add_argument("--config", default="configs/pilot_xor_readout_v1.yaml")
    parser.add_argument("--output-dir", default=f"docs/generated/{PROTOCOL_ID}")
    args = parser.parse_args()
    run_root = Path(args.runs_dir)
    config_path = Path(args.config)
    output = Path(args.output_dir)
    if not run_root.is_absolute(): run_root = BASE / run_root
    if not config_path.is_absolute(): config_path = BASE / config_path
    if not output.is_absolute(): output = BASE / output
    output.mkdir(parents=True, exist_ok=True)
    with config_path.open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    factors = protocol["factors"]
    expected_per_group = len(factors["seeds"])
    expected_total = (
        len(protocol["task"]["K"]) * len(factors["hidden_size"]) *
        expected_per_group * len(factors["delay_conditions"]) *
        len(factors["observation_modes"]) * len(factors["readout_types"])
    )

    rows = []
    for eval_path in sorted(run_root.rglob("eval_results.json")):
        cfg_path = eval_path.with_name("config.json")
        if not cfg_path.exists():
            continue
        cfg, result = load_json(cfg_path), load_json(eval_path)
        if cfg.get("protocol_id") != PROTOCOL_ID:
            continue
        ledger = result.get("resource_ledger", {})
        rows.append({
            "run_path": eval_path.parent.relative_to(BASE).as_posix(),
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

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["observation_mode"], row["readout_type"])].append(row)

    summary = []
    metric_fields = [
        "accuracy", "worst_query_accuracy", "exact_trial_accuracy", "balanced_accuracy",
        "trainable_parameters", "model_scalar_storage_elements",
        "delay_buffer_elements_per_sample", "neuron_updates_per_trial",
        "dense_synapse_macs_per_trial", "decoder_weight_macs_per_trial",
        "mean_synaptic_events_total",
    ]
    for key, values in sorted(groups.items()):
        row: dict[str, Any] = {
            "condition": key[0], "observation_mode": key[1], "readout_type": key[2],
            "n_completed_seeds": len(values),
            "expected_seeds": expected_per_group,
            "group_complete": len(values) == expected_per_group,
        }
        for field in metric_fields:
            numbers = [float(value[field]) for value in values if value.get(field) is not None]
            row[f"{field}_mean"] = float(np.mean(numbers)) if numbers else None
            row[f"{field}_sd"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
        summary.append(row)

    def write_csv(path: Path, content: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(content[0]) if content else ["condition"])
            writer.writeheader(); writer.writerows(content)

    write_csv(output / "run_level.csv", rows)
    write_csv(output / "group_summary.csv", summary)

    # Plot only complete groups; partial groups must not become visual evidence.
    complete = [row for row in summary if row["group_complete"]]
    if complete:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        modes = ["late_window", "all_time", "time_binned"]
        for decoder in sorted({row["readout_type"] for row in complete}):
            for condition in sorted({row["condition"] for row in complete}):
                subset = [row for row in complete if row["readout_type"] == decoder and row["condition"] == condition]
                lookup = {row["observation_mode"]: row for row in subset}
                x = [mode for mode in modes if mode in lookup]
                if not x:
                    continue
                label = f"{decoder}/{condition}"
                axes[0].errorbar(
                    x, [lookup[m]["worst_query_accuracy_mean"] for m in x],
                    yerr=[lookup[m]["worst_query_accuracy_sd"] or 0 for m in x],
                    fmt="o-", capsize=3, label=label,
                )
                axes[1].errorbar(
                    x, [lookup[m]["exact_trial_accuracy_mean"] for m in x],
                    yerr=[lookup[m]["exact_trial_accuracy_sd"] or 0 for m in x],
                    fmt="o-", capsize=3, label=label,
                )
        for ax, metric in zip(axes, ["worst-query accuracy", "exact-trial accuracy"]):
            ax.set_ylim(0, 1.02); ax.set_ylabel(metric); ax.set_xlabel("observation mode")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle("XOR readout-interface pilot v1 (all groups have 3 seeds)")
        fig.tight_layout(); fig.savefig(output / "accuracy_summary.png", dpi=180); plt.close(fig)

        # Resource plot: static architecture counts, repeated across seeds.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for decoder in sorted({row["readout_type"] for row in complete}):
            # Use WAD row for each decoder/mode: d0 has identical implemented
            # storage except for trainability, and that distinction remains in CSV.
            subset = [row for row in complete if row["readout_type"] == decoder and row["condition"] == "w_and_d"]
            lookup = {row["observation_mode"]: row for row in subset}
            x = [mode for mode in modes if mode in lookup]
            if not x:
                continue
            axes[0].plot(x, [lookup[m]["model_scalar_storage_elements_mean"] for m in x], "o-", label=decoder)
            axes[1].plot(x, [lookup[m]["decoder_weight_macs_per_trial_mean"] for m in x], "o-", label=decoder)
        axes[0].set_ylabel("model scalar storage elements")
        axes[1].set_ylabel("decoder weight MACs / trial")
        for ax in axes:
            ax.set_xlabel("observation mode"); ax.set_yscale("log")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle("Resource change induced by observation interface (WAD implementation)")
        fig.tight_layout(); fig.savefig(output / "resource_summary.png", dpi=180); plt.close(fig)

    metadata = {
        "protocol_id": PROTOCOL_ID,
        "expected_total_cells": expected_total,
        "completed_cells": len(rows),
        "complete": len(rows) == expected_total,
        "interpretation": (
            "Exploratory mechanism pilot only. Figures are written only when every plotted group has all preregistered seeds."
        ),
    }
    (output / "README.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

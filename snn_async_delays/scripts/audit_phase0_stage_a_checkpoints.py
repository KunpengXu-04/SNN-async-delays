"""Read-only mechanism audit for Phase-0 Stage-A best/final checkpoints.

The script never trains or writes into run directories. It replays the exact
four-pattern XOR truth table through both saved checkpoints and records output
existence/exclusivity, target-time voltage margins, causal hidden activity,
parameter drift, and realized delays.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from torch.utils.data import DataLoader

from data.boolean_dataset import ExhaustiveFixedOperationQueryDataset
from data.encoding import encode_simultaneous_trial
from scripts.run_spatial_vs_temporal_pareto_phase0 import build_model


BASE = Path(__file__).resolve().parents[1]
PROTOCOL = "spatial_vs_temporal_pareto_phase0"
ROOT = BASE / "runs/exploratory" / PROTOCOL / "stage_a"
STAGE_A_CELLS = (
    BASE / "docs/generated" / PROTOCOL / "stage_a" / "cells.csv"
)
OUT = BASE / "docs/generated" / PROTOCOL / "stage_a_checkpoint_audit"
THRESHOLD_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _formal_cell_rows() -> dict[tuple[int, int], dict]:
    with STAGE_A_CELLS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(int(row["hidden_size"]), int(row["seed"])): row for row in rows}


def _final_category(row: dict) -> str:
    if _as_bool(row["passes_all_locked_gates"]):
        return "successful"
    if float(row["collision_rate"]) > 0.10:
        return "collision"
    if float(row["silent_rate"]) > 0.10:
        return "silent"
    return "other_failure"


def _best_epoch(run_dir: Path) -> tuple[int, float]:
    with (run_dir / "train_log.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    best_epoch, best_acc = -1, float("-inf")
    for row in rows:
        accuracy = float(row["val_acc"])
        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = int(row["epoch"])
    return best_epoch, best_acc


def _encode_truth_table(cfg: dict, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        ExhaustiveFixedOperationQueryDataset(cfg["query_ops"]),
        batch_size=4, shuffle=False,
    )
    A, B, op_ids, labels = next(iter(loader))
    spike_input = encode_simultaneous_trial(
        A, B,
        win_len=cfg["win_len"], read_len=cfg["read_len"],
        r_on=cfg["r_on"], r_off=cfg["r_off"], dt=cfg["dt"], device=device,
        op_ids=op_ids, n_ops=cfg.get("n_ops", 0),
        encoding_mode=cfg["encoding_mode"],
        burst_n_spikes_on=cfg.get("burst_n_spikes_on", 1),
        burst_n_spikes_off=cfg.get("burst_n_spikes_off", 1),
        burst_phase_on=cfg.get("burst_phase_on", 1.0),
        burst_phase_off=cfg.get("burst_phase_off", 1.0),
        burst_jitter_ms=cfg.get("burst_jitter_ms", 0),
        one_hot_phase=cfg["one_hot_phase"],
        one_hot_n_spikes=cfg["one_hot_n_spikes"],
    )
    return spike_input, labels, torch.stack((A[:, 0], B[:, 0]), dim=1)


def _checkpoint_audit(run_dir: Path, cfg: dict, checkpoint_name: str,
                      final_category: str, device: str) -> tuple[list[dict], dict]:
    model = build_model(cfg).to(device)
    state = torch.load(run_dir / checkpoint_name, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    spike_input, labels, bits = _encode_truth_table(cfg, device)
    with torch.no_grad():
        logits, info = model(spike_input.to(device), record=True)

    logits = logits.detach().cpu().numpy()[:, 0]
    labels_np = labels.numpy()[:, 0].astype(int)
    bits_np = bits.numpy().astype(int)
    output_spikes = info["output_spike_train"].numpy()
    output_voltage = info["output_membrane_train"].numpy()
    output_current = info["output_synaptic_current_train"].numpy()
    hidden_spikes = info["hidden_spike_train"].numpy()
    threshold = float(cfg["lif_output_threshold"])
    target_time = int(cfg["win_len"] + cfg["output_target_offset_steps"])
    causal_hidden_time = target_time - 1
    output_counts = output_spikes.sum(axis=1)
    delay_values = model.get_delays()
    d_ih = delay_values["ih"].detach().cpu().numpy()
    d_ho = delay_values["ho"].detach().cpu().numpy()

    pattern_rows: list[dict] = []
    for index, ((a, b), target_class) in enumerate(zip(bits_np, labels_np)):
        wrong_class = 1 - target_class
        total_output = float(output_counts[index].sum())
        correct_count = float(output_counts[index, target_class])
        wrong_count = float(output_counts[index, wrong_class])
        if total_output == 0:
            interface_status = "silent"
        elif correct_count > 0 and wrong_count > 0:
            interface_status = "collision"
        elif total_output == 1 and correct_count == 1:
            interface_status = "valid"
        elif wrong_count > 0:
            interface_status = "wrong_class"
        else:
            interface_status = "extra_correct_spikes"

        correct_voltage = float(output_voltage[index, target_time, target_class])
        wrong_voltage = float(output_voltage[index, target_time, wrong_class])
        correct_spike_times = np.where(output_spikes[index, :, target_class] > 0)[0]
        wrong_spike_times = np.where(output_spikes[index, :, wrong_class] > 0)[0]
        correct_peak_time = int(np.argmax(output_voltage[index, :, target_class]))
        wrong_peak_time = int(np.argmax(output_voltage[index, :, wrong_class]))
        correct_peak_voltage = float(output_voltage[index, correct_peak_time, target_class])
        wrong_peak_voltage = float(output_voltage[index, wrong_peak_time, wrong_class])
        pattern_rows.append({
            "cell": run_dir.name,
            "hidden_size": int(cfg["n_hidden"]),
            "seed": int(cfg["seed"]),
            "final_category": final_category,
            "checkpoint": checkpoint_name.removesuffix("_model.pt"),
            "A": int(a),
            "B": int(b),
            "pattern": f"{a}{b}",
            "target_class": int(target_class),
            "logit": float(logits[index]),
            "prediction": int(logits[index] > 0),
            "classification_correct": bool((logits[index] > 0) == target_class),
            "output_class0_spikes": float(output_counts[index, 0]),
            "output_class1_spikes": float(output_counts[index, 1]),
            "total_output_spikes": total_output,
            "interface_status": interface_status,
            "valid_one_target_spike": interface_status == "valid",
            "target_time": target_time,
            "target_time_correct_spike": float(output_spikes[index, target_time, target_class]),
            "target_time_wrong_spike": float(output_spikes[index, target_time, wrong_class]),
            "correct_spike_times": "|".join(map(str, correct_spike_times.tolist())),
            "wrong_spike_times": "|".join(map(str, wrong_spike_times.tolist())),
            "target_time_correct_voltage": correct_voltage,
            "target_time_wrong_voltage": wrong_voltage,
            "correct_threshold_margin": correct_voltage - threshold,
            "wrong_suppression_margin": threshold - wrong_voltage,
            "class_voltage_separation": correct_voltage - wrong_voltage,
            "correct_peak_time": correct_peak_time,
            "wrong_peak_time": wrong_peak_time,
            "correct_peak_voltage": correct_peak_voltage,
            "wrong_peak_voltage": wrong_peak_voltage,
            "correct_peak_threshold_margin": correct_peak_voltage - threshold,
            "wrong_global_suppression_margin": threshold - wrong_peak_voltage,
            "target_time_correct_current": float(output_current[index, target_time, target_class]),
            "target_time_wrong_current": float(output_current[index, target_time, wrong_class]),
            "hidden_spikes_total": float(hidden_spikes[index].sum()),
            "active_hidden_neurons": int((hidden_spikes[index].sum(axis=0) > 0).sum()),
            "causal_hidden_spikes_at_t_minus_1": float(
                hidden_spikes[index, causal_hidden_time].sum()
            ),
        })

    statuses = Counter(row["interface_status"] for row in pattern_rows)
    summary = {
        "cell": run_dir.name,
        "hidden_size": int(cfg["n_hidden"]),
        "seed": int(cfg["seed"]),
        "final_category": final_category,
        "checkpoint": checkpoint_name.removesuffix("_model.pt"),
        "truth_patterns_correct": sum(row["classification_correct"] for row in pattern_rows),
        "valid_interface_patterns": statuses["valid"],
        "silent_patterns": statuses["silent"],
        "collision_patterns": statuses["collision"],
        "wrong_class_patterns": statuses["wrong_class"],
        "extra_correct_spike_patterns": statuses["extra_correct_spikes"],
        "min_correct_threshold_margin": min(
            row["correct_threshold_margin"] for row in pattern_rows
        ),
        "min_wrong_suppression_margin": min(
            row["wrong_suppression_margin"] for row in pattern_rows
        ),
        "min_wrong_global_suppression_margin": min(
            row["wrong_global_suppression_margin"] for row in pattern_rows
        ),
        "min_class_voltage_separation": min(
            row["class_voltage_separation"] for row in pattern_rows
        ),
        "mean_hidden_spikes": float(np.mean([
            row["hidden_spikes_total"] for row in pattern_rows
        ])),
        "input_hidden_delay_min": float(d_ih.min()),
        "input_hidden_delay_max": float(d_ih.max()),
        "hidden_output_delay_min": float(d_ho.min()),
        "hidden_output_delay_max": float(d_ho.max()),
    }
    return pattern_rows, summary


def _parameter_drift(run_dir: Path, cfg: dict, summaries: dict[str, dict]) -> dict:
    best = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    final = torch.load(run_dir / "last_model.pt", map_location="cpu", weights_only=True)

    def delta_norm(key: str) -> float:
        return float(torch.linalg.vector_norm(final[key] - best[key]).item())

    all_deltas = torch.cat([
        (final[key] - best[key]).reshape(-1).float()
        for key in final if torch.is_floating_point(final[key])
    ])
    best_epoch, best_val_acc = _best_epoch(run_dir)
    return {
        "cell": run_dir.name,
        "hidden_size": int(cfg["n_hidden"]),
        "seed": int(cfg["seed"]),
        "final_category": summaries["final"]["final_category"],
        "best_epoch": best_epoch,
        "best_logged_validation_accuracy": best_val_acc,
        "input_hidden_weight_l2_drift": delta_norm("syn_ih.weight"),
        "hidden_output_weight_l2_drift": delta_norm("syn_ho.weight"),
        "all_float_state_l2_drift": float(torch.linalg.vector_norm(all_deltas).item()),
        "best_truth_patterns_correct": summaries["best"]["truth_patterns_correct"],
        "final_truth_patterns_correct": summaries["final"]["truth_patterns_correct"],
        "best_valid_interface_patterns": summaries["best"]["valid_interface_patterns"],
        "final_valid_interface_patterns": summaries["final"]["valid_interface_patterns"],
        "valid_interface_pattern_change_final_minus_best": (
            summaries["final"]["valid_interface_patterns"]
            - summaries["best"]["valid_interface_patterns"]
        ),
        "best_min_correct_threshold_margin": summaries["best"]["min_correct_threshold_margin"],
        "final_min_correct_threshold_margin": summaries["final"]["min_correct_threshold_margin"],
        "best_min_wrong_suppression_margin": summaries["best"]["min_wrong_suppression_margin"],
        "final_min_wrong_suppression_margin": summaries["final"]["min_wrong_suppression_margin"],
        "best_min_wrong_global_suppression_margin": summaries["best"]["min_wrong_global_suppression_margin"],
        "final_min_wrong_global_suppression_margin": summaries["final"]["min_wrong_global_suppression_margin"],
    }


def _plot(pattern_rows: list[dict], ordered_cells: list[str]) -> None:
    patterns = ["00", "10", "01", "11"]
    final_rows = {
        (row["cell"], row["pattern"]): row
        for row in pattern_rows if row["checkpoint"] == "last"
    }
    best_rows = {
        (row["cell"], row["pattern"]): row
        for row in pattern_rows if row["checkpoint"] == "best"
    }
    correct = np.array([
        [final_rows[(cell, pattern)]["correct_threshold_margin"] for pattern in patterns]
        for cell in ordered_cells
    ])
    suppression = np.array([
        [final_rows[(cell, pattern)]["wrong_global_suppression_margin"] for pattern in patterns]
        for cell in ordered_cells
    ])
    separation_change = np.array([
        [final_rows[(cell, pattern)]["class_voltage_separation"]
         - best_rows[(cell, pattern)]["class_voltage_separation"] for pattern in patterns]
        for cell in ordered_cells
    ])
    status_codes = {"valid": 0, "silent": 1, "collision": 2,
                    "wrong_class": 3, "extra_correct_spikes": 4}
    status = np.array([
        [status_codes[final_rows[(cell, pattern)]["interface_status"]]
         for pattern in patterns]
        for cell in ordered_cells
    ])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10), constrained_layout=True)
    for ax, matrix, title, colorbar_label in (
        (axes[0, 0], correct, "Final: correct-neuron threshold margin",
         r"$V_{correct}(t^*)-\theta$"),
        (axes[0, 1], suppression, "Final: wrong-neuron global suppression margin",
         r"$\theta-\max_t V_{wrong}(t)$"),
        (axes[1, 1], separation_change, "Final minus best: class-voltage separation",
         r"$\Delta(V_{correct}-V_{wrong})$"),
    ):
        limit = max(float(np.abs(matrix).max()), 1e-6)
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                          norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        fig.colorbar(image, ax=ax, shrink=.75, label=colorbar_label)
        ax.set_title(title)

    status_cmap = ListedColormap(["#4daf4a", "#999999", "#e41a1c", "#984ea3", "#ff7f00"])
    image = axes[1, 0].imshow(status, aspect="auto", cmap=status_cmap,
                              vmin=-.5, vmax=4.5)
    cbar = fig.colorbar(image, ax=axes[1, 0], shrink=.75, ticks=range(5))
    cbar.ax.set_yticklabels(["valid", "silent", "collision", "wrong", "extra"])
    axes[1, 0].set_title("Final checkpoint output-interface state")

    for ax in axes.flat:
        ax.set_xticks(range(len(patterns)), patterns)
        ax.set_xlabel("XOR input pattern AB")
        ax.set_yticks(range(len(ordered_cells)), ordered_cells, fontsize=7)
    fig.savefig(OUT / "stage_a_checkpoint_mechanism_audit.png", dpi=180)
    plt.close(fig)


def _plot_threshold_sensitivity(rows: list[dict]) -> None:
    thresholds = [row["output_threshold"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(thresholds, [row["valid_patterns"] for row in rows], marker="o",
            label="valid one-target patterns")
    ax.plot(thresholds, [row["silent_patterns"] for row in rows], marker="o",
            label="silent patterns")
    ax.plot(thresholds, [row["collision_patterns"] for row in rows], marker="o",
            label="collision patterns")
    ax.axvline(.03, color="black", linestyle="--", linewidth=1,
               label="formal threshold=.03")
    ax.set_xlabel("counterfactual output threshold")
    ax.set_ylabel("patterns across 15 cells (maximum 60)")
    ax.set_title("Exploratory read-only threshold sensitivity")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_sensitivity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    formal = _formal_cell_rows()
    run_dirs = sorted(
        (path for path in ROOT.iterdir() if path.is_dir()),
        key=lambda path: (
            int(path.name.split("_")[0].removeprefix("h")),
            int(path.name.split("seed")[1]),
        ),
    )
    if len(run_dirs) != 15:
        raise SystemExit(f"Expected 15 Stage-A run directories, found {len(run_dirs)}")

    pattern_rows: list[dict] = []
    checkpoint_summaries: list[dict] = []
    drift_rows: list[dict] = []
    configs_by_cell: dict[str, dict] = {}
    for run_dir in run_dirs:
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        configs_by_cell[run_dir.name] = cfg
        key = (int(cfg["n_hidden"]), int(cfg["seed"]))
        category = _final_category(formal[key])
        per_checkpoint: dict[str, dict] = {}
        for checkpoint_name, short_name in (
            ("best_model.pt", "best"), ("last_model.pt", "final")
        ):
            patterns, summary = _checkpoint_audit(
                run_dir, cfg, checkpoint_name, category, device
            )
            pattern_rows.extend(patterns)
            checkpoint_summaries.append(summary)
            per_checkpoint[short_name] = summary
        drift_rows.append(_parameter_drift(run_dir, cfg, per_checkpoint))

    OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT / "checkpoint_pattern_audit.csv", pattern_rows)
    _write_csv(OUT / "checkpoint_summary.csv", checkpoint_summaries)
    _write_csv(OUT / "best_final_drift.csv", drift_rows)
    ordered_cells = [path.name for path in run_dirs]
    _plot(pattern_rows, ordered_cells)

    threshold_cell_rows: list[dict] = []
    threshold_summary_rows: list[dict] = []
    formal_threshold = float(next(iter(configs_by_cell.values()))["lif_output_threshold"])
    thresholds = [formal_threshold * multiplier for multiplier in THRESHOLD_MULTIPLIERS]
    for threshold in thresholds:
        threshold_patterns: list[dict] = []
        per_cell: list[dict] = []
        for run_dir in run_dirs:
            cfg = dict(configs_by_cell[run_dir.name])
            cfg["lif_output_threshold"] = threshold
            key = (int(cfg["n_hidden"]), int(cfg["seed"]))
            patterns, summary = _checkpoint_audit(
                run_dir, cfg, "last_model.pt", _final_category(formal[key]), device
            )
            threshold_patterns.extend(patterns)
            cell_row = {
                "output_threshold": threshold,
                "cell": run_dir.name,
                "hidden_size": int(cfg["n_hidden"]),
                "seed": int(cfg["seed"]),
                "truth_patterns_correct": summary["truth_patterns_correct"],
                "valid_interface_patterns": summary["valid_interface_patterns"],
                "silent_patterns": summary["silent_patterns"],
                "collision_patterns": summary["collision_patterns"],
                "full_truth_and_interface": (
                    summary["truth_patterns_correct"] == 4
                    and summary["valid_interface_patterns"] == 4
                ),
            }
            threshold_cell_rows.append(cell_row)
            per_cell.append(cell_row)
        passing_widths = []
        for hidden_size in sorted({row["hidden_size"] for row in per_cell}):
            width_rows = [row for row in per_cell if row["hidden_size"] == hidden_size]
            if len(width_rows) == 3 and all(row["full_truth_and_interface"] for row in width_rows):
                passing_widths.append(hidden_size)
        counts = Counter(row["interface_status"] for row in threshold_patterns)
        threshold_summary_rows.append({
            "output_threshold": threshold,
            "cells_with_full_truth_and_interface": sum(
                row["full_truth_and_interface"] for row in per_cell
            ),
            "valid_patterns": counts["valid"],
            "silent_patterns": counts["silent"],
            "collision_patterns": counts["collision"],
            "wrong_class_patterns": counts["wrong_class"],
            "extra_correct_spike_patterns": counts["extra_correct_spikes"],
            "widths_with_all_three_seeds_full": "|".join(map(str, passing_widths)),
        })
    _write_csv(OUT / "threshold_sensitivity_by_cell.csv", threshold_cell_rows)
    _write_csv(OUT / "threshold_sensitivity_summary.csv", threshold_summary_rows)
    _plot_threshold_sensitivity(threshold_summary_rows)

    final_summaries = [row for row in checkpoint_summaries if row["checkpoint"] == "last"]
    best_summaries = [row for row in checkpoint_summaries if row["checkpoint"] == "best"]
    decision = {
        "protocol": PROTOCOL,
        "audit": "stage_a_checkpoint_mechanism_audit_v1",
        "read_only": True,
        "cells": len(run_dirs),
        "checkpoints": len(checkpoint_summaries),
        "truth_pattern_forward_passes": len(pattern_rows),
        "final_category_counts": dict(Counter(row["final_category"] for row in final_summaries)),
        "best_checkpoints_with_four_valid_interface_patterns": sum(
            row["valid_interface_patterns"] == 4 for row in best_summaries
        ),
        "final_checkpoints_with_four_valid_interface_patterns": sum(
            row["valid_interface_patterns"] == 4 for row in final_summaries
        ),
        "cells_losing_valid_interface_patterns_best_to_final": sum(
            row["valid_interface_pattern_change_final_minus_best"] < 0 for row in drift_rows
        ),
        "all_input_hidden_delays_zero": all(
            row["input_hidden_delay_min"] == 0 and row["input_hidden_delay_max"] == 0
            for row in checkpoint_summaries
        ),
        "all_hidden_output_delays_zero": all(
            row["hidden_output_delay_min"] == 0 and row["hidden_output_delay_max"] == 0
            for row in checkpoint_summaries
        ),
        "exploratory_output_thresholds": thresholds,
        "thresholds_with_any_width_full_across_three_seeds": [
            row["output_threshold"] for row in threshold_summary_rows
            if row["widths_with_all_three_seeds_full"]
        ],
        "test_split_opened": False,
    }
    (OUT / "audit_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

"""Audit and summarize XOR delay-control matrix v1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.generate_diagnostic_plots import _load_model_from_run


BASE = Path(__file__).resolve().parents[1]
PROTOCOL = "xor_delay_control_matrix_v1"
ROOT = BASE / "runs" / "exploratory" / PROTOCOL
OUT = BASE / "docs" / "generated"


RESOURCE_KEYS = [
    "trainable_parameters", "delay_value_storage_elements",
    "trainable_delay_parameters", "model_scalar_storage_elements",
    "neuron_updates_per_trial", "dense_synapse_macs_per_trial",
    "decoder_weight_macs_per_trial", "mean_synaptic_events_total",
]


def read_training_rows() -> pd.DataFrame:
    rows = []
    for path in ROOT.rglob("validation_results.json"):
        if "interventions" in path.parts:
            continue
        run = path.parent
        result = json.loads(path.read_text(encoding="utf-8"))
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        ledger = result["resource_ledger"]
        rows.append({
            "run_path": run.relative_to(BASE).as_posix(),
            "setting": cfg["matrix_setting"], "K": cfg["K"],
            "hidden_size": cfg["n_hidden"], "readout_type": cfg["readout_type"],
            "condition": cfg["name"], "seed": cfg["seed"],
            "evaluation_split": result.get("evaluation_split"),
            "accuracy": result["accuracy"],
            "worst_query_accuracy": result["worst_query_accuracy"],
            "exact_trial_accuracy": result["exact_trial_accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "mean_hidden_spikes": result["mean_hidden_spikes"],
            **{key: ledger.get(key) for key in RESOURCE_KEYS},
        })
    return pd.DataFrame(rows)


def read_intervention_rows(training: pd.DataFrame) -> pd.DataFrame:
    base = training.set_index(["setting", "readout_type", "condition", "seed"])
    rows = []
    for path in ROOT.rglob("validation_results.json"):
        if "interventions" not in path.parts:
            continue
        source = path.parents[2]
        cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (cfg["matrix_setting"], cfg["readout_type"], cfg["name"], cfg["seed"])
        original = base.loc[key]
        rows.append({
            "source_run": source.relative_to(BASE).as_posix(),
            "setting": key[0], "readout_type": key[1], "condition": key[2],
            "seed": key[3], "intervention": path.parent.name,
            "original_worst": original["worst_query_accuracy"],
            "intervention_worst": result["worst_query_accuracy"],
            "worst_drop": original["worst_query_accuracy"] - result["worst_query_accuracy"],
            "original_exact": original["exact_trial_accuracy"],
            "intervention_exact": result["exact_trial_accuracy"],
            "exact_drop": original["exact_trial_accuracy"] - result["exact_trial_accuracy"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    training = read_training_rows()
    if len(training) != 60 or training["run_path"].nunique() != 60:
        raise SystemExit(f"Expected 60 training cells, found {len(training)}")
    if set(training["evaluation_split"]) != {"val"}:
        raise SystemExit("Test or unknown split detected")
    interventions = read_intervention_rows(training)
    counts = interventions["intervention"].value_counts().to_dict()
    if counts != {"late_window_probe": 60, "shuffle_learned_delays": 15}:
        raise SystemExit(f"Intervention count mismatch: {counts}")

    training.to_csv(OUT / f"{PROTOCOL}_run_level.csv", index=False)
    interventions.to_csv(OUT / f"{PROTOCOL}_intervention_run_level.csv", index=False)
    group = training.groupby(["setting", "readout_type", "condition"], as_index=False).agg(
        n=("seed", "count"), worst_mean=("worst_query_accuracy", "mean"),
        worst_sd=("worst_query_accuracy", "std"),
        exact_mean=("exact_trial_accuracy", "mean"), exact_sd=("exact_trial_accuracy", "std"),
        balanced_mean=("balanced_accuracy", "mean"),
        hidden_spikes_mean=("mean_hidden_spikes", "mean"),
        **{key: (key, "mean") for key in RESOURCE_KEYS},
    )
    group.to_csv(OUT / f"{PROTOCOL}_group_summary.csv", index=False)
    intervention_group = interventions.groupby(
        ["intervention", "setting", "readout_type", "condition"], as_index=False
    ).agg(
        n=("seed", "count"), original_worst=("original_worst", "mean"),
        intervention_worst=("intervention_worst", "mean"),
        worst_drop_mean=("worst_drop", "mean"), worst_drop_sd=("worst_drop", "std"),
        positive_worst_drops=("worst_drop", lambda x: int((x > 0).sum())),
        exact_drop_mean=("exact_drop", "mean"),
    )
    intervention_group.to_csv(OUT / f"{PROTOCOL}_intervention_summary.csv", index=False)

    primary = training[(training.setting == "primary") & (training.readout_type == "linear")]
    pivot = primary.pivot(index="seed", columns="condition",
                          values=["worst_query_accuracy", "exact_trial_accuracy", "mean_hidden_spikes"])
    best_control = (
        primary[primary.condition != "w_and_d"].groupby("condition")["worst_query_accuracy"]
        .mean().idxmax()
    )
    paired = pd.DataFrame({
        "seed": pivot.index,
        "best_control": best_control,
        "wad_worst": pivot["worst_query_accuracy"]["w_and_d"].values,
        "control_worst": pivot["worst_query_accuracy"][best_control].values,
        "worst_delta_wad_minus_control": (
            pivot["worst_query_accuracy"]["w_and_d"] - pivot["worst_query_accuracy"][best_control]
        ).values,
        "wad_exact": pivot["exact_trial_accuracy"]["w_and_d"].values,
        "control_exact": pivot["exact_trial_accuracy"][best_control].values,
        "exact_delta_wad_minus_control": (
            pivot["exact_trial_accuracy"]["w_and_d"] - pivot["exact_trial_accuracy"][best_control]
        ).values,
    })
    paired.to_csv(OUT / f"{PROTOCOL}_primary_paired_gate.csv", index=False)

    shuffle = interventions[
        (interventions.intervention == "shuffle_learned_delays") &
        (interventions.setting == "primary") &
        (interventions.readout_type == "linear")
    ]
    mean_delta = float(paired["worst_delta_wad_minus_control"].mean())
    positive = int((paired["worst_delta_wad_minus_control"] > 0).sum())
    shuffle_positive = int((shuffle["worst_drop"] > 0).sum())
    decision = {
        "protocol_id": PROTOCOL,
        "training_cells": len(training), "late_window_probes": 60,
        "shuffle_probes": 15, "evaluation_split": "val",
        "primary_best_nonlearned_control": best_control,
        "primary_wad_minus_control_worst_mean": mean_delta,
        "primary_positive_paired_seeds": positive,
        "primary_total_paired_seeds": 5,
        "primary_shuffle_worst_drop_mean": float(shuffle["worst_drop"].mean()),
        "primary_shuffle_positive_drops": shuffle_positive,
        "gate": {
            "mean_margin_at_least_0.03": mean_delta >= 0.03,
            "positive_in_at_least_4_of_5": positive >= 4,
            "shuffle_drop_in_at_least_4_of_5": shuffle_positive >= 4,
        },
        "decision": "reject_positive_learned_delay_superiority_programme",
        "reason": (
            "Optimized scalar is the strongest primary non-learned control; "
            "WAD is worse on mean primary worst-query and wins only one paired seed. "
            "Shuffle sensitivity supports co-adapted placement but not superiority."
        ),
        "test_split_opened": False,
    }
    (OUT / f"{PROTOCOL}_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    delay_rows = []
    for _, row in training[(training.setting == "primary") &
                           (training.readout_type == "linear") &
                           (training.condition.isin(["scalar", "fixed_heterogeneous", "w_and_d"]))].iterrows():
        model, _ = _load_model_from_run(str(BASE / row.run_path), "cpu")
        delays = model.get_delays()["ih"].detach().cpu().numpy()
        delay_rows.append({
            "condition": row.condition, "seed": row.seed,
            "delay_mean": delays.mean(), "delay_sd": delays.std(),
            "delay_min": delays.min(), "delay_max": delays.max(),
        })
    pd.DataFrame(delay_rows).to_csv(OUT / f"{PROTOCOL}_delay_summary.csv", index=False)

    order = ["d0", "scalar", "fixed_heterogeneous", "w_and_d"]
    colors = ["#4C72B0", "#55A868", "#8172B2", "#C44E52"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (setting, readout) in zip(axes, [("primary", "linear"), ("primary", "mlp"), ("stress", "linear")]):
        sub = group[(group.setting == setting) & (group.readout_type == readout)].set_index("condition").loc[order]
        ax.bar(np.arange(4), sub.worst_mean, yerr=sub.worst_sd, color=colors, capsize=4)
        ax.set_xticks(np.arange(4)); ax.set_xticklabels(["d0", "scalar", "fixed het.", "WAD"], rotation=20)
        ax.set_ylim(0.45, 0.95); ax.set_ylabel("Validation worst-query")
        ax.set_title(f"{setting}, {readout}"); ax.grid(axis="y", alpha=.2)
    fig.suptitle("XOR delay-control matrix v1 (mean ± SD, 5 seeds; validation only)")
    fig.tight_layout(); fig.savefig(OUT / f"{PROTOCOL}_performance.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].bar(paired.seed.astype(str), paired.worst_delta_wad_minus_control,
                color=["#55A868" if x > 0 else "#C44E52" for x in paired.worst_delta_wad_minus_control])
    axes[0].axhline(0, color="black"); axes[0].axhline(.03, color="gray", linestyle="--")
    axes[0].set_title(f"Primary WAD − {best_control}"); axes[0].set_ylabel("Worst-query difference")
    axes[1].bar(shuffle.seed.astype(str), shuffle.worst_drop, color="#8172B2")
    axes[1].axhline(0, color="black"); axes[1].set_title("WAD original − shuffled")
    axes[1].set_ylabel("Worst-query drop");
    for ax in axes: ax.set_xlabel("Paired seed"); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / f"{PROTOCOL}_gate_and_shuffle.png", dpi=180); plt.close(fig)

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

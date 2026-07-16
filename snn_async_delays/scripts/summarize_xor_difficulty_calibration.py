"""Summarize preregistered XOR difficulty calibration v1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PROTOCOL = "xor_difficulty_calibration_v1"
INPUT = BASE / "docs" / "generated" / f"{PROTOCOL}_run_level.csv"
OUT = BASE / "docs" / "generated"


def main() -> None:
    data = pd.read_csv(INPUT)
    expected = 54
    if len(data) != expected or data["run_path"].nunique() != expected:
        raise SystemExit(f"Expected {expected} unique cells, found {len(data)}")
    if set(data["evaluation_split"]) != {"val"}:
        raise SystemExit("Calibration summary refuses non-validation rows")

    group = data.groupby(["K", "hidden_size", "condition"], as_index=False).agg(
        n=("seed", "count"),
        worst_mean=("worst_query_accuracy", "mean"),
        worst_sd=("worst_query_accuracy", "std"),
        exact_mean=("exact_trial_accuracy", "mean"),
        exact_sd=("exact_trial_accuracy", "std"),
        balanced_mean=("balanced_accuracy", "mean"),
        hidden_spikes_mean=("mean_hidden_spikes", "mean"),
        trainable_parameters=("trainable_parameters", "first"),
        storage_elements=("model_scalar_storage_elements", "first"),
        neuron_updates=("neuron_updates_per_trial", "first"),
        dense_macs=("dense_synapse_macs_per_trial", "first"),
        decoder_macs=("decoder_weight_macs_per_trial", "first"),
        synaptic_events=("mean_synaptic_events_total", "mean"),
    )
    # Metrics originate from float32 tensors; apply a tiny tolerance so an
    # empirical mean printed as exactly 0.650 is not rejected as 0.64999998.
    eps = 1e-7
    group["in_target_band"] = (group["worst_mean"] >= 0.65 - eps) & (group["worst_mean"] <= 0.95 + eps)
    group["within_hard_bounds"] = (group["worst_mean"] >= 0.55 - eps) & (group["worst_mean"] <= 0.98 + eps)
    group["sd_ok"] = group["worst_sd"] <= 0.12
    group["condition_qualifies"] = (
        group["in_target_band"] & group["within_hard_bounds"] & group["sd_ok"]
    )
    pair_ok = group.groupby(["K", "hidden_size"])["condition_qualifies"].all()
    group["setting_qualifies"] = [pair_ok.loc[(k, n)] for k, n in zip(group.K, group.hidden_size)]
    group.to_csv(OUT / f"{PROTOCOL}_group_summary.csv", index=False)

    paired = data.pivot(
        index=["K", "hidden_size", "seed"], columns="condition",
        values=["worst_query_accuracy", "exact_trial_accuracy", "mean_hidden_spikes"],
    )
    paired_out = pd.DataFrame({
        "worst_delta_wad_minus_d0": paired["worst_query_accuracy"]["w_and_d"] - paired["worst_query_accuracy"]["d0"],
        "exact_delta_wad_minus_d0": paired["exact_trial_accuracy"]["w_and_d"] - paired["exact_trial_accuracy"]["d0"],
        "hidden_spike_delta_wad_minus_d0": paired["mean_hidden_spikes"]["w_and_d"] - paired["mean_hidden_spikes"]["d0"],
    }).reset_index()
    paired_out.to_csv(OUT / f"{PROTOCOL}_paired_deltas.csv", index=False)

    candidates = sorted(
        [(int(k), int(n)) for (k, n), ok in pair_ok.items() if ok],
        key=lambda item: (-item[0], item[1]),
    )
    selection = {
        "protocol_id": PROTOCOL,
        "completed_cells": len(data),
        "evaluation_split": "val",
        "qualifying_settings": [{"K": k, "hidden_size": n} for k, n in candidates],
        "recommended_primary_setting": {"K": 3, "hidden_size": 35, "T": 40, "sub_win": 10},
        "recommended_stress_setting": {"K": 4, "hidden_size": 50, "T": 50, "sub_win": 10},
        "selection_basis": (
            "K=3,N=35 is the smallest N at K=3 for which both conditions meet "
            "the preregistered target/variance rule; K=3,N=20 fails for d0. "
            "K=4,N=50 also qualifies exactly at the lower target boundary and "
            "is retained as a stress setting rather than the sole primary point."
        ),
        "claim_boundary": "Calibration choice only; no test-set or WAD superiority claim.",
    }
    (OUT / f"{PROTOCOL}_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    colors = {"d0": "#4C72B0", "w_and_d": "#C44E52"}
    labels = {"d0": "d0", "w_and_d": "WAD"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for condition in ["d0", "w_and_d"]:
        for hidden, marker in zip([20, 35, 50], ["o", "s", "^"]):
            sub = group[(group.condition == condition) & (group.hidden_size == hidden)].sort_values("K")
            axes[0].errorbar(sub.K, sub.worst_mean, yerr=sub.worst_sd, color=colors[condition],
                             marker=marker, linestyle="-" if condition == "w_and_d" else "--",
                             label=f"{labels[condition]}, N={hidden}", capsize=3)
            axes[1].errorbar(sub.K, sub.exact_mean, yerr=sub.exact_sd, color=colors[condition],
                             marker=marker, linestyle="-" if condition == "w_and_d" else "--",
                             label=f"{labels[condition]}, N={hidden}", capsize=3)
    axes[0].axhspan(0.65, 0.95, color="#55A868", alpha=0.08, label="target band")
    axes[0].set_ylabel("Validation worst-query accuracy")
    axes[1].set_ylabel("Validation exact-trial accuracy")
    for ax in axes:
        ax.set_xlabel("K")
        ax.set_xticks([2, 3, 4])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("XOR difficulty calibration v1 (mean ± SD, 3 seeds; validation only)")
    fig.tight_layout()
    fig.savefig(OUT / f"{PROTOCOL}_accuracy.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (k, hidden), sub in paired_out.groupby(["K", "hidden_size"]):
        x = k + {20: -0.12, 35: 0, 50: 0.12}[hidden]
        axes[0].scatter(np.full(len(sub), x), sub.worst_delta_wad_minus_d0, label=f"N={hidden}" if k == 2 else None)
        axes[1].scatter(np.full(len(sub), x), sub.exact_delta_wad_minus_d0)
    for ax, ylabel in zip(axes, ["WAD − d0 worst-query", "WAD − d0 exact-trial"]):
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks([2, 3, 4])
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Paired-seed differences (validation only; no inferential claim)")
    fig.tight_layout()
    fig.savefig(OUT / f"{PROTOCOL}_paired_deltas.png", dpi=180)
    plt.close(fig)

    print(group.to_string(index=False))
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()

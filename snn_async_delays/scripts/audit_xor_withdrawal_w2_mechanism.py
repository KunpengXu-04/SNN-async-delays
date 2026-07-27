"""Reproduce the post-W2 integer-boundary and saturation mechanism audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts import run_xor_delay_granularity_level1b as level1b
from scripts import run_xor_task_derived_timing_withdrawal as withdrawal


BASE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "xor_task_derived_timing_withdrawal_v1"
W0_ROOT = BASE / "runs" / "exploratory" / PROTOCOL_ID / "stage_w0_foundation" / "per_hidden_oracle_foundation"
W2_ROOT = BASE / "runs" / "exploratory" / PROTOCOL_ID / "stage_w2_withdrawal_recovery"
OUTPUT = BASE / "docs" / "generated" / PROTOCOL_ID / "stage_w2"
DELAYS = sorted(set(np.arange(2.5, 5.51, .25).tolist() + [4.9, 4.99, 4.999, 5.001, 5.01, 5.1]))


def global_scalar_gradient_rows(device: str = "cpu") -> list[dict[str, Any]]:
    protocol = withdrawal.load_protocol()
    adapted = withdrawal.level1b_protocol(protocol)
    rows: list[dict[str, Any]] = []
    for delay in DELAYS:
        for seed in protocol["optimization"]["formal_seeds"]:
            spec = {
                **withdrawal._branch_spec(protocol, int(seed), "task_delay_only", float(delay), updates=0),
                "granularity": "global",
                "delay_tying": "global",
                "independent_delay_parameters": 1,
            }
            model = level1b.build_model(adapted, spec)
            state = model.state_dict()
            parent_path = W0_ROOT / f"seed_{seed}" / "final_model.pt"
            parent = torch.load(parent_path, map_location=device, weights_only=True)
            state.update({
                key: value for key, value in parent.items()
                if key in state and state[key].shape == value.shape and "delay_raw" not in key
            })
            _, result = level1b.train_cell(
                adapted,
                spec,
                device=device,
                initial_state_dict=state,
                functional_delay_override=float(delay),
                trainable_components={"input_hidden_delays"},
                task_loss_weight=1.0,
            )
            gradient = float(np.asarray(result["initial_gradients"]["task"]).reshape(-1)[0])
            rows.append({
                "delay_steps": float(delay),
                "seed": int(seed),
                "raw_task_gradient": gradient,
                "target_directed": bool(gradient * (float(delay) - 4.0) > 0.0),
                "nonzero": bool(abs(gradient) > 1e-10),
                "exact_interface_patterns": int(result["history"]["exact_interface_patterns"][0]),
                "updates": 0,
                "formal_cell": False,
            })
    return rows


def d5_saturation_summary() -> dict[str, Any]:
    delays = []
    for path in sorted((W2_ROOT / "task_delay_only" / "perturb_5").glob("seed_*/plots/diagnostic_data.npz")):
        with np.load(path) as data:
            delays.extend(np.asarray(data["final_independent_delays"]).reshape(-1).tolist())
    values = np.asarray(delays, dtype=np.float64)
    return {
        "coordinates": int(values.size),
        "below_0p5": int(np.sum(values < .5)),
        "above_7p5": int(np.sum(values > 7.5)),
        "near_either_bound": int(np.sum((values < .5) | (values > 7.5))),
        "near_either_bound_fraction": float(np.mean((values < .5) | (values > 7.5))),
        "minimum_delay_steps": float(values.min()),
        "maximum_delay_steps": float(values.max()),
    }


def write_outputs(rows: list[dict[str, Any]]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "integer_boundary_gradient_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_delay = []
    for delay in DELAYS:
        selected = [row for row in rows if row["delay_steps"] == float(delay)]
        gradients = np.asarray([row["raw_task_gradient"] for row in selected])
        by_delay.append({
            "delay_steps": float(delay),
            "mean_raw_task_gradient": float(gradients.mean()),
            "minimum_raw_task_gradient": float(gradients.min()),
            "maximum_raw_task_gradient": float(gradients.max()),
            "target_directed_cells": int(sum(row["target_directed"] for row in selected)),
            "nonzero_cells": int(sum(row["nonzero"] for row in selected)),
        })
    summary = {
        "protocol_id": PROTOCOL_ID,
        "audit_class": "post_result_read_only_non_formal_mechanism_audit",
        "formal_result_changed": False,
        "seeds": sorted({row["seed"] for row in rows}),
        "global_scalar_gradient_field": by_delay,
        "d5_per_hidden_task_delay_saturation": d5_saturation_summary(),
        "integer_boundary_comparison": {
            "delay_4p999_mean_raw_gradient": next(
                row["mean_raw_task_gradient"] for row in by_delay if row["delay_steps"] == 4.999
            ),
            "delay_5p000_mean_raw_gradient": next(
                row["mean_raw_task_gradient"] for row in by_delay if row["delay_steps"] == 5.0
            ),
            "interpretation": "gradient_sign_flips_at_exact_integer_under_right_sided_floor_detach_backward",
        },
    }
    level1b._strict_write_json(OUTPUT / "mechanism_audit.json", summary)
    x = np.asarray([row["delay_steps"] for row in by_delay])
    mean = np.asarray([row["mean_raw_task_gradient"] for row in by_delay])
    low = np.asarray([row["minimum_raw_task_gradient"] for row in by_delay])
    high = np.asarray([row["maximum_raw_task_gradient"] for row in by_delay])
    fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    axis.plot(x, mean, marker="o", label="mean over five W0 seeds")
    axis.fill_between(x, low, high, alpha=.2, label="seed range")
    axis.axhline(0, color="black", linewidth=1)
    axis.axvline(4, color="tab:green", linestyle="--", label="target d4")
    axis.axvline(5, color="tab:red", linestyle=":", label="registered d5 boundary")
    axis.set(
        title="Post-W2 global-scalar task gradient field",
        xlabel="functional delay (steps)",
        ylabel="gradient with respect to raw delay",
    )
    axis.grid(alpha=.2)
    axis.legend(frameon=False)
    fig.savefig(OUTPUT / "integer_boundary_gradient_audit.png", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rows = global_scalar_gradient_rows(device=args.device)
    write_outputs(rows)
    print(json.dumps({"protocol_id": PROTOCOL_ID, "audit_rows": len(rows), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
